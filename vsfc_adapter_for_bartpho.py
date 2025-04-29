import argparse
import time
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.strategies import DDPStrategy
from transformers import AutoTokenizer
from adapters import AutoAdapterModel, SeqBnConfig
from torchmetrics import Accuracy, F1Score
from mint.uit_vsfc_helpers import VSFCLoader, load_aivivn, AIVIVNDataset, AIVIVNLoader
from underthesea import word_tokenize
from torch.utils.data import random_split, DataLoader
import GPUtil

torch.set_float32_matmul_precision('high')


def bartpho_adapter_parse_args():
    """Parse command line arguments for Adapter fine-tuning"""
    parser = argparse.ArgumentParser(
        description="Adapter Fine-tuning for Vietnamese Sentiment Analysis"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['uit', 'aivi'],
        default='uit',
        help="Dataset to use: 'uit' for UIT-VSFC, 'aivi' for AIVIVN-2019"
    )
    parser.add_argument(
        "--adapter_size",
        type=int,
        default=64,
        help="Hidden dimension size of adapter (default: 64)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated list of GPU indices to use"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    return parser.parse_args()


class BARTphoAdapter(L.LightningModule):
    """Adapter-based Fine-tuning for BARTpho"""
    def __init__(self, num_labels=3, adapter_size=64, lr=2e-5):
        super().__init__()
        self.save_hyperparameters()
        # Load base adapter model
        self.model = AutoAdapterModel.from_pretrained("vinai/bartpho-word")

        # Compute reduction factor
        hidden_size = self.model.config.hidden_size
        reduction_factor = hidden_size // adapter_size
        if hidden_size % adapter_size != 0 or reduction_factor < 1:
            raise ValueError(
                f"Adapter size {adapter_size} must divide hidden size {hidden_size}."
            )

        # Configure and add adapter
        adapter_config = SeqBnConfig(
            mh_adapter=True,
            output_adapter=True,
            reduction_factor=reduction_factor,
            non_linearity="relu",
            original_ln_before=True
        )
        self.model.add_adapter("vsbc_adapter", config=adapter_config)
        self.model.train_adapter("vsbc_adapter")

        # Classification head
        self.classifier = torch.nn.Linear(self.model.config.d_model, num_labels)

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_labels, average='macro')
        self.test_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_labels, average='macro')

    def forward(self, input_ids, attention_mask):
        outs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=torch.zeros_like(input_ids),
            output_hidden_states=True
        )
        encoder_hidden = outs.encoder_hidden_states[-1]
        pooled = encoder_hidden[:, 0, :]
        return self.classifier(pooled)

    def training_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = torch.nn.functional.cross_entropy(logits, batch['labels'])
        self.train_acc(logits.argmax(dim=1), batch['labels'])
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc, prog_bar=True)
        if torch.cuda.is_available():
            for gpu in GPUtil.getGPUs(): self.log(f"GPU_{gpu.id}_mem", gpu.memoryUsed)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = torch.nn.functional.cross_entropy(logits, batch['labels'])
        self.val_acc(logits.argmax(dim=1), batch['labels'])
        self.val_f1(logits.argmax(dim=1), batch['labels'])
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_f1', self.val_f1, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        self.test_acc(logits.argmax(dim=1), batch['labels'])
        self.test_f1(logits.argmax(dim=1), batch['labels'])
        self.log('test_acc', self.test_acc, prog_bar=True)
        self.log('test_f1', self.test_f1, prog_bar=True)
        return {'test_acc': self.test_acc, 'test_f1': self.test_f1}

    def configure_optimizers(self):
        return torch.optim.AdamW(
            list(self.model.parameters())+list(self.classifier.parameters()),
            lr=self.hparams.lr,
            weight_decay=0.01
        )


if __name__ == '__main__':
    args = bartpho_adapter_parse_args()
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "vinai/bartpho-word", use_fast=False, model_max_length=256
    )
    # Data loading
    if args.dataset == 'aivi':
        texts, labels = load_aivivn('train')
        texts = [word_tokenize(t, format='text') for t in texts]
        full_ds = AIVIVNDataset(texts, labels, tokenizer, max_length=256)
        val_n = int(len(full_ds) * 0.1)
        tr_n = len(full_ds) - val_n
        tr_ds, val_ds = random_split(full_ds, [tr_n, val_n])
        train_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader  = AIVIVNLoader(tokenizer, batch_size=args.batch_size).load_data('test')
    else:
        loader = VSFCLoader(tokenizer, batch_size=args.batch_size)
        train_loader = loader.load_data('train')
        val_loader   = loader.load_data('val')
        test_loader  = loader.load_data('test')

    # Initialize model
    model = BARTphoAdapter(num_labels=3, adapter_size=args.adapter_size, lr=args.learning_rate)
    # Seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # GPUs & Trainer
    gpu_list = [int(g) for g in args.gpus.split(',')]
    trainer = L.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=gpu_list,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3)],
        precision='16-mixed',
        enable_progress_bar=True
    )
    # Train
    start = time.time()
    trainer.fit(model, train_loader, val_loader)
    if trainer.is_global_zero:
        print(f"Training time: {time.time() - start:.2f}s")
    # Test
    trainer.test(model, test_loader)