import time
import argparse
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchmetrics import Accuracy, F1Score
import GPUtil
from mint.uit_vsfc_helpers import VSFCLoader, load_aivivn, AIVIVNDataset, AIVIVNLoader
from underthesea import word_tokenize
from torch.utils.data import random_split, DataLoader

torch.set_float32_matmul_precision('high')


def fft_parse_args():
    """
        Parse command line arguments for full fine-tuning of Vietnamese sentiment analysis models.
    """
    parser = argparse.ArgumentParser(
        description="Full Fine-tuning for Vietnamese Sentiment Analysis"
    )
    parser.add_argument(
        "--model",
        type=int,
        choices=[1, 2, 3, 4],
        required=True,
        help="Select model: 1=PhoBERT-base-v2, 2=PhoBERT-large, 3=BARTpho, 4=ViT5"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['uit', 'aivi'],
        default='uit',
        help="Dataset to use: 'uit' for UIT-VSFC, 'aivi' for AIVIVN-2019"
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
        help="Comma-separated list of GPU indices to use, e.g., '0,1,2' (default: '0')"
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


class FFT4VSA(L.LightningModule):
    """
        Full Fine-tuning for Vietnamese Sentiment Analysis
    """

    def __init__(self, model_name, num_labels, lr=2e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, ignore_mismatched_sizes=True
        )
        self.lr = lr

        self.train_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_labels, average='macro')
        self.test_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_labels, average='macro')

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        self.train_acc(preds, batch['labels'])
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc, prog_bar=True)

        if torch.cuda.is_available():
            for gpu in GPUtil.getGPUs():
                self.log(f"GPU_{gpu.id}", gpu.memoryUsed)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        self.val_acc(preds, batch['labels'])
        self.val_f1(preds, batch['labels'])
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_f1', self.val_f1, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        preds = torch.argmax(outputs.logits, dim=1)
        acc = self.test_acc(preds, batch['labels'])
        f1 = self.test_f1(preds, batch['labels'])
        self.log('test_acc', acc, prog_bar=True)
        self.log('test_f1', f1, prog_bar=True)
        return {'test_acc': acc, 'test_f1': f1}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


if __name__ == '__main__':
    args = fft_parse_args()

    # Map model choice to pretrained model name
    if args.model == 1:
        model_name = "vinai/phobert-base-v2"
    elif args.model == 2:
        model_name = "vinai/phobert-large"
    elif args.model == 3:
        model_name = "vinai/bartpho-word"
    elif args.model == 4:
        model_name = "VietAI/vit5-large"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Data loading
    if args.dataset == 'aivi':
        # Load full train set and split for validation
        texts, labels = load_aivivn('train')
        texts = [word_tokenize(t, format='text') for t in texts]
        full_ds = AIVIVNDataset(texts, labels, tokenizer, max_length=128)
        n_val = int(len(full_ds) * 0.1)
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(full_ds, [n_train, n_val])
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = AIVIVNLoader(tokenizer, batch_size=args.batch_size).load_data('test')
    else:
        loader = VSFCLoader(tokenizer, batch_size=args.batch_size)
        train_loader = loader.load_data(subset='train')
        val_loader = loader.load_data(subset='val')
        test_loader = loader.load_data(subset='test')

    # Initialize model
    model = FFT4VSA(model_name=model_name, num_labels=3, lr=args.learning_rate)
    print('\n\n')

    # Trainer setup
    GPUs = [int(g) for g in args.gpus.split(',')]
    callbacks = [EarlyStopping(monitor='val_loss', patience=3)] if 'val_loader' in locals() else []
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=GPUs,
        callbacks=callbacks
    )

    # Train
    start_time = time.time()
    if 'val_loader' in locals():
        trainer.fit(model, train_loader, val_loader)
    else:
        trainer.fit(model, train_loader)
    if trainer.is_global_zero:
        print(f"Training time: {time.time() - start_time:.2f} seconds")

    # Test
    trainer.test(model, test_loader)
