import argparse
import time
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from torchmetrics import Accuracy, F1Score
from mint.uit_vsfc_helpers import VSFCLoader
import GPUtil

torch.set_float32_matmul_precision('high')

def lora_parse_args():
    """Parse command line arguments for QLoRA fine-tuning"""
    parser = argparse.ArgumentParser(
        description="QLoRA Fine-tuning for Vietnamese Sentiment Analysis"
    )
    parser.add_argument(
        "--model",
        type=int,
        choices=[1, 2, 3, 4],
        required=True,
        help="Model selection: 1=PhoBERT-base-v2, 2=PhoBERT-large, 3=BARTpho, 4=ViT5"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="Rank of LoRA matrices (default: 8)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="Alpha parameter for LoRA (default: 16)"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="Dropout rate for LoRA layers (default: 0.1)"
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


def get_4bit_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16
    )


def get_lora_config(model_name: str):
    target_modules_map = {
        'vinai/phobert-base-v2': ["query", "value"],
        'vinai/phobert-large': ["query", "value"],
        'vinai/bartpho-word': ["q_proj", "v_proj"],
        'VietAI/vit5-large': ["q", "v"] 
    }
    
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=target_modules_map[model_name],
        task_type="SEQ_CLS",
        inference_mode=False
    )


class QLoRA4VSA(L.LightningModule):
    def __init__(self, model_name, num_labels, lr=3e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Download model with 4-bit quantization
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            quantization_config=get_4bit_config(),
        )
        
        # 2. Convert to LoRA model
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, get_lora_config(model_name))
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_labels, average='macro')
        self.test_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_labels, average='macro')
        

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        
        self.train_acc(preds, batch['labels'])
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc, prog_bar=True)
        
        if torch.cuda.is_available():
            for gpu in GPUtil.getGPUs():
                self.log(f"GPU {gpu.id}", gpu.memoryUsed)
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
        
        self.test_acc(preds, batch['labels'])
        self.test_f1(preds, batch['labels'])
        self.log('test_acc', self.test_acc, prog_bar=True)
        self.log('test_f1', self.test_f1, prog_bar=True)
        return {'test_acc': self.test_acc, 'test_f1': self.test_f1}

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.hparams.lr,
            weight_decay=0.01
        )

    
if __name__ == "__main__":
    args = lora_parse_args()
    
    # Model mapping
    model_map = {
        1: "vinai/phobert-base-v2",
        2: "vinai/phobert-large",
        3: "vinai/bartpho-word",
        4: "VietAI/vit5-large"
    }
    
    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Donwload tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained(model_map[args.model])
    loader = VSFCLoader(tokenizer, batch_size=args.batch_size)
    train_loader = loader.load_data(subset='train')
    val_loader = loader.load_data(subset='val')
    test_loader = loader.load_data(subset='test')
    

    # Setup model
    model = QLoRA4VSA(
        model_name=model_map[args.model],
        num_labels=3,
        lr=args.learning_rate
    )
    
    # Setup GPUs
    GPUs = [int(gpu) for gpu in args.gpus.split(',')]

    # Setup trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=GPUs,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3)],
        precision="bf16-mixed"
    )
    
    # Training
    if trainer.is_global_zero:
        start_time = time.time()

    trainer.fit(model, train_loader, val_loader)
    
    if trainer.is_global_zero:
        end_time = time.time()
        print(f"Training time: {end_time - start_time} seconds")
    
    # Evaluation
    trainer.test(model, test_loader)
