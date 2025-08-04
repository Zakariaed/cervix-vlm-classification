import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=1e-6
        )
        
        # Initialize wandb
        if config['use_wandb']:
            wandb.init(project="cervix-vlm-classification", config=config)
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            logits, _ = self.model(images)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Store predictions
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(targets, predictions)
        f1 = f1_score(targets, predictions, average='weighted')
        
        return avg_loss, accuracy, f1
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits, _ = self.model(images)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(targets, predictions)
        f1 = f1_score(targets, predictions, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        return avg_loss, accuracy, f1, cm
    
    def train(self):
        best_val_acc = 0
        
        for epoch in range(self.config['epochs']):
            # Training
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_acc, val_f1, cm = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Logging
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            print(f"Confusion Matrix:\n{cm}")
            
            if self.config['use_wandb']:
                wandb.log({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'train_f1': train_f1,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, f'best_model.pth')
                print(f"Saved best model with validation accuracy: {val_acc:.4f}")

# Main training function
def main():
    config = {
        'data_dir': 'data/grid_cropped_cervix_images_74',
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'use_wandb': True,
        'model_name': 'openai/clip-vit-base-patch32',
        'freeze_backbone': False
    }
    
    # Create datasets
    train_dataset = CervixDataset(
        config['data_dir'], 
        split='train',
        transform=get_transforms('train')
    )
    
    val_dataset = CervixDataset(
        config['data_dir'],
        split='val',
        transform=get_transforms('val')
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = CervixVLMClassifier(
        model_name=config['model_name'],
        num_classes=4,
        use_text_prompts=True,
        freeze_backbone=config['freeze_backbone']
    )
    
    # Create trainer and train
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()

if __name__ == "__main__":
    main()
