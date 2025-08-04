#!/usr/bin/env python3
"""
Complete training script for Cervix VLM Classification
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import wandb
from transformers import CLIPModel, CLIPProcessor
import warnings
warnings.filterwarnings('ignore')

# ============== Dataset Class ==============
class CervixDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, test_size=0.2, random_state=42):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        # Class mapping
        self.classes = {
            'High squamous intra-epithelial lesion': 0,
            'Low squamous intra-epithelial lesion': 1,
            'Negative for Intraepithelial malignancy': 2,
            'Squamous cell carcinoma': 3
        }
        
        self.class_names = list(self.classes.keys())
        self.num_classes = len(self.classes)
        
        # Load data
        self.images, self.labels = self._load_data()
        
        # Split data
        if split in ['train', 'val']:
            X_train, X_val, y_train, y_val = train_test_split(
                self.images, self.labels, test_size=test_size, 
                random_state=random_state, stratify=self.labels
            )
            if split == 'train':
                self.images, self.labels = X_train, y_train
            else:
                self.images, self.labels = X_val, y_val
    
    def _load_data(self):
        images = []
        labels = []
        
        for class_name, class_idx in self.classes.items():
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.jpg'):
                        img_path = os.path.join(class_dir, img_name)
                        images.append(img_path)
                        labels.append(class_idx)
            else:
                print(f"Warning: Directory {class_dir} not found!")
        
        print(f"Loaded {len(images)} images across {len(self.classes)} classes")
        return images, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'path': img_path,
            'class_name': self.class_names[label]
        }

# Data augmentation and preprocessing
def get_transforms(split='train'):
    if split == 'train':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# ============== Model Class ==============
class CervixVLMClassifier(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch32', num_classes=4, 
                 use_text_prompts=True, freeze_backbone=False):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.use_text_prompts = use_text_prompts
        
        # Load pre-trained VLM
        print(f"Loading {model_name}...")
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # Get feature dimensions
        self.feature_dim = self.clip_model.config.projection_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Text prompts for each class
        self.text_prompts = [
            "High squamous intra-epithelial lesion cervical tissue",
            "Low squamous intra-epithelial lesion cervical tissue",
            "Normal cervical tissue negative for intraepithelial malignancy",
            "Squamous cell carcinoma cervical tissue"
        ]
        
    def forward(self, images, use_text_guidance=True):
        # Extract image features
        image_features = self.clip_model.get_image_features(pixel_values=images)
        image_features = F.normalize(image_features, p=2, dim=-1)
        
        if self.use_text_prompts and use_text_guidance:
            # Get text features for guidance
            text_inputs = self.processor(
                text=self.text_prompts, 
                return_tensors="pt", 
                padding=True
            ).to(images.device)
            
            text_features = self.clip_model.get_text_features(**text_inputs)
            text_features = F.normalize(text_features, p=2, dim=-1)
            
            # Compute similarity scores
            similarity = torch.matmul(image_features, text_features.T)
            
            # Combine with classification
            logits = self.classifier(image_features)
            
            # Weighted combination
            combined_logits = logits + 0.5 * similarity
            
            return combined_logits, similarity
        else:
            # Direct classification without text guidance
            logits = self.classifier(image_features)
            return logits, None

# ============== Trainer Class ==============
class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
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
        if config.get('use_wandb', False):
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
            
            # Forward pass
            logits, _ = self.model(images)
            loss = self.criterion(logits, labels)
            
            # Backward pass
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
            
            if self.config.get('use_wandb', False):
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
                os.makedirs('checkpoints', exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, 'checkpoints/best_model.pth')
                print(f"Saved best model with validation accuracy: {val_acc:.4f}")

# ============== Custom collate function for CLIP ==============
def clip_collate_fn(batch):
    """Custom collate function to handle CLIP preprocessing"""
    images = [item['image'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    
    # Convert images back to PIL for CLIP processor
    pil_images = []
    for img in images:
        # Denormalize
        img = img.clone()
        img[0] = img[0] * 0.229 + 0.485
        img[1] = img[1] * 0.224 + 0.456
        img[2] = img[2] * 0.225 + 0.406
        img = torch.clamp(img, 0, 1)
        
        # Convert to PIL
        img = transforms.ToPILImage()(img)
        pil_images.append(img)
    
    # Process with CLIP processor
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    processed = processor(images=pil_images, return_tensors="pt")
    
    return {
        'image': processed['pixel_values'],
        'label': labels
    }

# ============== Main Function ==============
def main():
    # Configuration
    config = {
        'data_dir': '/kaggle/input/grid-cropped-cervix-images-74/grid_cropped_cervix_images_74',
        'batch_size': 16,  # Reduced for memory efficiency
        'epochs': 30,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'use_wandb': False,  # Set to True if you have wandb configured
        'model_name': 'openai/clip-vit-base-patch32',
        'freeze_backbone': False
    }
    
    # Check if data directory exists
    if not os.path.exists(config['data_dir']):
        print(f"Error: Data directory {config['data_dir']} not found!")
        print("Please update the data_dir path in the config.")
        return
    
    # Create datasets
    print("Creating datasets...")
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
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=clip_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=clip_collate_fn
    )
    
    # Create model
    print("Creating model...")
    model = CervixVLMClassifier(
        model_name=config['model_name'],
        num_classes=4,
        use_text_prompts=True,
        freeze_backbone=config['freeze_backbone']
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer and train
    print("\nStarting training...")
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
