import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split

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
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.jpg'):
                    img_path = os.path.join(class_dir, img_name)
                    images.append(img_path)
                    labels.append(class_idx)
        
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
