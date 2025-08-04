import torch
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import pandas as pd

class Evaluator:
    def __init__(self, model, test_loader, class_names, device='cuda'):
        self.model = model
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def evaluate(self):
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                images = batch['image'].to(self.device)
                labels = batch['label']
                
                logits, _ = self.model(images)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Generate classification report
        report = classification_report(
            all_labels, 
            all_preds, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Calculate AUC for each class
        auc_scores = {}
        for i, class_name in enumerate(self.class_names):
            y_true = (all_labels == i).astype(int)
            y_score = all_probs[:, i]
            auc_scores[class_name] = roc_auc_score(y_true, y_score)
        
        return {
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'classification_report': report,
            'auc_scores': auc_scores
        }
    
    def plot_confusion_matrix(self, predictions, labels):
        cm = confusion_matrix(labels, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
    
    def plot_roc_curves(self, labels, probabilities):
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(self.class_names):
            y_true = (labels == i).astype(int)
            y_score = probabilities[:, i]
            
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for All Classes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves.png')
        plt.show()
