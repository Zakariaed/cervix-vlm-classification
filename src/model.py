import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoProcessor
import torch.nn.functional as F

class CervixVLMClassifier(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch32', num_classes=4, 
                 use_text_prompts=True, freeze_backbone=False):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.use_text_prompts = use_text_prompts
        
        # Load pre-trained VLM
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

class CervixVLMWithAttention(nn.Module):
    """Enhanced model with attention mechanism"""
    def __init__(self, model_name='openai/clip-vit-base-patch32', num_classes=4):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        self.feature_dim = self.clip_model.config.projection_dim
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # Classification head with residual connections
        self.fc1 = nn.Linear(self.feature_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, images):
        # Extract image features
        outputs = self.clip_model.vision_model(pixel_values=images)
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        
        # Apply attention
        attended_features, _ = self.attention(
            sequence_output, sequence_output, sequence_output
        )
        
        # Global average pooling
        pooled_features = attended_features.mean(dim=1)
        
        # Classification
        x = F.relu(self.fc1(pooled_features))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        
        return logits, attended_features
