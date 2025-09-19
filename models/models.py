
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, ViTModel
from PIL import Image
import json
import os

# Architecture du Modèle
class VisionTextAlignmentModel(nn.Module):
    def __init__(self, text_model_name, vision_model_name, projection_dim=4096):
        super().__init__()
        # Charger les modèles
        print("Loading vision model...")
        self.vision_model = ViTModel.from_pretrained(vision_model_name)
        print("Loading text model...")
        self.text_model = AutoModel.from_pretrained(text_model_name)


        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.text_model.parameters():
            param.requires_grad = False

        # Le connecteur projette les features de la vision vers l'espace du texte
        vision_output_dim = self.vision_model.config.hidden_size 
        self.connector = nn.Sequential(
            nn.Linear(vision_output_dim, projection_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),                 
            nn.Linear(projection_dim * 2, projection_dim),
        )
    
    def encode_text(self, input_ids, attention_mask):
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        # On utilise la représentation du dernier token, typique pour les modèles causaux comme Llama
        text_features = outputs.last_hidden_state[:, -1, :]
        return text_features
    
    def encode_image(self, pixel_values, coordinates_batch):
        batch_size = pixel_values.shape[0]
        
        outputs = self.vision_model(pixel_values=pixel_values)
        feature_map = outputs.last_hidden_state[:, 1:, :]

        # La carte de features doit être redimensionnée en 2D (ex: 14x14 pour ViT-B/16 sur une image 224x224)
        patch_grid_size = int((feature_map.shape[1])**0.5)
        feature_map_2d = feature_map.view(batch_size, patch_grid_size, patch_grid_size, -1)
        
        pooled_features = []
        # Itérer sur chaque élément du batch
        for i in range(batch_size):
     
            x_start = int(coordinates_batch['left'][i] / self.vision_model.config.image_size * patch_grid_size)
            x_end = int(coordinates_batch['right'][i] / self.vision_model.config.image_size * patch_grid_size)
            y_start = int(coordinates_batch['top'][i] / self.vision_model.config.image_size * patch_grid_size)
            y_end = int(coordinates_batch['bottom'][i] / self.vision_model.config.image_size * patch_grid_size)
            
            # S'assurer que les indices sont valides
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            x_end = min(patch_grid_size, x_end + 1)
            y_end = min(patch_grid_size, y_end + 1)

            if x_start >= x_end or y_start >= y_end:
                # Si la boîte est trop petite ou invalide, prendre un patch central par défaut
                region_features = feature_map_2d[i, patch_grid_size//2, patch_grid_size//2, :]
            else:
                # Extraire la région de la carte de features pour l'élément i
                region_features = feature_map_2d[i, y_start:y_end, x_start:x_end, :]
            
            # Appliquer un pooling (moyenne) sur la région extraite
            pooled_feature = torch.mean(region_features.reshape(-1, region_features.shape[-1]), dim=0)
            pooled_features.append(pooled_feature)
            
        vision_features = torch.stack(pooled_features)

        return self.connector(vision_features)

    def forward(self, pixel_values, input_ids, attention_mask, coordinates):
        image_features = self.encode_image(pixel_values, coordinates)
        text_features = self.encode_text(input_ids, attention_mask)
        return image_features, text_features


class ContrastiveDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, image_processor):
        self.data = [json.loads(line) for line in open(jsonl_path, 'r')]
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        # Pour reconstruire les chemins d'images correctement
        self.base_dir = os.path.dirname(jsonl_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
 
        image_path = os.path.join(self.base_dir, item['screenshot_path'])
        try:
            image = Image.open(image_path).convert("RGB")
            processed_image = self.image_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        except Exception as e:
            print(f"Warning: Could not load image {image_path}. Returning dummy data. Error: {e}")
            processed_image = torch.zeros((3, 224, 224)) # Image factice

        # Tokenizer le texte
        tokenized_text = self.tokenizer(
            item['parsed_text'],
            padding='max_length',
            truncation=True,
            max_length=128, 
            return_tensors="pt"
        )
        
        return {
            "pixel_values": processed_image,
            "input_ids": tokenized_text['input_ids'].squeeze(0),
            "attention_mask": tokenized_text['attention_mask'].squeeze(0),
            "coordinates": item['coordinates'] 
        }