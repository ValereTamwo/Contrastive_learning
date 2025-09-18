import torch 
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer,ViTImageProcessor,ViTModel
from PIL import Image
import json
from tqdm import tqdm
import argparse
from accelerate import Accelerator
import os


# Architecture du Modele
class VisionTextAlignmentModel(nn.Module):
    def __init__(self,text_model_name, vision_model_name, vision_output_dim=768, projection_dim=4096):
        super().__init__()
        # Charger les modeles geles
        print("loading vision model")
        self.vision_model = ViTModel.from_pretrained(vision_model_name)
        print("Loading text model...")
        self.text_model = AutoModel.from_pretrained(text_model_name)

        # gelé les poids de ces models 
        for param in self.vision_model.parameters():
            param.required_grad = False
        for param in self.text_model.parameters():
            param.required_grad = False

        self.connector = nn.Sequential(
            nn.Linear(vision_output_dim, projection_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),                 
            nn.Linear(projection_dim * 2, projection_dim),
        )

        def encode_text(self,input_ids,attention_mask):
            outputs = self.text_model(input_ids = input_ids, attention_mask = attention_mask)
            text_features = outputs.last_hidden_state[:, 0, :]
            return text_features
        
        def encode_image(self, pixel_values, coordinates_batch):
            # encodage global de l'image 
            outputs = self.vision_model(pixel_values=pixel_values)
            feature_map = outputs.last_hidden_state[:, 1:, :]

            # La feature map doit être redimensionnée en 2D (ex: 14x14 pour ViT-B/16)
            patch_grid_size = int((feature_map.shape[1])**0.5)
            feature_map_2d = feature_map.view(feature_map.shape[0], patch_grid_size, patch_grid_size, -1)            
            #  Pooling par coordonnées 
            pooled_features = []
            for i, coordinates in enumerate(coordinates_batch):
                # Normaliser les coordonnées par rapport à la taille de la grille de patchs
                x_start = int(coordinates['left'] / self.vision_model.config.image_size * patch_grid_size)
                x_end = int(coordinates['right'] / self.vision_model.config.image_size * patch_grid_size)
                y_start = int(coordinates['top'] / self.vision_model.config.image_size * patch_grid_size)
                y_end = int(coordinates['bottom'] / self.vision_model.config.image_size * patch_grid_size)
                
                x_start = max(0, x_start)
                y_start = max(0, y_start)
                x_end = min(patch_grid_size, x_end + 1)
                y_end = min(patch_grid_size, y_end + 1)

                if x_start >= x_end or y_start >= y_end:
                    # Si la boîte est trop petite, prendre un patch par défaut
                    region_features = feature_map_2d[i, patch_grid_size//2, patch_grid_size//2, :]
                else:
                    region_features = feature_map_2d[i, y_start:y_end, x_start:x_end, :]
                
                pooled_feature = torch.mean(region_features.reshape(-1, region_features.shape[-1]), dim=0)
                pooled_features.append(pooled_feature)
                
            vision_features = torch.stack(pooled_features)

            return self.connector(vision_features)
        

class ContrastiveDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, image_processor):
        self.data = [json.loads(line) for line in open(jsonl_path, 'r')]
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.json_pathl = jsonl_path
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Charger et traiter l'image
        image_path = os.path.join(os.path.dirname(os.path.dirname(self.json_pathl)), item['screenshot_path'])
        try:
            image = Image.open(image_path).convert("RGB")
            processed_image = self.image_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        except Exception as e:
            print(f"Warning: Could not load image {image_path}. Returning dummy data. Error: {e}")
            processed_image = torch.zeros((3, 224, 224)) # Dummy image

        # Tokenizer le texte
        tokenized_text = self.tokenizer(
            item['parsed_text'],
            padding='max_length',
            truncation=True,
            max_length=128, 
            return_tensors="pt"
        )
        
        coords = item['coordinates']
        bbox = {
            'left': coords['left'], 
            'top': coords['top'], 
            'right': coords['right'], 
            'bottom': coords['bottom'],
            'width': coords['width'],
            'height': coords['height']
        }


        return {
            "pixel_values": processed_image,
            "input_ids": tokenized_text['input_ids'].squeeze(0),
            "attention_mask": tokenized_text['attention_mask'].squeeze(0),
            "coordinates": bbox
        }
