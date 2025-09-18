# 2_train_connector.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, ViTImageProcessor, ViTModel
from PIL import Image
import json
from tqdm import tqdm
import argparse
from accelerate import Accelerator

# ==============================================================================
#  1. Définition de l'Architecture du Modèle
# ==============================================================================

class VisionTextAlignmentModel(nn.Module):
    def __init__(self, text_model_name, vision_model_name, vision_output_dim=768, projection_dim=4096):
        super().__init__()
        # --- Modèles Pré-entraînés (Gelés) ---
        print("Loading vision model...")
        self.vision_model = ViTModel.from_pretrained(vision_model_name)
        print("Loading text model...")
        self.text_model = AutoModel.from_pretrained(text_model_name)
        
        # Geler les poids de ces modèles
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.text_model.parameters():
            param.requires_grad = False
            
        # --- Connecteur (Partie Entraînable) ---
        self.connector = nn.Sequential(
            nn.Linear(vision_output_dim, projection_dim * 2),
            nn.ReLU(),
            nn.Linear(projection_dim * 2, projection_dim)
        )

    def encode_text(self, input_ids, attention_mask):
        # Utiliser les embeddings de la dernière couche cachée
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        # Prendre le vecteur de l'embedding [CLS]
        text_features = outputs.last_hidden_state[:, 0, :]
        return text_features

    def encode_image(self, pixel_values, coordinates_batch):
        # 1. Encodage Global de l'image
        # La sortie est (batch_size, num_patches + 1, hidden_size), ex: (B, 197, 768)
        outputs = self.vision_model(pixel_values=pixel_values)
        feature_map = outputs.last_hidden_state[:, 1:, :]  # Ignorer le token [CLS]
        
        # La feature map doit être redimensionnée en 2D (ex: 14x14 pour ViT-B/16)
        patch_grid_size = int((feature_map.shape[1])**0.5)
        feature_map_2d = feature_map.view(feature_map.shape[0], patch_grid_size, patch_grid_size, -1)
        
        # 2. Pooling par coordonnées (RoI Pooling simplifié)
        pooled_features = []
        for i, coordinates in enumerate(coordinates_batch):
            # Normaliser les coordonnées par rapport à la taille de la grille de patchs
            x_start = int(coordinates['left'] / self.vision_model.config.image_size * patch_grid_size)
            x_end = int(coordinates['right'] / self.vision_model.config.image_size * patch_grid_size)
            y_start = int(coordinates['top'] / self.vision_model.config.image_size * patch_grid_size)
            y_end = int(coordinates['bottom'] / self.vision_model.config.image_size * patch_grid_size)
            
            # S'assurer que les indices sont valides
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            x_end = min(patch_grid_size, x_end + 1)
            y_end = min(patch_grid_size, y_end + 1)

            if x_start >= x_end or y_start >= y_end:
                 # Si la boîte est trop petite, prendre un patch par défaut
                region_features = feature_map_2d[i, patch_grid_size//2, patch_grid_size//2, :]
            else:
                region_features = feature_map_2d[i, y_start:y_end, x_start:x_end, :]
            
            # Agréger les features de la région (simple moyenne)
            pooled_feature = torch.mean(region_features.reshape(-1, region_features.shape[-1]), dim=0)
            pooled_features.append(pooled_feature)
            
        vision_features = torch.stack(pooled_features)
        
        # 3. Passer à travers le connecteur
        return self.connector(vision_features)

# ==============================================================================
#  2. Dataset et Dataloader
# ==============================================================================

class ContrastiveDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, image_processor):
        self.data = [json.loads(line) for line in open(jsonl_path, 'r')]
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Charger et traiter l'image
        image_path = os.path.join(os.path.dirname(os.path.dirname(jsonl_path)), item['screenshot_path'])
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
            max_length=128, # Longueur max pour un snippet de code
            return_tensors="pt"
        )
        
        # Extraire les coordonnées
        coords = item['coordinates']
        # Playwright donne x,y,width,height. Convertissons en left,top,right,bottom
        bbox = {
            'left': coords['x'], 
            'top': coords['y'], 
            'right': coords['x'] + coords['width'], 
            'bottom': coords['y'] + coords['height']
        }

        return {
            "pixel_values": processed_image,
            "input_ids": tokenized_text['input_ids'].squeeze(0),
            "attention_mask": tokenized_text['attention_mask'].squeeze(0),
            "coordinates": bbox
        }

# ==============================================================================
#  3. Logique d'Entraînement
# ==============================================================================

def contrastive_loss(image_features, text_features, temperature=0.07):
    # Normaliser les features pour le calcul de la similarité cosinus
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # Matrice de similarité cosinus
    logit_scale = (1 / temperature).exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    # Créer les labels (la diagonale est la paire positive)
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)

    loss_i = nn.CrossEntropyLoss()(logits_per_image, labels)
    loss_t = nn.CrossEntropyLoss()(logits_per_text, labels)
    
    return (loss_i + loss_t) / 2

def main(args):
    accelerator = Accelerator()
    
    # --- Modèles et Tokenizers ---
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    image_processor = ViTImageProcessor.from_pretrained(args.vision_model)
    model = VisionTextAlignmentModel(
        text_model_name=args.text_model,
        vision_model_name=args.vision_model,
        projection_dim=args.projection_dim
    )

    # --- Dataset ---
    dataset = ContrastiveDataset(args.dataset_path, tokenizer, image_processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # --- Optimiseur ---
    optimizer = optim.AdamW(model.connector.parameters(), lr=args.learning_rate)
    
    # --- Préparation avec Accelerate ---
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # --- Boucle d'entraînement ---
    model.train()
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        for batch in progress_bar:
            with accelerator.accumulate(model):
                # Extraire les features
                image_features = model.encode_image(batch['pixel_values'], batch['coordinates'])
                text_features = model.encode_text(batch['input_ids'], batch['attention_mask'])

                # Calculer la perte
                loss = contrastive_loss(image_features, text_features)
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            progress_bar.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

    # --- Sauvegarde du Connecteur ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.connector.state_dict(), args.output_path)
        print(f"Connector saved to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the Vision-Text Connector.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the phase1_dataset.jsonl file.")
    parser.add_argument("--text_model", type=str, default="zai-org/webrl-llama-3.1-8b", help="Hugging Face path for the text model.")
    parser.add_argument("--vision_model", type=str, default="google/vit-base-patch16-224-in21k", help="Hugging Face path for the vision model.")
    parser.add_argument("--projection_dim", type=int, default=4096, help="Dimension of Llama-2-7b hidden states.")
    parser.add_argument("--output_path", type=str, default="connector.pth", help="Path to save the trained connector weights.")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    args = parser.parse_args()
    
    main(args)