import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer, ViTImageProcessor, ViTModel
from PIL import Image
import json
from tqdm import tqdm
import argparse
from accelerate import Accelerator
import os
from models.models import VisionTextAlignmentModel, ContrastiveDataset
from models.utils import contrastive_loss

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
    # On entraîne uniquement les poids du connecteur
    optimizer = optim.AdamW(model.connector.parameters(), lr=args.learning_rate)

    # --- Préparation avec Accelerate ---
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # --- Boucle d'entraînement ---
    model.train()
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        for batch in progress_bar:
            with accelerator.accumulate(model):
                # Extraire les features en une seule passe avant (forward pass)
                image_features, text_features = model(
                    pixel_values=batch['pixel_values'],
                    coordinates=batch['coordinates'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )

                # Calculer la perte
                loss = contrastive_loss(image_features, text_features)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.is_local_main_process:
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
    parser.add_argument("--projection_dim", type=int, default=4096, help="Dimension of the text model's hidden states.")
    parser.add_argument("--output_path", type=str, default="connector.pth", help="Path to save the trained connector weights.")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    args = parser.parse_args()

    main(args)