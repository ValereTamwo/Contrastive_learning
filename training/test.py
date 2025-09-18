from models.models import  VisionTextAlignmentModel,ContrastiveDataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, ViTImageProcessor, ViTModel
from PIL import Image
import json
from tqdm import tqdm
import argparse
import os

@torch.no_grad()
def test_retrieval(args):
    """
    Évalue le connecteur entraîné sur une tâche de retrieval.
    """
    print("--- Starting Validation of the Trained Connector ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Charger le modèle et les poids du connecteur entraîné
    model = VisionTextAlignmentModel(
        text_model_name=args.text_model,
        vision_model_name=args.vision_model,
        projection_dim=args.projection_dim
    )
    
    try:
        model.connector.load_state_dict(torch.load(args.connector_path, map_location=device))
        print(f"Successfully loaded connector weights from {args.connector_path}")
    except FileNotFoundError:
        print(f"Error: Connector weights file not found at {args.connector_path}")
        return
        
    model.to(device)
    model.eval()

    # 2. Préparer le dataset de test
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    image_processor = ViTImageProcessor.from_pretrained(args.vision_model)
    test_dataset = ContrastiveDataset(args.test_dataset_path, args.data_root_dir, tokenizer, image_processor)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    all_image_features = []
    all_text_features = []

    print(f"Encoding all {len(test_dataset)} samples from the test set...")
    for batch in tqdm(test_dataloader, desc="Encoding test set"):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        coordinates_batch = []
        keys = batch['coordinates'].keys()
        for i in range(len(next(iter(batch['coordinates'].values())))):
            coordinates_batch.append({key: batch['coordinates'][key][i].item() for key in keys})

        image_features = model.encode_image(pixel_values, coordinates_batch)
        text_features = model.encode_text(input_ids, attention_mask)
        
        all_image_features.append(image_features.cpu())
        all_text_features.append(text_features.cpu())

    all_image_features = torch.cat(all_image_features, dim=0).to(device)
    all_text_features = torch.cat(all_text_features, dim=0).to(device)
    
    # Normaliser pour le calcul de similarité cosinus
    all_image_features /= all_image_features.norm(dim=1, keepdim=True)
    all_text_features /= all_text_features.norm(dim=1, keepdim=True)

    print("Calculating retrieval metrics...")
    similarity_matrix = all_image_features @ all_text_features.t()
    num_samples = similarity_matrix.shape[0]
    
    targets = torch.arange(num_samples, device=device)
    
    # Top-1 Accuracy
    top1_preds = similarity_matrix.argmax(dim=1)
    top1_accuracy = (top1_preds == targets).float().mean().item()
    
    # Top-5 Accuracy
    _, top5_preds = similarity_matrix.topk(5, dim=1)
    top5_accuracy = (top5_preds == targets.unsqueeze(1)).any(dim=1).float().mean().item()

    # Mean Reciprocal Rank (MRR)
    sorted_indices = similarity_matrix.argsort(dim=1, descending=True)
    ranks = (sorted_indices == targets.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
    mrr = (1.0 / ranks.float()).mean().item()

    print("\n" + "="*30)
    print("  Validation Results")
    print("="*30)
    print(f"Test set size: {num_samples} samples")
    print(f"Top-1 Accuracy: {top1_accuracy * 100:.2f}%")
    print(f"Top-5 Accuracy: {top5_accuracy * 100:.2f}%")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print("="*30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test the trained Vision-Text Connector.")
    parser.add_argument("--test_dataset_path", type=str, required=True, help="Path to the test .jsonl file (e.g., phase1_test_dataset.jsonl).")
    parser.add_argument("--data_root_dir", type=str, required=True, help="Path to the root directory where site folders (containing screenshots) are stored (e.g., 'raw_data/').")
    parser.add_argument("--connector_path", type=str, default="connector.pth", help="Path to the trained connector weights file.")
    parser.add_argument("--text_model", type=str, default="zai-org/webrl-llama-3.1-8b", help="Hugging Face path for the text model.")
    parser.add_argument("--vision_model", type=str, default="google/vit-base-patch16-224-in21k", help="Hugging Face path for the vision model.")
    parser.add_argument("--projection_dim", type=int, default=4096, help="Dimension of Llama-2-7b hidden states.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for encoding. Use the largest possible that fits in your VRAM.")
    
    args = parser.parse_args()
    test_retrieval(args)