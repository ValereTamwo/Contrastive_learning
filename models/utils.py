import torch.nn as nn
import torch

def contrastive_loss(image_features, text_features, logit_scale): # Prend logit_scale en argument
    # Normaliser les features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # Le .clamp(max=100) est une sécurité pour éviter les valeurs extrêmes pendant l'entraînement
    scale = logit_scale.exp().clamp(max=100)

    # Calculer la similarité et appliquer l'échelle
    logits_per_image = scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    # Créer les labels
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)

    loss_i = nn.CrossEntropyLoss()(logits_per_image, labels)
    loss_t = nn.CrossEntropyLoss()(logits_per_text, labels)
    
    return (loss_i + loss_t) / 2.