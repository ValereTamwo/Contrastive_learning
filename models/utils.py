import torch.nn as nn
import torch
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