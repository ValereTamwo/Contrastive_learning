import json
import random
from typing import List, Dict
import os

def load_dataset(dataset_path: str) -> List[Dict]:
    """Load JSON dataset from file."""
    with open(dataset_path, 'r') as f:
        return json.load(f)

def generate_negative_pairs(dataset: List[Dict], num_negatives_per_positive: int = 5) -> List[Dict]:
    """
    Generate negative pairs for contrastive learning.
    Args:
        dataset: List of JSON entries with screenshot_path, element_id, parsed_text, coordinates.
        num_negatives_per_positive: Number of negative pairs to generate per positive pair.
    Returns:
        List of negative pair entries with similar structure to dataset.
    """
    negative_pairs = []
    
    # Group dataset by screenshot for efficiency
    screenshot_groups = {}
    for entry in dataset:
        screenshot = entry['screenshot_path']
        if screenshot not in screenshot_groups:
            screenshot_groups[screenshot] = []
        screenshot_groups[screenshot].append(entry)
    
    for idx, positive_entry in enumerate(dataset):
        screenshot = positive_entry['screenshot_path']
        element_id = positive_entry['element_id']
        parsed_text = positive_entry['parsed_text']
        coordinates = positive_entry['coordinates']
        
        # Generate negative pairs
        for _ in range(num_negatives_per_positive):
            strategy = random.choice([1, 2, 3, 4, 5])  # Randomly select a strategy
            
            if strategy == 1:  # Random HTML from same screenshot
                other_entries = [e for e in screenshot_groups[screenshot] if e['element_id'] != element_id]
                if other_entries:
                    other_entry = random.choice(other_entries)
                    negative_pairs.append({
                        'screenshot_path': screenshot,
                        'element_id': f"neg_{idx}_{_}",
                        'parsed_text': other_entry['parsed_text'],
                        'coordinates': coordinates,
                        'is_positive': False
                    })
            
            elif strategy == 2:  # Random HTML from different screenshot
                other_screenshots = [s for s in screenshot_groups.keys() if s != screenshot]
                if other_screenshots:
                    other_screenshot = random.choice(other_screenshots)
                    other_entry = random.choice(screenshot_groups[other_screenshot])
                    negative_pairs.append({
                        'screenshot_path': screenshot,
                        'element_id': f"neg_{idx}_{_}",
                        'parsed_text': other_entry['parsed_text'],
                        'coordinates': coordinates,
                        'is_positive': False
                    })
            
            elif strategy == 3:  # Randomized coordinates with correct HTML
                # Generate random coordinates within reasonable bounds (e.g., image size 1280x720)
                new_coords = {
                    'left': random.uniform(0, 1280),
                    'top': random.uniform(0, 720),
                    'width': random.uniform(10, 200),
                    'height': random.uniform(10, 200)
                }
                new_coords['right'] = new_coords['left'] + new_coords['width']
                new_coords['bottom'] = new_coords['top'] + new_coords['height']
                negative_pairs.append({
                    'screenshot_path': screenshot,
                    'element_id': f"neg_{idx}_{_}",
                    'parsed_text': parsed_text,
                    'coordinates': new_coords,
                    'is_positive': False
                })
            
            elif strategy == 4:  # Synthetic HTML mismatch
                tag, text = parsed_text.split('|')
                tag = tag.strip('<>')
                # Change tag or text randomly
                new_tag = random.choice(['a', 'button', 'input', 'span', 'svg', 'path'])
                new_text = text if random.random() > 0.5 else random.choice(['Click me', 'Submit', 'Link', ''])
                synthetic_html = f"<{new_tag}[{element_id}]| {new_text} >"
                negative_pairs.append({
                    'screenshot_path': screenshot,
                    'element_id': f"neg_{idx}_{_}",
                    'parsed_text': synthetic_html,
                    'coordinates': coordinates,
                    'is_positive': False
                })
            
            elif strategy == 5:  # Cross-website element swap
                same_tag_entries = [e for e in dataset if e['parsed_text'].startswith(f"<{parsed_text.split('|')[0].strip('<>')}") and e['screenshot_path'] != screenshot]
                if same_tag_entries:
                    other_entry = random.choice(same_tag_entries)
                    negative_pairs.append({
                        'screenshot_path': screenshot,
                        'element_id': f"neg_{idx}_{_}",
                        'parsed_text': other_entry['parsed_text'],
                        'coordinates': coordinates,
                        'is_positive': False
                    })
    
    return negative_pairs

def create_training_dataset(dataset: List[Dict], output_path: str, num_negatives_per_positive: int = 5):
    """Create a training dataset with positive and negative pairs."""
    for entry in dataset:
        entry['is_positive'] = True
    
    negative_pairs = generate_negative_pairs(dataset, num_negatives_per_positive)
    
    training_dataset = dataset + negative_pairs
    
    random.shuffle(training_dataset)
    
    with open(output_path, 'w') as f:
        json.dump(training_dataset, f, indent=2)
    
    print(f"Created training dataset with {len(dataset)} positive pairs and {len(negative_pairs)} negative pairs at {output_path}")

if __name__ == "__main__":
    dataset_path = "phase1_contrastive_dataset.jsonl"  
    output_path = "training_dataset.jsonl"
    
    dataset = load_dataset(dataset_path)
    
    create_training_dataset(dataset, output_path, num_negatives_per_positive=5)