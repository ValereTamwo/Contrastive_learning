import json
import os

def create_contrastive_dataset(json_input_path: str, image_dir: str, output_file: str):
    """
    Transforme le JSON de données annotées en un dataset .jsonl pour
    l'apprentissage contrastif de la Phase 1.

    Args:
        json_input_path (str): Chemin vers le fichier JSON contenant raw_html, parsed_html, et items.
        image_dir (str): Chemin vers le dossier contenant le screenshot.
        output_file (str): Chemin vers le fichier .jsonl de sortie.
    """
    
    try:
        with open(json_input_path, 'r', encoding='utf-8') as f:
            annotated_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {json_input_path}: {e}")
        return

    base_name = os.path.splitext(os.path.basename(json_input_path))[0]
    screenshot_filename = "marked_image.png"
    screenshot_path = os.path.join(image_dir, screenshot_filename)

    if not os.path.exists(screenshot_path):
        print(f"Warning: Screenshot not found at {screenshot_path}. Skipping.")
        return

    items = annotated_data.get("items", [])
    if not items:
        print("No 'items' found in the JSON file.")
        return

    contrastive_samples = []
    
    for item in items:
        if not item.get("keep", False):
            continue
            
        element_id = item.get("id")
        rects = item.get("rects")
        tag = item.get("tag")
        text = item.get("text", "").strip().replace('\n', ' ')

        if not all([element_id, rects, tag]):
            continue


        parsed_text_representation = f'<{tag}[{element_id}]| {text} >'

        sample = {
            "screenshot_path": screenshot_path,
            "element_id": element_id,
            "parsed_text": parsed_text_representation,
            "coordinates": rects 
        }
        contrastive_samples.append(sample)

    with open(output_file, 'a', encoding='utf-8') as f:
        for sample in contrastive_samples:
            f.write(json.dumps(sample) + '\n')

    print(f"Added {len(contrastive_samples)} samples to {output_file} from {json_input_path}")


if __name__ == '__main__':

    
    
    root_data_dir = "." 
    final_dataset_path = "phase1_contrastive_dataset.jsonl"
    
    if os.path.exists(final_dataset_path):
        os.remove(final_dataset_path)

    for site_dir in os.listdir(root_data_dir):
        site_path = os.path.join(root_data_dir, site_dir)
        if os.path.isdir(site_path):
            for filename in os.listdir(site_path):
                if filename.endswith(".json"):
                    json_file_path = os.path.join(site_path, filename)
                    create_contrastive_dataset(
                        json_input_path=json_file_path,
                        image_dir=site_path,
                        output_file=final_dataset_path
                    )