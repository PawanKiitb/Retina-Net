import os
import torch
import cv2
import numpy as np
from torchvision import transforms as T
from models.retinaNet import create_model  # Adjust import as needed
from config import CONFIG  

# Utility: Draw boxes and labels
def draw_boxes(image, boxes, labels, scores, class_names, score_thresh=0.3):
    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        text = f"{class_names[label]}: {score:.2f}"
        cv2.putText(image, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return image

# Preprocessing for inference (must match your training normalization)
def preprocess_image(image):
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize if needed (match training size)
    # image = cv2.resize(image, (800, 800))  # Uncomment if needed
    # Convert to float and normalize
    image = image / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    # To tensor
    image = torch.from_numpy(image).permute(2,0,1).float()
    return image

def main():
    # 1. Load model
    model = create_model(CONFIG['NUM_CLASSES'])
    checkpoint = torch.load(CONFIG['CHECKPOINT_PATH'], map_location=CONFIG['DEVICE'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(CONFIG['DEVICE'])
    model.eval()

    # 2. Class names
    class_names = CONFIG.get('CLASSES', ['__background__', 'Pedestrian'])

    # 3. Input images
    input_dir = "data/imgs/test"  # Set your test images folder
    output_dir = "inference_results"
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]

    # 4. Inference loop
    for img_name in image_files:
        img_path = os.path.join(input_dir, img_name)
        orig_img = cv2.imread(img_path)
        image = preprocess_image(orig_img)
        image = image.unsqueeze(0).to(CONFIG['DEVICE'])

        with torch.no_grad():
            outputs = model(image)
        # outputs is a list of dicts (one per image)
        out = outputs[0]
        boxes = out['boxes'].cpu().numpy()
        labels = out['labels'].cpu().numpy()
        scores = out['scores'].cpu().numpy()

        # Draw predictions
        result_img = draw_boxes(orig_img.copy(), boxes, labels, scores, class_names, score_thresh=0.3)
        save_path = os.path.join(output_dir, img_name)
        cv2.imwrite(save_path, result_img)
        print(f"Saved: {save_path}")

    print("Inference complete. Results saved in:", output_dir)

if __name__ == "__main__":
    main()
