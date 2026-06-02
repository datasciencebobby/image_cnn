import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image

class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image and convert to PyTorch tensor [C, H, W] in range [0, 1]
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        image_tensor = F.to_tensor(image)
        
        label_name = img_name.replace('.jpg', '.txt')
        label_path = os.path.join(self.label_dir, label_name)
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    
                    # Convert YOLO [x_c, y_c, w, h] to PyTorch [x_min, y_min, x_max, y_max]
                    x_min = (x_center - width / 2) * w
                    y_min = (y_center - height / 2) * h
                    x_max = (x_center + width / 2) * w
                    y_max = (y_center + height / 2) * h
                    
                    boxes.append([x_min, y_min, x_max, y_max])
                    
                    labels.append(int(class_id) + 1)
                    
        # Handle images with no objects
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }
            
        return image_tensor, target

# Required to handle variable number of bounding boxes per image in a batch
def collate_fn(batch):
    return tuple(zip(*batch))

# ---------------------------------------------------------
# 2. Model Setup
# ---------------------------------------------------------
def get_object_detection_model(num_classes):
    # Load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one (tailored to your number of classes)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    if torch.device == "cuda":
        NUM_WORKERS = 2
    else:
        NUM_WORKERS = 0

    NUM_CLASSES = 2 
    BATCH_SIZE = 5
    NUM_EPOCHS = 2
    LEARNING_RATE = .000001
    LOSS_THRESHOLD = 0.02
    MODEL_SAVE_PATH = rf"C:\Users\{os.getlogin()}\Documents\image_cnn\models\custom_faster_rcnn.pt"
    
    img_directory = rf"C:\Users\{os.getlogin()}\Documents\image_cnn\data\formatted\license_plate_detection\random_train\images"
    label_directory = rf"C:\Users\{os.getlogin()}\Documents\image_cnn\data\formatted\license_plate_detection\random_train\labels"
    
    dataset = YOLODataset(img_dir=img_directory, label_dir=label_directory)
    data_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        collate_fn=collate_fn # Crucial for object detection
    )

    model = get_object_detection_model(NUM_CLASSES)
    model.to(device)
    
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading existing model from {MODEL_SAVE_PATH} to continue training...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    else:
        print("No previous weights found. Starting fresh from COCO base...")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    avg_epoch_loss = 1


    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        for i, (images, targets) in enumerate(data_loader):
            

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Batch [{i}/{len(data_loader)}] | Loss: {losses.item():.4f}")

        
        avg_epoch_loss = epoch_loss/len(data_loader)

        if avg_epoch_loss < LOSS_THRESHOLD:
            print(f"Early stopping triggered! Average loss ({avg_epoch_loss:.4f}) is below target threshold ({LOSS_THRESHOLD}).")
            break
                
        
        print(f"--- Epoch {epoch+1} Completed | Average Loss: {avg_epoch_loss:.4f} ---")

        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}\n")

        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}\n")


main()