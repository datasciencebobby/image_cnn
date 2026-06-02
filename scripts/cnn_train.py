import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import pandas as pd

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
                    
                    # convert YOLO format to pytorch format
                    x_min = (x_center - width / 2) * w
                    y_min = (y_center - height / 2) * h
                    x_max = (x_center + width / 2) * w
                    y_max = (y_center + height / 2) * h
                    
                    boxes.append([x_min, y_min, x_max, y_max])
                    
                    labels.append(int(class_id) + 1)
                    
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


def get_object_detection_model(num_classes):
    # load mobilenet_v3 model
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
    
    # replace model classifier head with desired number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def get_current_map(val_loader, device, model, metric):
# use torch.no_grad to validate model without updating weights
    model.eval()
    with torch.no_grad():

        # loop through batches of images/bounding boxes in val_loader
        for images, targets in val_loader:

            # load images to device
            images = list(image.to(device) for image in images)
            
            # load targets to device
            targets_formatted = []
            for t in targets:
                targets_formatted.append({
                    "boxes": t["boxes"].to(device),
                    "labels": t["labels"].to(device)
                })
                
            # run batch of images through model
            predictions = model(images)
        
            # update metric with batch prediction
            metric.update(predictions, targets_formatted)

    # compute metrics across batches
    metrics_result = metric.compute()
    current_map = metrics_result['map'].item()
    return current_map

def main(fp_dict, n_classes, batch_size, n_epochs, l_rate, loss_thresh):
    
    final_metrics = []

    model_save_path = fp_dict["model_save_path"]
    img_directory_train = fp_dict["img_train_dir"]
    label_directory_train = fp_dict["label_train_dir"]
    img_directory_val = fp_dict["img_val_dir"]
    label_directory_val = fp_dict["label_val_dir"]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # determine how many parallel processes can be used
    if device.type == "cuda":
        n_workers = 4
    else:
        n_workers = 2
    
    metric = MeanAveragePrecision(class_metrics=True)

    train_dataset = YOLODataset(img_dir=img_directory_train, label_dir=label_directory_train)

    val_dataset = YOLODataset(img_dir=img_directory_val, label_dir=label_directory_val)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=n_workers, 
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=n_workers, 
        collate_fn=collate_fn
    )

    model = get_object_detection_model(n_classes)
    model.to(device)
    
    if os.path.exists(model_save_path):
        print(f"Loading existing model from {model_save_path} to continue training...")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
    else:
        print("No previous weights found. Starting fresh from COCO base...")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=l_rate)

    print("Starting training...\n")

    best_map = get_current_map(val_loader, device, model, metric)

    for epoch in range(n_epochs):
        
        # set model to train state
        model.train()
        epoch_loss = 0
        
        # loop through batches of images and bounding boxes
        for i, (images, targets) in enumerate(train_loader):
            
            # load image and bounding boxes to the device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # run images and bounding boxes through model
            loss_dict = model(images, targets)

            # determine loss
            losses = sum(loss for loss in loss_dict.values())
            
            # zero gradients
            optimizer.zero_grad()

            # perform backpropagation
            losses.backward()

            # update weights based on optimizer
            optimizer.step()
            
            # add losses to epoch total loss
            epoch_loss += losses.item()
            
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{n_epochs}] | Batch [{i}/{len(train_loader)}] | Loss: {losses.item():.4f}\n")

        avg_epoch_loss = epoch_loss/len(train_loader)

        # switch to evaluation mode to run validation
        model.eval()
        
        print("Running validation inference...\n")
        
        # use torch.no_grad to validate model without updating weights
        with torch.no_grad():

            # loop through batches of images/bounding boxes in val_loader
            for images, targets in val_loader:

                # load images to device
                images = list(image.to(device) for image in images)
                
                # load targets to device
                targets_formatted = []
                for t in targets:
                    targets_formatted.append({
                        "boxes": t["boxes"].to(device),
                        "labels": t["labels"].to(device)
                    })
                    
                # run batch of images through model
                predictions = model(images)
            
                # update metric with batch prediction
                metric.update(predictions, targets_formatted)

        # compute metrics across batches
        metrics_result = metric.compute()
        current_map = metrics_result['map'].item()

        print(f"Mean Avg Precision (IoU=0.50:0.95): {metrics_result['map'].item():.4f}")
        print(f"Mean Avg Precision (IoU=0.50): {metrics_result['map_50'].item():.4f}")
        print(f"Mean Avg Recall (IoU=0.50:0.95, maxDets=100): {metrics_result['mar_100'].item():.4f}")
        print(f"mAP (IoU=0.75): {metrics_result['map_75'].item():.4f}")
        print("\n")

        # append validation results to metrics list
        final_metrics.append(metrics_result)

        # reset metric for next epoch
        metric.reset()

        # epoch summary
        print(f"Epoch {epoch+1}/{n_epochs} -> Train Loss: {avg_epoch_loss:.4f} | ")

        if avg_epoch_loss < loss_thresh:
            print(f"Early stopping triggered! Average loss ({avg_epoch_loss:.4f}) is below target threshold ({loss_thresh}).")
            break
        
        print(f"--- Epoch {epoch+1} Completed | Average Loss: {avg_epoch_loss:.4f} ---")

        if current_map > best_map:
            print(f"mAP improved from {best_map:.4f} to {current_map:.4f}. Saving model...")
            best_map = current_map
            torch.save(model.state_dict(), model_save_path)
        else:
            print(f"mAP did not improve from {best_map:.4f}. Model not saved.")
            print("\n")
    
    
    return pd.DataFrame(final_metrics)


if __name__ == '__main__':

    model_save_path = rf"C:\Users\{os.getlogin()}\Documents\image_cnn\models\lp_cnn.pt"
    
    img_directory_train = rf"C:\Users\{os.getlogin()}\Documents\image_cnn\data\formatted\license_plate_detection\random_train\images"
    
    label_directory_train = rf"C:\Users\{os.getlogin()}\Documents\image_cnn\data\formatted\license_plate_detection\random_train\labels"
    
    img_directory_val = rf"C:\Users\{os.getlogin()}\Documents\image_cnn\data\formatted\license_plate_detection\random_validate\images"
    
    label_directory_val = rf"C:\Users\{os.getlogin()}\Documents\image_cnn\data\formatted\license_plate_detection\random_validate\labels"
    
    fp_dict = {
        "model_save_path" : model_save_path,
        "img_train_dir" : img_directory_train,
        "label_train_dir": label_directory_train,
        "img_val_dir": img_directory_val,
        "label_val_dir": label_directory_val
    }
    
    run_metrics = main(
        fp_dict = fp_dict,
        n_classes=2, 
        batch_size=4,
        n_epochs=5, 
        l_rate=0.0001,
        loss_thresh=0.02
        )
    
    metric_fp = rf"C:\Users\{os.getlogin()}\Documents\image_cnn\training_results\metrics.csv"
    run_metrics.to_csv(metric_fp)