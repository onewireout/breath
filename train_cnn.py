import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SpectrogramCNN, self).__init__()
        
        # Input: RGB spectrogram, 3 x 224 x 224
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 112 x 112
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 56 x 56
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 28 x 28
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # 14 x 14
        
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 14 * 14, 512)
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main(data_dir='Training-Images', save_path='best_mel_cnn.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                             std=[0.229, 0.224, 0.225])  # ImageNet std
    ])

    if not os.path.isdir(data_dir):
        print(f"Directory {data_dir} not found. Please ensure it exists.")
        return

    # ImageFolder: one subfolder per class
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    print(f"Total dataset size: {len(dataset)} images")

    # 80/20 train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Training size: {train_size}")
    print(f"Validation size: {val_size}")

    # num_workers=0: multiprocessing overhead isn't worth it on small datasets
    batch_size = 64
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )

    # Class weights, in case the dataset is imbalanced
    targets = train_dataset.dataset.targets
    # Counted over the whole dataset, not just the train split
    class_counts = torch.bincount(torch.tensor(targets))
    total_samples = len(targets)
    # weight = total / (num_classes * class_count)
    class_weights = total_samples / (len(dataset.classes) * class_counts.float())
    
    # Missing a breath is worse than a false alarm, so weight it harder
    breath_idx = dataset.classes.index('Breath') if 'Breath' in dataset.classes else 0
    class_weights[breath_idx] *= 2.0
    
    class_weights = class_weights.to(device)
    print(f"Class Weights applied: {class_weights.tolist()}")

    model = SpectrogramCNN(num_classes=len(dataset.classes)).to(device)
    
    # Resume from an existing model if there is one
    is_finetuning = False
    base_lr = 0.0005
    if os.path.exists(save_path):
        print(f"\n--> Found existing model '{save_path}'. Loading weights to continue training...")
        model.load_state_dict(torch.load(save_path, map_location=device))
        is_finetuning = True
        
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Learning rate depends on whether we're fine-tuning
    current_lr = 0.0001 if is_finetuning else base_lr
    if is_finetuning:
        print(f"--> Fine-tuning mode: Lowering learning rate to {current_lr} to protect existing knowledge.")
    else:
        print(f"--> Training from scratch: Using base learning rate {current_lr}.")
        
    optimizer = optim.Adam(model.parameters(), lr=current_lr)

    # GradScaler for mixed precision (AMP)
    scaler = torch.amp.GradScaler(device.type) if device.type == 'cuda' else None

    num_epochs = 20
    best_val_f1 = 0.0
    patience = 5
    epochs_no_improve = 0

    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / train_size
        train_acc = 100 * correct_train / total_train
        
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        # TP/FP/FN for the Breath class, for F1
        # Breath is index 0 (ImageFolder sorts alphabetically)
        breath_idx = dataset.classes.index('Breath') if 'Breath' in dataset.classes else 0
        tp = 0
        fp = 0
        fn = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Autocast for validation too
                with torch.autocast(device_type=device.type):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                tp += ((predicted == breath_idx) & (labels == breath_idx)).sum().item()
                fp += ((predicted == breath_idx) & (labels != breath_idx)).sum().item()
                fn += ((predicted != breath_idx) & (labels == breath_idx)).sum().item()
                
        epoch_val_loss = val_loss / val_size
        val_acc = 100 * correct_val / total_val
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        val_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Target F1: {val_f1:.4f} (Acc: {val_acc:.2f}%)")
        
        # Keep the best model by F1, not accuracy
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best model with Target F1: {val_f1:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in F1 Score. Patience: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                break

    print("\nTraining complete!")
    print(f"Best validation F1 Score: {best_val_f1:.4f}")

    # Reload the best model for the confusion matrix
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            # Match the precision used in training
            with torch.autocast(device_type=device.type):
                outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    classes = dataset.classes
    num_classes = len(classes)
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for t, p in zip(all_labels, all_preds):
        cm[t][p] += 1
        
    print("\n--- Final Confusion Matrix (Validation Set) ---")
    header = f"{'True \\ Predicted':<20}" + "".join([f"{classes[i]:<15}" for i in range(num_classes)])
    print(header)
    print("-" * len(header))
    for i in range(num_classes):
        row_str = f"{classes[i]:<20}" + "".join([f"{cm[i][j]:<15}" for j in range(num_classes)])
        print(row_str)
        
    if 'Breath' in classes and len(classes) == 2:
        breath_idx = classes.index('Breath')
        speaking_idx = 1 - breath_idx
        tp = cm[breath_idx][breath_idx]
        fp = cm[speaking_idx][breath_idx]
        fn = cm[breath_idx][speaking_idx]
        print(f"\nBreath Detection Summary:")
        print(f"  True Positives  (Correctly identified Breaths)       : {tp}")
        print(f"  False Positives (Speaking mistaken for Breath)       : {fp}  <-- DANGER (Would delete speech)")
        print(f"  False Negatives (Missed Breaths mistaken as Speaking): {fn}  <-- ANNOYING (Leaves breath in)")


if __name__ == "__main__":
    main()
