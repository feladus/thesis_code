import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.manifold import TSNE
import seaborn as sns
from PIL import Image
import timm
from tqdm import tqdm
import cv2  # Added for attention overlay

# ====================
# 0. Dataset Statistics
# ====================
def compute_dataset_stats(root_dir):
    """Compute mean and standard deviation of the dataset for normalization."""
    class TempDataset(Dataset):
        def __init__(self, root_dir):
            self.root_dir = root_dir
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()
            ])
            self.image_paths = []
            self.labels = []
            
            for label, class_dir in enumerate(sorted(os.listdir(root_dir))):
                class_path = os.path.join(root_dir, class_dir)
                self.image_paths += [os.path.join(class_path, f) for f in os.listdir(class_path)]
                self.labels += [label] * len(os.listdir(class_path))

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            return self.transform(Image.open(self.image_paths[idx])), self.labels[idx]

    temp_ds = TempDataset(root_dir)
    temp_loader = DataLoader(temp_ds, batch_size=64, num_workers=4)
    
    print("Computing dataset statistics...")
    mean, std, nb_samples = 0., 0., 0
    for images, _ in tqdm(temp_loader):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        nb_samples += batch_samples
        
    mean, std = mean/nb_samples, std/nb_samples
    print(f"Computed stats - Mean: {mean.item():.4f}, Std: {std.item():.4f}")
    return mean.item(), std.item()

stats_file = 'dataset_stats.pt'
if os.path.exists(stats_file):
    stats = torch.load(stats_file)
    dataset_mean, dataset_std = stats['mean'], stats['std']
else:
    dataset_mean, dataset_std = compute_dataset_stats('/uoa/scratch/users/t58am23/ThesisCodes/DataSets/train')
    torch.save({'mean': dataset_mean, 'std': dataset_std}, stats_file)

# ====================
# 1. Medical DINO Implementation
# ====================
class MedicalDINODataset(Dataset):
    """Custom dataset for Medical DINO with augmentations."""
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[dataset_mean], std=[dataset_std])
        ])
        self.image_paths = []
        self.labels = []
        
        for label, class_dir in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_dir)
            self.image_paths += [os.path.join(class_path, f) for f in os.listdir(class_path)]
            self.labels += [label] * len(os.listdir(class_path))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.image_paths[idx])), self.labels[idx]

class DINOHead(nn.Module):
    """Projection head for DINO model."""
    def __init__(self, in_dim, out_dim=2048):  # Reduced from 8192 to 2048 to limit overfitting
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 4096),
            nn.GELU(),
            nn.Linear(4096, out_dim),
        )
    def forward(self, x):
        return self.mlp(x)

class MedicalDINO(nn.Module):
    """Medical DINO model with student and teacher networks."""
    def __init__(self):
        super().__init__()
        # Student backbone using timm
        self.student_backbone = timm.create_model(
            'vit_small_patch16_224',
            pretrained=False,
            num_classes=0,  # Remove classification head
            in_chans=1  # For grayscale input
        )
        
        # Teacher backbone
        self.teacher_backbone = timm.create_model(
            'vit_small_patch16_224',
            pretrained=False,
            num_classes=0,
            in_chans=1
        )
        
        # Projection heads with reduced output dimension
        self.student_head = DINOHead(384, out_dim=2048)  # Reduced from 8192 to 2048
        self.teacher_head = DINOHead(384, out_dim=2048)  # Reduced from 8192 to 2048
        
        # Freeze teacher
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False
        for param in self.teacher_head.parameters():
            param.requires_grad = False

    def forward(self, x):
        student_feats = self.student_backbone(x)
        teacher_feats = self.teacher_backbone(x)
        return self.student_head(student_feats), self.teacher_head(teacher_feats)

# ====================
# 2. Training Setup
# ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = MedicalDINODataset('/uoa/scratch/users/t58am23/ThesisCodes/DataSets/train')
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
eval_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)  # For evaluation
model = MedicalDINO().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)  # Lowered from 0.0005 to 0.0001
warmup_epochs, total_epochs = 10, 300
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs - warmup_epochs)

# Temperature parameters
teacher_temperature = 0.04  # Lower teacher temperature
temperature = 0.1           # Student temperature

# ====================
# 3. Training Loop
# ====================
train_losses = []
for epoch in range(total_epochs):
    epoch_loss = 0.0
    model.train()
    
    # Warmup phase
    if epoch < warmup_epochs:
        lr_scale = (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001 * lr_scale  # Adjusted to new base LR
    
    # Training loop
    for images, _ in tqdm(loader, desc=f'Epoch {epoch+1}/{total_epochs}'):
        images = images.to(device)
        
        student_out, teacher_out = model(images)

        # Convert teacher outputs to probabilities
        teacher_out = teacher_out.detach() / teacher_temperature
        teacher_probs = torch.softmax(teacher_out, dim=-1)

        # Soften student predictions
        student_out = student_out / temperature
        student_log_probs = torch.log_softmax(student_out, dim=-1)
        
        # Compute DINO loss
        loss = -torch.mean(torch.sum(teacher_probs * student_log_probs, dim=-1))

        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item() * images.size(0)
    
    # EMA update for teacher network
    with torch.no_grad():
        m = 0.99  # Lowered from 0.996 to 0.99 for faster adaptation
        for s, t in zip(model.student_backbone.parameters(), model.teacher_backbone.parameters()):
            t.data.mul_(m).add_((1 - m) * s.detach().data)
        for s, t in zip(model.student_head.parameters(), model.teacher_head.parameters()):
            t.data.mul_(m).add_((1 - m) * s.detach().data)
    
    # Scheduler step and loss tracking
    if epoch >= warmup_epochs:
        scheduler.step()
    epoch_loss /= len(loader.dataset)
    train_losses.append(epoch_loss)
    print(f'Epoch {epoch+1}/{total_epochs} | Loss: {epoch_loss:.4f}')

# Save model and plot loss curve
torch.save(model.state_dict(), 'medical_dino.pth')
plt.plot(train_losses)
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('training_loss.png')
plt.close()