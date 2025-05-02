import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset
import seaborn as sns
from PIL import Image
import timm
from tqdm import tqdm
import cv2  

# 0. Dataset Statistics
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

# 1. DINO Implementation
class MedicalDINODataset(Dataset):
    """Custom dataset for DINO with augmentations."""
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
    def __init__(self, in_dim, out_dim=2048): 
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
        self.student_head = DINOHead(384, out_dim=2048)  
        self.teacher_head = DINOHead(384, out_dim=2048) 
        
        # Freeze teacher
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False
        for param in self.teacher_head.parameters():
            param.requires_grad = False

    def forward(self, x):
        student_feats = self.student_backbone(x)
        teacher_feats = self.teacher_backbone(x)
        return self.student_head(student_feats), self.teacher_head(teacher_feats)

# 2. Analysis Setup & Model Loading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# For analysis, we only need an evaluation dataset
dataset = MedicalDINODataset('/uoa/scratch/users/t58am23/ThesisCodes/DataSets/val')
eval_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

# 3. Initialize the model architecture
model = MedicalDINO().to(device)
model.load_state_dict(torch.load('medical_dino.pth', map_location=device))

# 4.  Clustering Analysis with KMeans
model.eval()

model.teacher_head.mlp[-1] = nn.Linear(4096, 2).to(device)
model.teacher_backbone = nn.Sequential(model.teacher_backbone, model.teacher_head)

features, true_labels = [], []
with torch.no_grad():
    for images, labels in eval_loader:
        feats = model.teacher_backbone(images.to(device)).cpu()
        features.append(feats)
        true_labels.append(labels)

features = torch.cat(features).numpy()
true_labels = torch.cat(true_labels).numpy()
image_paths = dataset.image_paths 

# Silhouette Analysis
k_values = [5, 6, 7, 8]
best_score, best_k = -1, k_values[0]
for k in k_values:
    cluster_labels = KMeans(n_clusters=k, random_state=42,).fit_predict(features)
    score = silhouette_score(features, cluster_labels)
    if score > best_score:
        best_score, best_k = score, k
    print(f'k={k}, Silhouette Score: {score:.4f}')
print(f'Best k: {best_k} with Silhouette Score: {best_score:.4f}')

# Fit KMeans with the best k
kmeans = KMeans(n_clusters=best_k, random_state=42)
pred_labels = kmeans.fit_predict(features)

# Pretrained clustering confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('pretrained_clusters.png')
plt.close()

# t-SNE Visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=5, n_jobs=1,)
tsne_results = tsne.fit_transform(features)
plt.figure(figsize=(12,8))
plt.scatter(tsne_results[:,0], tsne_results[:,1], c=pred_labels, cmap='tab10')
plt.colorbar(label='Cluster Label')
plt.title('t-SNE Clustering Visualization')
plt.savefig('pretrained_clusters.png')
plt.close()

# 5. Linear Probe Training (Supervised Evaluation on Pseudo-labels)
# Generates pseudo-labels by applying the kmeans model on teacher backbone features.
def get_features_and_pseudo_labels(dataset, model, kmeans, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    all_features = []
    for images, _ in dataloader:
        with torch.no_grad():
            feats = model.teacher_backbone(images.to(device))
        all_features.append(feats.cpu())
    all_features = torch.cat(all_features)
    pseudo_labels = kmeans.predict(all_features.numpy())
    return all_features, pseudo_labels

train_probe_dataset = MedicalDINODataset('/uoa/scratch/users/t58am23/ThesisCodes/DataSets/train')
val_probe_dataset = MedicalDINODataset('/uoa/scratch/users/t58am23/ThesisCodes/DataSets/val')

train_features, train_pseudo = get_features_and_pseudo_labels(train_probe_dataset, model, kmeans, batch_size=32)
val_features, val_pseudo = get_features_and_pseudo_labels(val_probe_dataset, model, kmeans, batch_size=32)

train_probe_dataset_fixed = TensorDataset(train_features, torch.tensor(train_pseudo, dtype=torch.long))
val_probe_dataset_fixed = TensorDataset(val_features, torch.tensor(val_pseudo, dtype=torch.long))
train_probe_loader = DataLoader(train_probe_dataset_fixed, batch_size=32, shuffle=True)
val_probe_loader = DataLoader(val_probe_dataset_fixed, batch_size=32, shuffle=False)

# Linear classifier 
feature_dim = train_features.shape[1]
num_classes = best_k
classifier = nn.Linear(feature_dim, num_classes).to(device)
optimizer_probe = torch.optim.Adam(classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

num_probe_epochs = 200
train_loss_list, val_loss_list = [], []
train_acc_list, val_acc_list = [], []

for epoch in range(num_probe_epochs):
    classifier.train()
    epoch_train_loss = 0.0
    train_correct = 0
    total = 0
    for features_batch, pseudo_labels_batch in train_probe_loader:
        features_batch = features_batch.to(device)
        pseudo_labels_batch = pseudo_labels_batch.to(device)
        logits = classifier(features_batch)
        loss = criterion(logits, pseudo_labels_batch)
        optimizer_probe.zero_grad()
        loss.backward()
        optimizer_probe.step()
        epoch_train_loss += loss.item() * features_batch.size(0)
        preds = torch.argmax(logits, dim=1)
        train_correct += (preds == pseudo_labels_batch).sum().item()
        total += features_batch.size(0)
    epoch_train_loss /= total
    train_acc = train_correct / total * 100
    train_loss_list.append(epoch_train_loss)
    train_acc_list.append(train_acc)
    
    classifier.eval()
    epoch_val_loss = 0.0
    val_correct = 0
    total_val = 0
    all_preds, all_true = [], []
    with torch.no_grad():
        for features_batch, pseudo_labels_batch in val_probe_loader:
            features_batch = features_batch.to(device)
            pseudo_labels_batch = pseudo_labels_batch.to(device)
            logits = classifier(features_batch)
            loss = criterion(logits, pseudo_labels_batch)
            epoch_val_loss += loss.item() * features_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            val_correct += (preds == pseudo_labels_batch).sum().item()
            total_val += features_batch.size(0)
            all_preds.append(preds.cpu())
            all_true.append(pseudo_labels_batch.cpu())
    epoch_val_loss /= total_val
    val_acc = val_correct / total_val * 100
    val_loss_list.append(epoch_val_loss)
    val_acc_list.append(val_acc)
    print(f"Linear Probe Epoch {epoch+1}/{num_probe_epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# Confusion Matrix for classification on validation set
all_preds_tensor = torch.cat(all_preds)
all_true_tensor = torch.cat(all_true)
cm_classification = confusion_matrix(all_true_tensor.numpy(), all_preds_tensor.numpy())
plt.figure(figsize=(10,8))
sns.heatmap(cm_classification, annot=True, fmt='d', cmap='Blues')
plt.title('Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
plt.close()

# Training and validation accuracy and loss curves
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(range(1, num_probe_epochs+1), train_acc_list, label='Train Acc')
plt.plot(range(1, num_probe_epochs+1), val_acc_list, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1, num_probe_epochs+1), train_loss_list, label='Train Loss')
plt.plot(range(1, num_probe_epochs+1), val_loss_list, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.tight_layout()
plt.savefig('probe_training_curves.png')
plt.close()

# Attention Visualization
def visualize_attention(image_path):
    img = Image.open(image_path).convert('L')  # Ensures grayscale
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[dataset_mean], std=[dataset_std])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Forward hook to capture attention from the final block (block 11)
    attention = None
    def hook(module, input, output):
        nonlocal attention
        attention = output[1]  # attn_weights
    
    hook_handle = model.student_backbone.keys[11].attn.register_forward_hook(hook)
    with torch.no_grad():
        model.student_backbone(img_tensor)
    hook_handle.remove()
    
    # Attention map
    attention = attention.mean(dim=1)[0, 0, 1:]  # [CLS] token attention
    attention = attention.reshape(14, 14).cpu().numpy()
    attention = cv2.resize(attention, (224, 224), interpolation=cv2.INTER_LINEAR)
    attention = (attention - attention.min()) / (attention.max() - attention.min())
    return img, attention

def overlay_attention(img, attention):
    """Overlay the attention map in green on the original grayscale image."""
    img_np = np.array(img)
    img_rgb = np.stack([img_np] * 3, axis=-1)
    heatmap = np.zeros((224, 224, 3), dtype=np.uint8)
    heatmap[:, :, 1] = (attention * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img_rgb, 0.5, heatmap, 0.5, 0)
    return overlay

# Attention for clusters in a grid
fig, axes = plt.subplots(2, best_k, figsize=(5*best_k, 10))
for cluster in range(best_k):
    cluster_indices = np.where(pred_labels == cluster)[0]
    center = kmeans.cluster_centers_[cluster]
    distances = np.linalg.norm(features[cluster_indices] - center, axis=1)
    closest_idx = cluster_indices[np.argmin(distances)]
    img, attn = visualize_attention(image_paths[closest_idx])
    
    axes[0, cluster].imshow(img, cmap='gray')
    axes[0, cluster].set_title(f'Cluster {cluster} - Original')
    axes[0, cluster].axis('off')
    
    overlay_img = overlay_attention(img, attn)
    axes[1, cluster].imshow(overlay_img)
    axes[1, cluster].set_title(f'Cluster {cluster} - Attention')
    axes[1, cluster].axis('off')

plt.tight_layout()
plt.savefig('cluster_attention.png')
plt.close()