import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score  
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 128
learning_rate1 = 1e-3
num_epochs_pretrain = 700
temperature = 0.5
projection_dim = 256

# Custom Augmentation for Gaussian Noise
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

# SimCLR Transform with Augmentations
class SimCLRTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            AddGaussianNoise(mean=0., std=0.05),
        ])
    
    def __call__(self, x):
        return self.transform(x), self.transform(x)

# Fine-Tuning Transform
finetune_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# Data Loading
train_dataset = datasets.ImageFolder(
    root='/uoa/scratch/users/t58am23/ThesisCodes/DataSets/train',
    transform=SimCLRTransform()
)
finetune_dataset = datasets.ImageFolder(
    root='/uoa/scratch/users/t58am23/ThesisCodes/DataSets/train',
    transform=finetune_transform
)
valid_dataset = datasets.ImageFolder(
    root='/uoa/scratch/users/t58am23/ThesisCodes/DataSets/val',
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
)
test_dataset = datasets.ImageFolder(
    root='/uoa/scratch/users/t58am23/ThesisCodes/DataSets/test',
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
finetune_loader = DataLoader(finetune_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# SimCLR Model Definition
class SimCLR(nn.Module):
    def __init__(self, base_model, projection_dim=256):
        super(SimCLR, self).__init__()
        self.encoder = base_model
        
        if hasattr(self.encoder, 'fc') and isinstance(self.encoder.fc, nn.Linear):
            in_features = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        else:
            raise ValueError("The base_model does not have an 'fc' layer of type nn.Linear")
        
        self.projection = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        z = F.normalize(z, dim=1)
        return z

# Contrastive Loss for SimCLR
def contrastive_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)
    
    similarity_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    
    labels = torch.arange(batch_size).to(device)
    labels = torch.cat([labels + batch_size, labels], dim=0)
    
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
    
    logits = similarity_matrix / temperature
    targets = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)], dim=0).to(device)
    
    loss = F.cross_entropy(logits, targets)
    return loss

# Initializing SimCLR Model
def get_resnet18():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    return model

base_model = get_resnet18()
simclr_model = SimCLR(base_model, projection_dim=projection_dim).to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for SimCLR")
    simclr_model = nn.DataParallel(simclr_model)

optimizer = optim.Adam(simclr_model.parameters(), lr=learning_rate1)

# Pretraining Loop
print("Starting self-supervised pretraining...")
simclr_model.train()
pretrain_losses = []
for epoch in range(num_epochs_pretrain):
    total_loss = 0
    for (x_i, x_j), _ in tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{num_epochs_pretrain}"):
        x_i = x_i.to(device)
        x_j = x_j.to(device)
        optimizer.zero_grad()
        z_i = simclr_model(x_i)
        z_j = simclr_model(x_j)
        loss = contrastive_loss(z_i, z_j, temperature)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    pretrain_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs_pretrain}], Loss: {avg_loss:.4f}")

torch.save(simclr_model.state_dict(), 'simclr_pretrained.pth')

print("Extracting features from validation set using pretrained SimCLR encoder...")
features = []
labels = []
with torch.no_grad():
    for images, lbls in tqdm(valid_loader, desc="Extracting features"):
        images = images.to(device)
        encoder = simclr_model.module.encoder if isinstance(simclr_model, nn.DataParallel) else simclr_model.encoder
        feat = encoder(images)
        features.append(feat.cpu().numpy())
        labels.append(lbls.numpy())
features = np.concatenate(features, axis=0)
labels = np.concatenate(labels, axis=0)

if len(features) > 1000:
    indices = np.random.choice(len(features), 1000, replace=False)
    features_subset = features[indices]
else:
    features_subset = features

silhouette_scores = []
k_range = range(5, 7)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    cluster_labels = kmeans.fit_predict(features_subset)
    silhouette_avg = silhouette_score(features_subset, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For k={k}, silhouette score: {silhouette_avg:.4f}")

optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters: {optimal_k}")

kmeans = KMeans(n_clusters=optimal_k, random_state=0)
cluster_labels = kmeans.fit_predict(features_subset)

# Kmeans Clustering
tsne_2d = TSNE(n_components=2, random_state=0)
features_2d = tsne_2d.fit_transform(features_subset)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                      c=cluster_labels, cmap='tab10', alpha=0.7)
cbar = plt.colorbar(scatter)
cbar.set_ticks(range(optimal_k))
cbar.set_ticklabels([f'Cluster {i}' for i in range(optimal_k)])
plt.title('K-means Clusters')
plt.savefig('simclr_pretrain_kmeans_2d.png')
plt.show()


# Confusion Matrix
if len(features) > 1000:
    labels_subset = labels[indices]
else:
    labels_subset = labels

cm = confusion_matrix(labels_subset, cluster_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues)
plt.title('Confusion Matrix SimCLR')
plt.colorbar()
tick_marks = np.arange(len(train_dataset.classes))
plt.xticks(np.arange(optimal_k), [f'Cluster {i}' for i in range(optimal_k)], rotation=45)
plt.yticks(tick_marks, train_dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix_simclr.png')
plt.close()

# Losses and Accuracies plot
plt.figure()
plt.plot(range(1, num_epochs_pretrain + 1), pretrain_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Pretraining Loss')
plt.savefig('simclr_pretrain_loss.png')
plt.show()