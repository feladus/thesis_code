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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate2 = 1e-4
num_epochs_finetune = 20
batch_size = 128
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

# Classification Model Definition
class ClassificationModel(nn.Module):
    def __init__(self, simclr_model, num_classes):
        super(ClassificationModel, self).__init__()
        self.encoder = simclr_model.module.encoder if isinstance(simclr_model, nn.DataParallel) else simclr_model.encoder
        self.encoder.fc = nn.Identity()
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224).to(device)
            features = self.encoder(dummy_input)
            in_features = features.shape[1]
        self.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        features = self.encoder(x)
        out = self.classifier(features)
        return out

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

simclr_model.load_state_dict(torch.load('simclr_pretrained.pth', map_location=device))
simclr_model.eval()

num_classes = len(train_dataset.classes)
classification_model = ClassificationModel(simclr_model, num_classes).to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for Classification")
    classification_model = nn.DataParallel(classification_model)

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(classification_model.parameters(), lr=learning_rate2)


# Fine-Tuning Loop  zz
print("Starting fine-tuning for classification...")
finetune_losses = []
finetune_accuracies = []
val_losses = []
val_accuracies = []
patience = 5
delta = 0.005
best_val_loss = float('inf')
early_stop_counter = 0

for epoch in range(num_epochs_finetune):
    classification_model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(finetune_loader, desc=f"Fine-Tune Epoch {epoch+1}/{num_epochs_finetune}"):
        images = images.to(device)
        labels = labels.to(device)
        optimizer_ft.zero_grad()
        outputs = classification_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_ft.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(finetune_dataset)
    epoch_acc = correct / total
    finetune_losses.append(epoch_loss)
    finetune_accuracies.append(epoch_acc)
    
    classification_model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for val_images, val_labels in valid_loader:
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            val_outputs = classification_model(val_images)
            val_loss = criterion(val_outputs, val_labels)
            val_running_loss += val_loss.item() * val_images.size(0)
            _, val_predicted = torch.max(val_outputs, 1)
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()
    
    val_epoch_loss = val_running_loss / len(valid_dataset)
    val_epoch_acc = val_correct / val_total
    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_epoch_acc)
    
    print(f"Fine-Tune Epoch [{epoch+1}/{num_epochs_finetune}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")
    
    if val_epoch_loss < best_val_loss - delta:
        best_val_loss = val_epoch_loss
        early_stop_counter = 0
        torch.save(classification_model.state_dict(), 'best_classification_model.pth')
        print(f"Saved best model with validation loss: {best_val_loss:.4f}")
    else:
        early_stop_counter += 1
        print(f"No improvement in validation loss. Early stop counter: {early_stop_counter}/{patience}")
        if early_stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

classification_model.load_state_dict(torch.load('best_classification_model.pth'))
print("Loaded best model for evaluation.")

# Evaluation Function
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    return acc

val_acc = evaluate(classification_model, valid_loader)
test_acc = evaluate(classification_model, test_loader)
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")


plt.figure()
plt.plot(range(1, len(finetune_losses) + 1), finetune_losses, label='Train Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Fine-Tuning Loss')
plt.legend()
plt.savefig('simclr_finetune_loss.png')
plt.show()

plt.figure()
plt.plot(range(1, len(finetune_accuracies) + 1), finetune_accuracies, label='Train Acc')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Val Acc')
plt.axhline(y=test_acc, color='r', linestyle='--', label=f'Test Acc: {test_acc:.4f}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Fine-Tuning Accuracy')
plt.legend()
plt.savefig('simclr_finetune_accuracy.png')
plt.show()

classification_model.eval()
features = []
labels = []
with torch.no_grad():
    for images, lbls in valid_loader:
        images = images.to(device)
        encoder = classification_model.module.encoder if isinstance(classification_model, nn.DataParallel) else classification_model.encoder
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

# KMeans Clustering
tsne_2d = TSNE(n_components=2, random_state=0)
features_2d = tsne_2d.fit_transform(features_subset)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                      c=cluster_labels, cmap='tab10', alpha=0.7)
cbar = plt.colorbar(scatter)
cbar.set_ticks(range(optimal_k))
cbar.set_ticklabels([f'Cluster {i}' for i in range(optimal_k)])

plt.title('K-means Clusters (Finetuning)')
plt.savefig('simclr_kmeans_2d.png')
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
plt.savefig('confusion_matrix_finetune.png')
plt.show()

for i in range(10):
    print(i+2)