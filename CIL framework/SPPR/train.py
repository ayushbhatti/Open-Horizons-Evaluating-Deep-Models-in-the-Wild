"""
Training Script for Few-Shot Class-Incremental Learning
Trains base model on 7 classes and saves best checkpoint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import os
import argparse

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create directories
os.makedirs('./checkpoints', exist_ok=True)
os.makedirs('./plots', exist_ok=True)

# ============================================================================
# Configuration
# ============================================================================
parser = argparse.ArgumentParser(description='Train FSCIL Base Model')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--patience', type=int, default=15, help='early stopping patience')
parser.add_argument('--n_way', type=int, default=3, help='N-way for RESS episodes')
parser.add_argument('--k_shot', type=int, default=5, help='K-shot for RESS episodes')
parser.add_argument('--use_ress', action='store_true', help='use Random Episode Selection Strategy')
args = parser.parse_args()

print(f"\n{'='*70}")
print("TRAINING CONFIGURATION")
print(f"{'='*70}")
print(f"Epochs: {args.epochs}")
print(f"Batch size: {args.batch_size}")
print(f"Learning rate: {args.lr}")
print(f"Dropout: {args.dropout}")
print(f"Patience: {args.patience}")
if args.use_ress:
    print(f"RESS: {args.n_way}-way {args.k_shot}-shot")
else:
    print(f"RESS: Disabled")
print(f"{'='*70}")

# ============================================================================
# Model Architecture
# ============================================================================
class DynamicRelationProjection(nn.Module):
    def __init__(self, feature_dim=512, dropout=0.3):
        super(DynamicRelationProjection, self).__init__()
        
        self.transform_samples = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
        self.transform_prototypes = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
    def forward(self, new_embeddings, old_prototypes):
        if new_embeddings.dim() == 2:
            new_embeddings = new_embeddings.unsqueeze(-1).unsqueeze(-1)
        if old_prototypes.dim() == 2:
            old_prototypes = old_prototypes.unsqueeze(-1).unsqueeze(-1)
        
        T_s = self.transform_samples(new_embeddings).squeeze(-1).squeeze(-1)
        T_p = self.transform_prototypes(old_prototypes).squeeze(-1).squeeze(-1)
        T_all = torch.cat([T_s, T_p], dim=0)
        
        T_p_norm = F.normalize(T_p, p=2, dim=1)
        T_all_norm = F.normalize(T_all, p=2, dim=1)
        relation_matrix = torch.mm(T_p_norm, T_all_norm.t())
        
        old_prototypes_flat = old_prototypes.squeeze(-1).squeeze(-1)
        refined_prototypes = torch.mm(relation_matrix, 
                                     torch.cat([new_embeddings.squeeze(-1).squeeze(-1), 
                                               old_prototypes_flat], dim=0))
        
        return refined_prototypes, relation_matrix

class FSCILModel(nn.Module):
    def __init__(self, num_base_classes=7, feature_dim=512, dropout=0.3):
        super(FSCILModel, self).__init__()
        
        resnet = torchvision.models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze early layers
        for param in list(self.feature_extractor.parameters())[:-30]:
            param.requires_grad = False
        
        self.feat_dim = 2048
        self.feature_dim = feature_dim
        
        self.dimension_reduction = nn.Sequential(
            nn.Linear(self.feat_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout)
        )
        
        self.prototypes = nn.Parameter(torch.randn(num_base_classes, feature_dim))
        nn.init.kaiming_normal_(self.prototypes, mode='fan_out')
        
        self.relation_proj = DynamicRelationProjection(feature_dim, dropout)
        self.scale = nn.Parameter(torch.tensor(10.0))
        
    def extract_features(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        features = self.dimension_reduction(features)
        return F.normalize(features, p=2, dim=1)
    
    def forward(self, x):
        features = self.extract_features(x)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)
        logits = self.scale * torch.mm(features, prototypes_norm.t())
        return logits, features

# ============================================================================
# Data Preparation
# ============================================================================
print("\n" + "="*70)
print("DATA PREPARATION")
print("="*70)

base_classes = list(range(7))
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Base classes (7): {[class_names[i] for i in base_classes]}")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset_full = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform_train)

# 80/20 train/val split
base_train_indices = [i for i, (_, label) in enumerate(trainset_full) if label in base_classes]
np.random.shuffle(base_train_indices)
split_idx = int(0.8 * len(base_train_indices))
train_indices = base_train_indices[:split_idx]
val_indices = base_train_indices[split_idx:]

trainset_full_noaug = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=transform_test)

base_trainset = Subset(trainset_full, train_indices)
base_valset = Subset(trainset_full_noaug, val_indices)

print(f"Training samples: {len(base_trainset)}")
print(f"Validation samples: {len(base_valset)}")

# ============================================================================
# Training Function
# ============================================================================
def random_episode_selection(base_classes, N_way):
    """Randomly select N classes to simulate removal"""
    selected_classes = random.sample(base_classes, N_way)
    remaining_classes = [c for c in base_classes if c not in selected_classes]
    return selected_classes, remaining_classes

def train_model(model, trainloader, valloader):
    """Train model with validation and early stopping"""
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
                                momentum=0.9, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\nStarting training...")
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            logits, features = model(images)
            loss = criterion(logits, labels)
            
            # Apply RESS if enabled
            if args.use_ress and batch_idx % 5 == 0:
                selected_classes, remaining_classes = random_episode_selection(
                    base_classes, args.n_way)
                
                proto_mask = torch.tensor([1 if i in remaining_classes else 0 
                                          for i in range(len(base_classes))], 
                                         dtype=torch.bool).to(device)
                
                if proto_mask.sum() > 0:
                    masked_logits = logits[:, proto_mask]
                    label_mapping = {old: new for new, old in enumerate(remaining_classes)}
                    valid_mask = torch.tensor([l.item() in remaining_classes for l in labels])
                    
                    if valid_mask.sum() > 0:
                        masked_labels = torch.tensor([label_mapping[l.item()] 
                                                     for l in labels[valid_mask]]).to(device)
                        episode_loss = criterion(masked_logits[valid_mask], masked_labels)
                        loss = loss + 0.3 * episode_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/(batch_idx+1), 'acc': 100.*correct/total})
        
        train_loss = running_loss / len(trainloader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                logits, _ = model(images)
                loss = criterion(logits, labels)
                val_running_loss += loss.item()
                
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_running_loss / len(valloader)
        val_acc = 100. * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step()
        
        # Check for improvement
        improved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            improved = True
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': vars(args)
            }, './checkpoints/best_model.pth')
        else:
            patience_counter += 1
        
        # Print status
        status = "✓ BEST" if improved else f"({patience_counter}/{args.patience})"
        gap = train_acc - val_acc
        print(f"Epoch {epoch+1:3d}: TrL={train_loss:.4f} TrA={train_acc:.2f}% | "
              f"VaL={val_loss:.4f} VaA={val_acc:.2f}% | Gap={gap:.2f}% | {status}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_range = range(1, len(train_losses) + 1)
    best_epoch = val_losses.index(min(val_losses)) + 1
    
    # Loss plot
    ax1.plot(epochs_range, train_losses, label='Train', linewidth=2.5, color='#3498db')
    ax1.plot(epochs_range, val_losses, label='Val', linewidth=2.5, color='#e74c3c')
    ax1.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, 
                label=f'Best (Epoch {best_epoch})')
    ax1.scatter(best_epoch, min(val_losses), color='green', s=150, zorder=5, 
                edgecolors='black', linewidths=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs_range, train_accs, label='Train', linewidth=2.5, color='#3498db')
    ax2.plot(epochs_range, val_accs, label='Val', linewidth=2.5, color='#e74c3c')
    ax2.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2)
    ax2.scatter(best_epoch, val_accs[best_epoch-1], color='green', s=150, zorder=5,
                edgecolors='black', linewidths=2)
    if len(train_accs) > best_epoch + 2:
        ax2.axvspan(best_epoch, len(train_accs), alpha=0.15, color='red', label='Overfitting')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./plots/training_curves.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Training curves saved to './plots/training_curves.png'")
    
    # Summary
    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best val acc: {val_accs[best_epoch-1]:.2f}%")
    print(f"Train-val gap: {train_accs[best_epoch-1] - val_accs[best_epoch-1]:.2f}%")
    print(f"{'='*70}")
    
    return best_val_loss, best_epoch

# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("INITIALIZING MODEL")
    print("="*70)
    
    model = FSCILModel(num_base_classes=len(base_classes), 
                      feature_dim=512, 
                      dropout=args.dropout).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    trainloader = DataLoader(base_trainset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=4, pin_memory=True)
    valloader = DataLoader(base_valset, batch_size=100, 
                          shuffle=False, num_workers=4, pin_memory=True)
    
    best_val_loss, best_epoch = train_model(model, trainloader, valloader)
    
    print(f"\n{'='*70}")
    print("✓ TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Best model saved: ./checkpoints/best_model.pth")
    print(f"To test: python test.py --checkpoint ./checkpoints/best_model.pth --k_shot 5")
    print(f"{'='*70}")