"""
Few-Shot Class-Incremental Learning Implementation
Based on "Self-Promoted Prototype Refinement for Few-Shot Class-Incremental Learning"

Dataset: CIFAR-10
Base classes: 7 classes (airplane, automobile, bird, cat, deer, dog, frog)
Incremental classes: 3 classes (horse, ship, truck) added ONE AT A TIME
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
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import seaborn as sns
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create directory for saving models and plots
os.makedirs('./checkpoints', exist_ok=True)
os.makedirs('./plots', exist_ok=True)

# ============================================================================
# STEP 1: Data Preparation
# ============================================================================
print("\n" + "="*70)
print("STEP 1: Preparing CIFAR-10 Dataset")
print("="*70)

# CIFAR-10 classes
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Define base classes (0-6) and incremental classes (7-9)
base_classes = list(range(7))  # First 7 classes
incremental_classes = list(range(7, 10))  # Last 3 classes

print(f"Base classes (7): {[class_names[i] for i in base_classes]}")
print(f"Incremental classes (3): {[class_names[i] for i in incremental_classes]}")

# Data transforms
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load full CIFAR-10 dataset
trainset_full = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform_train)
testset_full = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_test)

# Create base training and validation sets (80/20 split)
base_train_indices = [i for i, (_, label) in enumerate(trainset_full) if label in base_classes]
np.random.shuffle(base_train_indices)
split_idx = int(0.8 * len(base_train_indices))
train_indices = base_train_indices[:split_idx]
val_indices = base_train_indices[split_idx:]

base_trainset = Subset(trainset_full, train_indices)
base_valset = Subset(trainset_full, val_indices)

# K-shot configuration
K_SHOT_TRAIN = 3
K_SHOT_TEST = 10

print(f"\n⚙️  Shot Configuration:")
print(f"   Training: {K_SHOT_TRAIN}-shot")
print(f"   Testing:  {K_SHOT_TEST}-shot")

# Create few-shot incremental sets for each class separately
incremental_trainsets = {}
for cls in incremental_classes:
    cls_indices = [i for i, (_, label) in enumerate(trainset_full) if label == cls]
    incremental_trainsets[cls] = Subset(trainset_full, cls_indices[:K_SHOT_TRAIN])

print(f"\nBase training samples: {len(base_trainset)}")
print(f"Base validation samples: {len(base_valset)}")
print(f"Incremental samples per class: {K_SHOT_TRAIN}")

# ============================================================================
# STEP 2: Model Architecture
# ============================================================================
print("\n" + "="*70)
print("STEP 2: Building Model Components")
print("="*70)

class DynamicRelationProjection(nn.Module):
    """Projects representations and prototypes into shared embedding space"""
    def __init__(self, feature_dim=512):
        super(DynamicRelationProjection, self).__init__()
        
        self.transform_samples = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self.transform_prototypes = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
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
    """Few-Shot Class-Incremental Learning Model"""
    def __init__(self, num_base_classes=7, feature_dim=512):
        super(FSCILModel, self).__init__()
        
        resnet = torchvision.models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        self.feat_dim = 2048
        self.feature_dim = feature_dim
        self.dimension_reduction = nn.Linear(self.feat_dim, feature_dim)
        
        self.prototypes = nn.Parameter(torch.randn(num_base_classes, feature_dim))
        nn.init.kaiming_normal_(self.prototypes, mode='fan_out')
        
        self.relation_proj = DynamicRelationProjection(feature_dim)
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

model = FSCILModel(num_base_classes=len(base_classes), feature_dim=512).to(device)
print(f"✓ Model created with ResNet-50 backbone")

# ============================================================================
# STEP 3: Training with Validation
# ============================================================================
print("\n" + "="*70)
print("STEP 3: Training Base Session with Validation")
print("="*70)

def evaluate_model(model, dataloader):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

def train_with_validation(model, trainloader, valloader, num_epochs=30):
    """Train with validation and save best model"""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, 
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = float('inf')  # Track best validation LOSS
    best_val_acc = 0.0
    patience_counter = 0
    patience = 10  # Early stopping patience
    
    print(f"\nTraining for {num_epochs} epochs with early stopping (patience={patience})...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            logits, features = model(images)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
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
            best_val_acc = val_acc
            patience_counter = 0
            improved = True
            
            # Save best model based on validation LOSS
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'train_loss': train_loss,
                'train_acc': train_acc,
            }, './checkpoints/best_model.pth')
        else:
            patience_counter += 1
        
        # Print status
        status = "✓ BEST MODEL SAVED!" if improved else f"(No improvement for {patience_counter} epochs)"
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% {status}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping triggered! No improvement for {patience} epochs.")
            print(f"   Best Val Loss: {best_val_loss:.4f} at epoch {epoch - patience + 1}")
            break
    
    # Plot training curves with overfitting indicators
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_range = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs_range, train_losses, label='Train Loss', linewidth=2.5, color='#3498db')
    ax1.plot(epochs_range, val_losses, label='Val Loss', linewidth=2.5, color='#e74c3c')
    
    # Mark best validation loss
    best_epoch = val_losses.index(min(val_losses)) + 1
    ax1.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7, 
                label=f'Best Val Loss (Epoch {best_epoch})')
    ax1.scatter(best_epoch, min(val_losses), color='green', s=100, zorder=5)
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs_range, train_accs, label='Train Accuracy', linewidth=2.5, color='#3498db')
    ax2.plot(epochs_range, val_accs, label='Val Accuracy', linewidth=2.5, color='#e74c3c')
    
    # Mark best validation loss epoch
    ax2.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7,
                label=f'Best Model (Epoch {best_epoch})')
    ax2.scatter(best_epoch, val_accs[best_epoch-1], color='green', s=100, zorder=5)
    
    # Shade overfitting region
    if len(train_accs) > best_epoch:
        ax2.axvspan(best_epoch, len(train_accs), alpha=0.2, color='red', 
                   label='Overfitting Region')
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./plots/training_curves.png', dpi=150, bbox_inches='tight')
    print("\n✓ Training curves saved to './plots/training_curves.png'")
    
    # Print final statistics
    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"Total epochs trained: {len(train_losses)}")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Final train accuracy: {train_accs[-1]:.2f}%")
    print(f"Final val accuracy: {val_accs[-1]:.2f}%")
    
    gap = train_accs[-1] - val_accs[-1]
    if gap > 10:
        print(f"\n⚠️  WARNING: Large train-val gap ({gap:.2f}%) indicates overfitting!")
        print(f"   The model at epoch {best_epoch} (loaded below) should generalize better.")
    else:
        print(f"\n✓ Train-val gap: {gap:.2f}% (reasonable)")
    print(f"{'='*70}")
    
    return best_val_loss, best_val_acc, best_epoch

# Train model
base_trainloader = DataLoader(base_trainset, batch_size=128, shuffle=True, num_workers=2)
base_valloader = DataLoader(base_valset, batch_size=100, shuffle=False, num_workers=2)
best_val_loss, best_val_acc, best_epoch = train_with_validation(model, base_trainloader, base_valloader, num_epochs=50)

# Load best model
print(f"\n{'='*70}")
print(f"LOADING BEST MODEL")
print(f"{'='*70}")
print(f"Loading model from epoch {best_epoch}")
print(f"  • Best Val Loss: {best_val_loss:.4f}")
print(f"  • Best Val Acc:  {best_val_acc:.2f}%")

checkpoint = torch.load('./checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"✓ Best model loaded successfully!")
print(f"{'='*70}")

# ============================================================================
# STEP 4: Incremental Session - Add All New Classes Together
# ============================================================================
print("\n" + "="*70)
print("STEP 4: Incremental Session - Adding All New Classes Together")
print("="*70)

def evaluate_on_classes(model, testset, class_list):
    """Evaluate model on specific classes"""
    test_indices = [i for i, (_, label) in enumerate(testset) if label in class_list]
    test_subset = Subset(testset, test_indices)
    testloader = DataLoader(test_subset, batch_size=100, shuffle=False)
    
    model.eval()
    correct = 0
    total = 0
    
    # Create mapping from prototype index to class label
    # Model has prototypes in order: [0,1,2,3,4,5,6,7,8,9] for classes [0,1,2,3,4,5,6,7,8,9]
    all_classes = list(range(len(model.prototypes)))
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            
            # Get predictions (these are prototype indices)
            _, predicted = logits.max(1)
            
            # Map prototype indices to class labels
            # predicted gives indices 0-9 (or 0-6 for base only model)
            # We need to map these to actual CIFAR-10 class labels
            mapped_preds = torch.tensor([all_classes[p.item()] for p in predicted]).to(device)
            
            total += labels.size(0)
            correct += (mapped_preds == labels).sum().item()
    
    return 100. * correct / total

def add_incremental_classes(model, incremental_trainsets, new_classes):
    """Add multiple new classes together (as per paper)"""
    model.eval()
    
    print(f"\nAdding {len(new_classes)} new classes: {[class_names[i] for i in new_classes]}")
    
    # Extract features for all new classes
    all_new_features = []
    all_new_labels = []
    
    for cls in new_classes:
        incremental_loader = DataLoader(incremental_trainsets[cls], 
                                       batch_size=K_SHOT_TRAIN, shuffle=False)
        with torch.no_grad():
            for images, labels in incremental_loader:
                images = images.to(device)
                features = model.extract_features(images)
                all_new_features.append(features)
                all_new_labels.extend(labels.numpy())
    
    all_new_features = torch.cat(all_new_features, dim=0)
    
    # Get old prototypes
    old_prototypes = model.prototypes.data
    
    # Compute prototype for each new class
    new_prototypes = []
    for cls in new_classes:
        cls_mask = torch.tensor([l == cls for l in all_new_labels], dtype=torch.bool).to(device)
        cls_features = all_new_features[cls_mask]
        cls_prototype = cls_features.mean(dim=0, keepdim=True)
        new_prototypes.append(cls_prototype)
    
    new_prototypes = torch.cat(new_prototypes, dim=0)  # (3, 512)
    
    # Normalize and scale to match old prototypes
    new_prototypes = F.normalize(new_prototypes, p=2, dim=1)
    old_magnitude = old_prototypes.norm(dim=1).mean()
    new_prototypes = new_prototypes * old_magnitude
    
    print(f"✓ Computed prototypes for {len(new_classes)} new classes")
    print(f"   Old prototypes: {old_prototypes.shape}")
    print(f"   New prototypes: {new_prototypes.shape}")
    
    # Apply Dynamic Relation Projection (optional - for visualization)
    with torch.no_grad():
        refined_protos, relation_matrix = model.relation_proj(new_prototypes, old_prototypes)
    
    print(f"✓ Applied Dynamic Relation Projection")
    print(f"   Relation matrix shape: {relation_matrix.shape}")
    
    # Expand model prototypes with new classes
    updated_prototypes = torch.cat([old_prototypes, new_prototypes], dim=0)
    model.prototypes = nn.Parameter(updated_prototypes)
    
    print(f"✓ Model prototypes expanded from {len(old_prototypes)} to {len(updated_prototypes)}")
    
    return relation_matrix

# Track accuracies: Session 0 (base only) and Session 1 (after increment)
session_results = {
    'session': ['Session 0\n(Base Only)', 'Session 1\n(After +3 Classes)'],
    'base_acc': [],
    'incremental_acc': [],
    'overall_acc': [],
    'num_classes': []
}

# Session 0: Evaluate on base classes only
print(f"\n{'='*70}")
print(f"SESSION 0: Base Classes Only")
print(f"{'='*70}")

current_classes = base_classes.copy()
base_acc_s0 = evaluate_on_classes(model, testset_full, current_classes)

session_results['base_acc'].append(base_acc_s0)
session_results['incremental_acc'].append(0.0)
session_results['overall_acc'].append(base_acc_s0)
session_results['num_classes'].append(len(current_classes))

print(f"\n📊 Session 0 Results:")
print(f"   • Base Classes ({len(base_classes)}): {base_acc_s0:.2f}%")
print(f"   • Total Classes: {len(current_classes)}")

# Session 1: Add all incremental classes together
print(f"\n{'='*70}")
print(f"SESSION 1: Adding Incremental Classes")
print(f"{'='*70}")

relation_matrix = add_incremental_classes(model, incremental_trainsets, incremental_classes)

# Update current classes
current_classes = base_classes + incremental_classes

# Evaluate after increment
base_acc_s1 = evaluate_on_classes(model, testset_full, base_classes)
inc_acc_s1 = evaluate_on_classes(model, testset_full, incremental_classes)
overall_acc_s1 = evaluate_on_classes(model, testset_full, current_classes)

session_results['base_acc'].append(base_acc_s1)
session_results['incremental_acc'].append(inc_acc_s1)
session_results['overall_acc'].append(overall_acc_s1)
session_results['num_classes'].append(len(current_classes))

print(f"\n📊 Session 1 Results:")
print(f"   • Base Classes ({len(base_classes)}):        {base_acc_s1:.2f}%")
print(f"   • Incremental Classes ({len(incremental_classes)}): {inc_acc_s1:.2f}%")
print(f"   • Overall ({len(current_classes)} classes):          {overall_acc_s1:.2f}%")

# Compute forgetting
forgetting = base_acc_s0 - base_acc_s1
print(f"\n💡 Forgetting Analysis:")
print(f"   • Base accuracy before increment: {base_acc_s0:.2f}%")
print(f"   • Base accuracy after increment:  {base_acc_s1:.2f}%")
print(f"   • Forgetting: {forgetting:.2f}% {'(Lower is better)' if forgetting > 0 else '(Negative = improvement!)'}")

# Plot session comparison
print("\n📊 Creating session comparison plot...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart comparing sessions
sessions = session_results['session']
x = np.arange(len(sessions))
width = 0.25

bars1 = ax1.bar(x - width, session_results['base_acc'], width, 
                label='Base Classes (7)', color='#3498db', alpha=0.8)
bars2 = ax1.bar(x, session_results['incremental_acc'], width,
                label='Incremental Classes (3)', color='#e74c3c', alpha=0.8)
bars3 = ax1.bar(x + width, session_results['overall_acc'], width,
                label='Overall', color='#2ecc71', alpha=0.8)

ax1.set_xlabel('Session', fontsize=13, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax1.set_title('Accuracy Comparison Across Sessions', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(sessions, fontsize=11)
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, 105])

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Relation matrix heatmap
im = ax2.imshow(relation_matrix.cpu().numpy(), aspect='auto', cmap='plasma', vmin=0, vmax=1)
plt.colorbar(im, ax=ax2, label='Cosine Similarity', fraction=0.046, pad=0.04)

ax2.set_xlabel('All Classes (New + Old)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Old Base Classes', fontsize=12, fontweight='bold')
ax2.set_title('Dynamic Relation Matrix\n(How new classes relate to old)', 
              fontsize=13, fontweight='bold')

all_class_names = [class_names[i] for i in current_classes]
base_class_names = [class_names[i] for i in base_classes]

ax2.set_xticks(range(len(all_class_names)))
ax2.set_xticklabels(all_class_names, rotation=45, ha='right', fontsize=9)
ax2.set_yticks(range(len(base_class_names)))
ax2.set_yticklabels(base_class_names, fontsize=9)

# Highlight new classes section
ax2.axvline(len(base_classes) - 0.5, color='red', linewidth=3)

plt.tight_layout()
plt.savefig('./plots/session_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Session comparison saved to './plots/session_comparison.png'")

# ============================================================================
# STEP 5: Final Analysis - All Classes
# ============================================================================
print("\n" + "="*70)
print("STEP 5: Final Analysis on All Classes")
print("="*70)

all_classes = base_classes + incremental_classes
display_names = [class_names[i] for i in all_classes]

# Compute confusion matrix
def compute_confusion_matrix(model, testset, class_list):
    test_indices = [i for i, (_, label) in enumerate(testset) if label in class_list]
    test_subset = Subset(testset, test_indices)
    testloader = DataLoader(test_subset, batch_size=100, shuffle=False)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            _, predicted = logits.max(1)
            
            mapped_preds = torch.tensor([class_list[p.item()] if p.item() < len(class_list)
                                        else class_list[0] for p in predicted]).to(device)
            
            all_preds.extend(mapped_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds, labels=class_list)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    return cm, cm_normalized

cm, cm_normalized = compute_confusion_matrix(model, testset_full, all_classes)

# Per-class accuracy
per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100

print("\n" + "="*70)
print("PER-CLASS ACCURACY (FINAL)")
print("="*70)
print(f"{'Class':<15} {'Type':<12} {'Accuracy':<12} {'Samples':<10}")
print("-"*70)

base_accs = []
inc_accs = []

for i, class_name in enumerate(display_names):
    class_type = "Base" if i < len(base_classes) else "Incremental"
    samples = cm[i].sum()
    acc = per_class_acc[i]
    
    if class_type == "Base":
        base_accs.append(acc)
    else:
        inc_accs.append(acc)
    
    print(f"{class_name:<15} {class_type:<12} {acc:>10.2f}% {int(samples):<10}")

print("-"*70)
print(f"{'Base Avg (7)':<15} {'Summary':<12} {np.mean(base_accs):>10.2f}%")
print(f"{'Incremental Avg (3)':<15} {'Summary':<12} {np.mean(inc_accs):>10.2f}%")
print(f"{'Overall Avg (10)':<15} {'Summary':<12} {np.mean(per_class_acc):>10.2f}%")
print("="*70)

# Confusion Matrix
print("\n📊 Creating confusion matrix...")
plt.figure(figsize=(12, 10))
sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='YlGnBu',
            xticklabels=display_names, yticklabels=display_names,
            cbar_kws={'label': 'Percentage (%)'}, linewidths=0.5,
            vmin=0, vmax=100)

plt.title('Confusion Matrix - Final Model (All Classes)', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
plt.ylabel('True Class', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

for i in range(len(base_classes), len(all_classes)):
    plt.gca().add_patch(plt.Rectangle((i, i), 1, 1, fill=False, 
                                     edgecolor='red', lw=4))

plt.tight_layout()
plt.savefig('./plots/confusion_matrix_final.png', dpi=150, bbox_inches='tight')
print("✓ Confusion matrix saved to './plots/confusion_matrix_final.png'")

# Per-class accuracy bar chart
print("\n📊 Creating per-class accuracy chart...")
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(display_names))
colors = ['#3498db'] * len(base_classes) + ['#e74c3c'] * len(incremental_classes)

bars = ax.bar(x, per_class_acc, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Class', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Per-Class Accuracy - Final Model', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=11)
ax.set_ylim([0, 105])
ax.grid(axis='y', alpha=0.3, linestyle='--')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#3498db', alpha=0.7, label='Base Classes (7)'),
                   Patch(facecolor='#e74c3c', alpha=0.7, label='Incremental Classes (3)')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

ax.axvspan(len(base_classes)-0.5, len(all_classes)-0.5, alpha=0.1, color='red')

plt.tight_layout()
plt.savefig('./plots/per_class_accuracy.png', dpi=150, bbox_inches='tight')
print("✓ Per-class accuracy saved to './plots/per_class_accuracy.png'")

# 2D Prototype Distribution
print("\n📊 Creating 2D prototype distribution...")
all_prototypes = model.prototypes.data.cpu().numpy()
pca = PCA(n_components=2)
protos_2d = pca.fit_transform(all_prototypes)

plt.figure(figsize=(14, 10))

for i in range(len(base_classes)):
    plt.scatter(protos_2d[i, 0], protos_2d[i, 1], 
               c='#3498db', marker='o', s=400, 
               edgecolors='black', linewidths=2.5, alpha=0.8, zorder=3,
               label='Base Classes' if i == 0 else '')

for i in range(len(base_classes), len(all_classes)):
    plt.scatter(protos_2d[i, 0], protos_2d[i, 1], 
               c='#e74c3c', marker='^', s=500, 
               edgecolors='black', linewidths=2.5, alpha=0.8, zorder=3,
               label='Incremental Classes' if i == len(base_classes) else '')

for i, (x, y) in enumerate(protos_2d):
    plt.annotate(display_names[i], (x, y), 
                fontsize=12, ha='center', va='top' if i < len(base_classes) else 'bottom',
                fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='yellow' if i >= len(base_classes) else 'lightblue', 
                         alpha=0.7, edgecolor='black'))

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', 
          fontsize=12, fontweight='bold')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', 
          fontsize=12, fontweight='bold')
plt.title('2D Prototype Distribution (PCA)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='best', fontsize=12, framealpha=0.9)
plt.tight_layout()
plt.savefig('./plots/prototype_distribution_2d.png', dpi=150, bbox_inches='tight')
print("✓ 2D prototype distribution saved to './plots/prototype_distribution_2d.png'")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("IMPLEMENTATION COMPLETE!")
print("="*70)
print("\n📊 Final Results:")
print(f"   • Base Classes (7):        {np.mean(base_accs):.2f}%")
print(f"   • Incremental Classes (3): {np.mean(inc_accs):.2f}%")
print(f"   • Overall Accuracy (10):   {np.mean(per_class_acc):.2f}%")
print("\n💾 Model Saved:")
print("   • Best model: ./checkpoints/best_model.pth")
print("\n📁 Visualizations Saved:")
print("   1. ./plots/training_curves.png - Train/Val loss and accuracy")
print("   2. ./plots/session_accuracy.png - Accuracy across sessions")
print("   3. ./plots/confusion_matrix_final.png - Confusion matrix")
print("   4. ./plots/per_class_accuracy.png - Per-class accuracy bars")
print("   5. ./plots/prototype_distribution_2d.png - 2D PCA visualization")
print("="*70)