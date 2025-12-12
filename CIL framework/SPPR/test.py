"""
Testing Script for Few-Shot Class-Incremental Learning
Loads trained base model, adds incremental classes, and evaluates
Can be run multiple times with different configurations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import os
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# Configuration
# ============================================================================
parser = argparse.ArgumentParser(description='Test FSCIL Model on Incremental Classes')
parser.add_argument('--checkpoint', type=str, required=True, 
                    help='path to trained model checkpoint')
parser.add_argument('--k_shot', type=int, default=5, 
                    help='K-shot for incremental classes (how many samples per new class)')
parser.add_argument('--output_dir', type=str, default='./results',
                    help='directory to save test results')
args = parser.parse_args()

print(f"\n{'='*70}")
print("TESTING CONFIGURATION")
print(f"{'='*70}")
print(f"Checkpoint: {args.checkpoint}")
print(f"K-shot (incremental): {args.k_shot}")
print(f"Output directory: {args.output_dir}")
print(f"{'='*70}")

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# ============================================================================
# Model Architecture (Same as training)
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
# Load Data - ONLY FOR TESTING
# ============================================================================
print("\n" + "="*70)
print("LOADING TEST DATA")
print("="*70)

base_classes = list(range(7))
incremental_classes = list(range(7, 10))
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Base classes (7): {[class_names[i] for i in base_classes]}")
print(f"Incremental classes (3): {[class_names[i] for i in incremental_classes]}")
print(f"K-shot per incremental class: {args.k_shot}")

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset_full = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform_test)
testset_full = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_test)

# Create K-shot incremental datasets
incremental_trainsets = {}
for cls in incremental_classes:
    cls_indices = [i for i, (_, label) in enumerate(trainset_full) if label == cls]
    incremental_trainsets[cls] = Subset(trainset_full, cls_indices[:args.k_shot])

print(f"✓ Loaded {args.k_shot} samples per incremental class")

# ============================================================================
# Load Trained Model
# ============================================================================
print("\n" + "="*70)
print("LOADING TRAINED MODEL")
print("="*70)

if not os.path.exists(args.checkpoint):
    print(f"❌ Error: Checkpoint not found at {args.checkpoint}")
    print("   Please train a model first using: python train.py")
    exit(1)

checkpoint = torch.load(args.checkpoint)
config = checkpoint.get('config', {})

model = FSCILModel(
    num_base_classes=checkpoint.get('num_base_classes', 7),
    feature_dim=512,
    dropout=config.get('dropout', 0.3)
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # CRITICAL: Set to eval mode (disables dropout)

print(f"✓ Model loaded successfully!")
print(f"\nModel Info:")
print(f"  • Trained epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"  • Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
print(f"  • Val Acc: {checkpoint.get('val_acc', 'N/A'):.2f}%")
print(f"\nTraining Config:")
print(f"  • Dropout: {config.get('dropout', 'N/A')}")
print(f"  • Used RESS: {config.get('use_ress', False)}")
if config.get('use_ress', False):
    print(f"  • N-way (training): {config.get('n_way', 'N/A')}")
    print(f"  • K-shot (training): {config.get('k_shot', 'N/A')}")

# ============================================================================
# Evaluation Functions
# ============================================================================
def evaluate_on_classes(model, testset, class_list, desc="Evaluating"):
    """Evaluate model on specific classes"""
    test_indices = [i for i, (_, label) in enumerate(testset) if label in class_list]
    test_subset = Subset(testset, test_indices)
    testloader = DataLoader(test_subset, batch_size=100, shuffle=False)
    
    model.eval()
    correct = 0
    total = 0
    
    all_classes = list(range(len(model.prototypes)))
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            _, predicted = logits.max(1)
            
            mapped_preds = torch.tensor([all_classes[p.item()] for p in predicted]).to(device)
            
            total += labels.size(0)
            correct += (mapped_preds == labels).sum().item()
    
    return 100. * correct / total

def add_incremental_classes(model, incremental_trainsets, new_classes):
    """Add multiple new classes to the model"""
    model.eval()
    
    print(f"\nAdding {len(new_classes)} incremental classes:")
    print(f"  {[class_names[i] for i in new_classes]}")
    
    all_new_features = []
    all_new_labels = []
    
    for cls in new_classes:
        incremental_loader = DataLoader(incremental_trainsets[cls], 
                                       batch_size=args.k_shot, shuffle=False)
        with torch.no_grad():
            for images, labels in incremental_loader:
                images = images.to(device)
                features = model.extract_features(images)
                all_new_features.append(features)
                all_new_labels.extend(labels.numpy())
    
    all_new_features = torch.cat(all_new_features, dim=0)
    old_prototypes = model.prototypes.data
    
    # Compute prototype for each new class
    new_prototypes = []
    for cls in new_classes:
        cls_mask = torch.tensor([l == cls for l in all_new_labels], dtype=torch.bool).to(device)
        cls_features = all_new_features[cls_mask]
        cls_prototype = cls_features.mean(dim=0, keepdim=True)
        new_prototypes.append(cls_prototype)
    
    new_prototypes = torch.cat(new_prototypes, dim=0)
    
    # Normalize and scale
    new_prototypes = F.normalize(new_prototypes, p=2, dim=1)
    old_magnitude = old_prototypes.norm(dim=1).mean()
    new_prototypes = new_prototypes * old_magnitude
    
    print(f"  ✓ Computed prototypes ({args.k_shot}-shot)")
    
    # Apply relation projection
    with torch.no_grad():
        refined_protos, relation_matrix = model.relation_proj(new_prototypes, old_prototypes)
    
    print(f"  ✓ Applied Dynamic Relation Projection")
    
    # Expand prototypes
    updated_prototypes = torch.cat([old_prototypes, new_prototypes], dim=0)
    model.prototypes = nn.Parameter(updated_prototypes)
    
    print(f"  ✓ Model expanded: {len(old_prototypes)} → {len(updated_prototypes)} classes")
    
    return relation_matrix

def compute_confusion_matrix(model, testset, class_list):
    """Compute confusion matrix"""
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
            
            # Predictions are prototype indices (0-9)
            # We need to map them to actual class labels
            # Since prototypes are ordered [0,1,2,3,4,5,6,7,8,9] for classes [0,1,2,3,4,5,6,7,8,9]
            # The mapping is direct: prototype_idx = class_label
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute confusion matrix with actual class labels
    cm = confusion_matrix(all_labels, all_preds, labels=class_list)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    return cm, cm_normalized

# ============================================================================
# Session 0: Test Base Model (Before Incremental Learning)
# ============================================================================
print("\n" + "="*70)
print("SESSION 0: BASE MODEL (BEFORE INCREMENTAL LEARNING)")
print("="*70)

base_acc_s0 = evaluate_on_classes(model, testset_full, base_classes)

print(f"\n📊 Base Model Performance:")
print(f"   Base Classes (7): {base_acc_s0:.2f}%")

# ============================================================================
# Session 1: Add Incremental Classes
# ============================================================================
print("\n" + "="*70)
print("SESSION 1: INCREMENTAL LEARNING")
print("="*70)

relation_matrix = add_incremental_classes(model, incremental_trainsets, incremental_classes)

# Evaluate after increment
all_classes = base_classes + incremental_classes
base_acc_s1 = evaluate_on_classes(model, testset_full, base_classes)
inc_acc_s1 = evaluate_on_classes(model, testset_full, incremental_classes)
overall_acc_s1 = evaluate_on_classes(model, testset_full, all_classes)

print(f"\n📊 After Adding Incremental Classes:")
print(f"   Base Classes (7):        {base_acc_s1:.2f}%")
print(f"   Incremental Classes (3): {inc_acc_s1:.2f}%")
print(f"   Overall (10):            {overall_acc_s1:.2f}%")

forgetting = base_acc_s0 - base_acc_s1
print(f"\n💡 Forgetting: {forgetting:+.2f}%", end="")
if forgetting < 0:
    print(" (Negative forgetting - base improved!)")
elif forgetting < 3:
    print(" (Excellent - minimal forgetting)")
elif forgetting < 5:
    print(" (Good)")
else:
    print(" (Needs improvement)")

# ============================================================================
# Detailed Per-Class Analysis
# ============================================================================
print("\n" + "="*70)
print("PER-CLASS ACCURACY ANALYSIS")
print("="*70)

cm, cm_normalized = compute_confusion_matrix(model, testset_full, all_classes)
per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100

display_names = [class_names[i] for i in all_classes]
base_accs = per_class_acc[:7]
inc_accs = per_class_acc[7:]

print(f"\n{' Class':<15} {'Type':<12} {'Accuracy':<12} {'Samples':<10}")
print("-"*70)

for i, class_name in enumerate(display_names):
    class_type = "Base" if i < 7 else "Incremental"
    samples = cm[i].sum()
    print(f"{class_name:<15} {class_type:<12} {per_class_acc[i]:>10.2f}% {int(samples):<10}")

print("-"*70)
print(f"{'Base Avg (7)':<15} {'Summary':<12} {np.mean(base_accs):>10.2f}%")
print(f"{'Inc Avg (3)':<15} {'Summary':<12} {np.mean(inc_accs):>10.2f}%")
print(f"{'Overall (10)':<15} {'Summary':<12} {np.mean(per_class_acc):>10.2f}%")
print("="*70)

# ============================================================================
# Visualizations
# ============================================================================
print("\n📊 Generating visualizations...")

# 1. Session Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

sessions = ['Session 0\n(Base Only)', 'Session 1\n(After Increment)']
x = np.arange(len(sessions))
width = 0.25

bars1 = ax1.bar(x - width, [base_acc_s0, base_acc_s1], width, 
                label='Base (7)', color='#3498db', alpha=0.8)
bars2 = ax1.bar(x, [0, inc_acc_s1], width,
                label='Incremental (3)', color='#e74c3c', alpha=0.8)
bars3 = ax1.bar(x + width, [base_acc_s0, overall_acc_s1], width,
                label='Overall', color='#2ecc71', alpha=0.8)

ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax1.set_title(f'Session Comparison ({args.k_shot}-shot)', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(sessions)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, 105])

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Relation matrix
im = ax2.imshow(relation_matrix.cpu().numpy(), aspect='auto', cmap='plasma', vmin=0, vmax=1)
plt.colorbar(im, ax=ax2, label='Cosine Similarity')
ax2.set_xlabel('All Classes', fontsize=12, fontweight='bold')
ax2.set_ylabel('Base Classes', fontsize=12, fontweight='bold')
ax2.set_title('Relation Matrix', fontsize=13, fontweight='bold')
ax2.axvline(6.5, color='red', linewidth=3, label='New Classes')
ax2.legend()

plt.tight_layout()
plt.savefig(f'{args.output_dir}/session_comparison_{args.k_shot}shot.png', dpi=150)
print(f"✓ Saved: {args.output_dir}/session_comparison_{args.k_shot}shot.png")

# 2. Confusion Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='YlGnBu',
            xticklabels=display_names, yticklabels=display_names,
            cbar_kws={'label': 'Percentage (%)'}, linewidths=0.5, vmin=0, vmax=100)
plt.title(f'Confusion Matrix ({args.k_shot}-shot)', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Predicted', fontsize=12, fontweight='bold')
plt.ylabel('True', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')

for i in range(7, 10):
    plt.gca().add_patch(plt.Rectangle((i, i), 1, 1, fill=False, 
                                     edgecolor='red', lw=4))

plt.tight_layout()
plt.savefig(f'{args.output_dir}/confusion_matrix_{args.k_shot}shot.png', dpi=150)
print(f"✓ Saved: {args.output_dir}/confusion_matrix_{args.k_shot}shot.png")

# 3. Per-Class Accuracy
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(display_names))
colors = ['#3498db'] * 7 + ['#e74c3c'] * 3

bars = ax.bar(x, per_class_acc, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Class', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title(f'Per-Class Accuracy ({args.k_shot}-shot)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(display_names, rotation=45, ha='right')
ax.set_ylim([0, 105])
ax.grid(axis='y', alpha=0.3)
ax.axvspan(6.5, 9.5, alpha=0.1, color='red')

plt.tight_layout()
plt.savefig(f'{args.output_dir}/per_class_accuracy_{args.k_shot}shot.png', dpi=150)
print(f"✓ Saved: {args.output_dir}/per_class_accuracy_{args.k_shot}shot.png")

# 4. Prototype Distribution
all_prototypes = model.prototypes.data.cpu().numpy()
pca = PCA(n_components=2)
protos_2d = pca.fit_transform(all_prototypes)

plt.figure(figsize=(14, 10))

for i in range(7):
    plt.scatter(protos_2d[i, 0], protos_2d[i, 1], 
               c='#3498db', marker='o', s=400, 
               edgecolors='black', linewidths=2.5, alpha=0.8, zorder=3,
               label='Base' if i == 0 else '')

for i in range(7, 10):
    plt.scatter(protos_2d[i, 0], protos_2d[i, 1], 
               c='#e74c3c', marker='^', s=500, 
               edgecolors='black', linewidths=2.5, alpha=0.8, zorder=3,
               label='Incremental' if i == 7 else '')

for i, (x, y) in enumerate(protos_2d):
    plt.annotate(display_names[i], (x, y), 
                fontsize=12, ha='center', va='top' if i < 7 else 'bottom',
                fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='yellow' if i >= 7 else 'lightblue', 
                         alpha=0.7, edgecolor='black'))

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12, fontweight='bold')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12, fontweight='bold')
plt.title(f'Prototype Distribution ({args.k_shot}-shot)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f'{args.output_dir}/prototype_distribution_{args.k_shot}shot.png', dpi=150)
print(f"✓ Saved: {args.output_dir}/prototype_distribution_{args.k_shot}shot.png")

# Save results summary
with open(f'{args.output_dir}/results_{args.k_shot}shot.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write(f"TESTING RESULTS ({args.k_shot}-SHOT)\n")
    f.write("="*70 + "\n")
    f.write(f"\nCheckpoint: {args.checkpoint}\n")
    f.write(f"K-shot: {args.k_shot}\n")
    f.write(f"\nSession 0 (Base Only):\n")
    f.write(f"  Base Accuracy: {base_acc_s0:.2f}%\n")
    f.write(f"\nSession 1 (After Increment):\n")
    f.write(f"  Base Accuracy: {base_acc_s1:.2f}%\n")
    f.write(f"  Incremental Accuracy: {inc_acc_s1:.2f}%\n")
    f.write(f"  Overall Accuracy: {overall_acc_s1:.2f}%\n")
    f.write(f"  Forgetting: {forgetting:+.2f}%\n")
    f.write(f"\nPer-Class Results:\n")
    f.write(f"  Base Classes Average: {np.mean(base_accs):.2f}%\n")
    f.write(f"  Incremental Classes Average: {np.mean(inc_accs):.2f}%\n")
    f.write(f"  Overall Average: {np.mean(per_class_acc):.2f}%\n")

print(f"✓ Saved: {args.output_dir}/results_{args.k_shot}shot.txt")

print("\n" + "="*70)
print("TESTING COMPLETE!")
print("="*70)
print(f"\n📊 Summary:")
print(f"   Base Accuracy: {base_acc_s0:.2f}% → {base_acc_s1:.2f}% (Δ {forgetting:+.2f}%)")
print(f"   Incremental Accuracy: {inc_acc_s1:.2f}%")
print(f"   Overall Accuracy: {overall_acc_s1:.2f}%")
print(f"\n📁 All results saved to: {args.output_dir}/")
print(f"   • session_comparison_{args.k_shot}shot.png")
print(f"   • confusion_matrix_{args.k_shot}shot.png")
print(f"   • per_class_accuracy_{args.k_shot}shot.png")
print(f"   • prototype_distribution_{args.k_shot}shot.png")
print(f"   • results_{args.k_shot}shot.txt")
print("="*70)