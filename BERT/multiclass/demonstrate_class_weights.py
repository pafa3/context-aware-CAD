#!/usr/bin/env python3
"""
Demonstrate how class weights work in PyTorch for imbalanced datasets
"""

import torch
import torch.nn as nn
import numpy as np

def demonstrate_class_weights():
    """Show how class weights affect loss calculation"""
    
    print("=" * 60)
    print("CLASS WEIGHT DEMONSTRATION FOR IMBALANCED DATASETS")
    print("=" * 60)
    
    # Example: 4 classes with imbalanced distribution
    # Let's say we have:
    # - Neutral: 5000 samples (majority class)
    # - IdentityDirectedAbuse: 1000 samples
    # - AffiliationDirectedAbuse: 500 samples  
    # - PersonDirectedAbuse: 500 samples
    
    class_names = ['Neutral', 'IdentityDirectedAbuse', 'AffiliationDirectedAbuse', 'PersonDirectedAbuse']
    class_counts = [5000, 1000, 500, 500]
    total_samples = sum(class_counts)
    num_classes = len(class_names)
    
    print("\nClass Distribution:")
    for name, count in zip(class_names, class_counts):
        percentage = (count / total_samples) * 100
        print(f"  {name}: {count} samples ({percentage:.1f}%)")
    
    # Calculate class weights using the formula:
    # weight[i] = total_samples / (num_classes * count[i])
    class_weights = []
    for i, count in enumerate(class_counts):
        weight = total_samples / (num_classes * count)
        class_weights.append(weight)
    
    weights_tensor = torch.FloatTensor(class_weights)
    
    print("\nCalculated Class Weights:")
    for name, weight in zip(class_names, class_weights):
        print(f"  {name}: {weight:.4f}")
    
    # Demonstrate effect on loss
    print("\n" + "=" * 60)
    print("EFFECT ON LOSS CALCULATION")
    print("=" * 60)
    
    # Create a batch of 8 samples
    batch_size = 8
    logits = torch.randn(batch_size, num_classes)
    
    # Create labels - mostly from majority class
    labels = torch.tensor([0, 0, 0, 0, 1, 2, 3, 0])  # 5 Neutral, 1 of each minority
    
    print(f"\nBatch labels: {labels.tolist()}")
    print("Label distribution in batch:")
    for i in range(num_classes):
        count = (labels == i).sum().item()
        print(f"  Class {i} ({class_names[i]}): {count}")
    
    # Calculate loss WITHOUT weights
    loss_fn_unweighted = nn.CrossEntropyLoss()
    loss_unweighted = loss_fn_unweighted(logits, labels)
    
    # Calculate loss WITH weights
    loss_fn_weighted = nn.CrossEntropyLoss(weight=weights_tensor)
    loss_weighted = loss_fn_weighted(logits, labels)
    
    print(f"\nLoss without weights: {loss_unweighted:.4f}")
    print(f"Loss with weights: {loss_weighted:.4f}")
    print(f"Difference: {abs(loss_weighted - loss_unweighted):.4f}")
    
    # Show per-sample contribution
    print("\n" + "=" * 60)
    print("PER-SAMPLE LOSS CONTRIBUTION")
    print("=" * 60)
    
    # Calculate individual losses
    print("\nIndividual sample contributions to loss:")
    print("Sample | Label | Class Name            | Weight  | Unweighted Loss | Weighted Loss")
    print("-" * 85)
    
    for i in range(batch_size):
        # Get individual loss
        single_logit = logits[i:i+1]
        single_label = labels[i:i+1]
        
        loss_single_unweighted = loss_fn_unweighted(single_logit, single_label)
        loss_single_weighted = loss_fn_weighted(single_logit, single_label)
        
        label_idx = labels[i].item()
        print(f"{i:6} | {label_idx:5} | {class_names[label_idx]:20} | {class_weights[label_idx]:7.4f} | "
              f"{loss_single_unweighted:15.4f} | {loss_single_weighted:13.4f}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("=" * 60)
    print("1. Minority classes (with fewer samples) get HIGHER weights")
    print("2. This makes the model pay more attention to minority class errors")
    print("3. The weighted loss penalizes misclassification of minority classes more heavily")
    print("4. This helps balance the learning process for imbalanced datasets")
    
    # Show the formula
    print("\n" + "=" * 60)
    print("CLASS WEIGHT FORMULA:")
    print("=" * 60)
    print("weight[class] = total_samples / (num_classes * samples_in_class)")
    print("\nThis ensures that:")
    print("- Sum of (weight[i] * count[i]) = total_samples / num_classes for each class")
    print("- The total weight contribution is balanced across classes")

if __name__ == "__main__":
    demonstrate_class_weights()