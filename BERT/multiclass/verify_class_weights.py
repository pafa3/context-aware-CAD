#!/usr/bin/env python3
"""
Verify class weights calculation and application
"""

import torch
import torch.nn as nn
from collections import Counter
import pandas as pd
import sys
sys.path.append('/workspace/BERT/multiclass')

# Import the dataset
from contextual_abuse_dataset4_improved import ContextualAbuseRedditDataset, CATEGORY_NAMES

def calculate_class_weights(labels):
    """Calculate class weights for imbalanced dataset"""
    label_counts = Counter(labels)
    total_count = sum(label_counts.values())
    class_weights = {label: total_count / (len(label_counts) * count) 
                    for label, count in label_counts.items()}
    weights = [class_weights[i] for i in range(len(class_weights))]
    return torch.FloatTensor(weights)

def main():
    print("Loading dataset to verify class weights...")
    
    # Load dataset
    dataset_builder = ContextualAbuseRedditDataset(level=3)
    dataset_builder.download_and_prepare()
    dataset = dataset_builder.as_dataset()
    
    # Get training data
    df_train = pd.DataFrame(dataset["train"])
    
    print(f"\nTotal training samples: {len(df_train)}")
    print(f"Category names: {CATEGORY_NAMES}")
    
    # Get label distribution
    labels = df_train['labels_info'].tolist()
    label_counts = Counter(labels)
    
    print("\nLabel distribution:")
    for label_idx, count in sorted(label_counts.items()):
        if label_idx < len(CATEGORY_NAMES):
            print(f"  {label_idx} ({CATEGORY_NAMES[label_idx]}): {count} samples")
    
    # Calculate class weights
    class_weights = calculate_class_weights(labels)
    print(f"\nCalculated class weights tensor: {class_weights}")
    
    print("\nClass weights by category:")
    for i, weight in enumerate(class_weights):
        if i < len(CATEGORY_NAMES):
            print(f"  {CATEGORY_NAMES[i]}: {weight:.4f}")
    
    # Verify the weights make sense
    # The weight should be inversely proportional to class frequency
    print("\nVerification:")
    print("Categories with fewer samples should have higher weights:")
    
    # Sort by count to see if weights are inversely proportional
    sorted_by_count = sorted([(CATEGORY_NAMES[idx], count, class_weights[idx].item()) 
                             for idx, count in label_counts.items() 
                             if idx < len(CATEGORY_NAMES)], 
                            key=lambda x: x[1])
    
    print("\nCategory (sorted by count) | Count | Weight")
    print("-" * 45)
    for cat, count, weight in sorted_by_count:
        print(f"{cat:20} | {count:5} | {weight:.4f}")
    
    # Test loss function with weights
    print("\n\nTesting CrossEntropyLoss with class weights:")
    
    # Create dummy predictions and labels
    batch_size = 10
    num_classes = len(CATEGORY_NAMES)
    
    # Create predictions (logits)
    logits = torch.randn(batch_size, num_classes)
    
    # Create labels with imbalanced distribution (more of class 0)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 3, 3])
    
    # Loss without weights
    loss_fn_no_weight = nn.CrossEntropyLoss()
    loss_no_weight = loss_fn_no_weight(logits, labels)
    
    # Loss with weights
    loss_fn_weighted = nn.CrossEntropyLoss(weight=class_weights)
    loss_weighted = loss_fn_weighted(logits, labels)
    
    print(f"Loss without weights: {loss_no_weight:.4f}")
    print(f"Loss with weights: {loss_weighted:.4f}")
    
    # The weighted loss should be different from unweighted
    print(f"\nWeights are {'correctly' if abs(loss_weighted - loss_no_weight) > 0.01 else 'NOT'} affecting the loss calculation")

if __name__ == "__main__":
    main()