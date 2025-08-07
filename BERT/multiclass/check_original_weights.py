#!/usr/bin/env python3
"""
Check if the weight calculation matches the original approach
"""

import sys
sys.path.append('/workspace/BERT/multiclass')
from collections import Counter
import pandas as pd
import torch

# Import the dataset
from contextual_abuse_dataset4_improved import ContextualAbuseRedditDataset, CATEGORY_NAMES

def original_calculate_class_weights(dataset_labels):
    """Original weight calculation from the notebook"""
    # Count each label's occurrences
    label_counts = Counter(dataset_labels)

    # Total number of samples
    total_count = sum(label_counts.values())

    # Calculate weight for each class
    class_weights = {label: total_count / (len(label_counts) * count) for label, count in label_counts.items()}

    # Convert to a list (if necessary for your framework)
    weights = [class_weights[i] for i in range(len(class_weights))]

    return weights

def main():
    print("Checking weight calculation approaches...")
    print("=" * 60)
    
    # Load dataset
    dataset_builder = ContextualAbuseRedditDataset(level=3)
    dataset_builder.download_and_prepare()
    dataset = dataset_builder.as_dataset()
    
    # Get training data
    df_train = pd.DataFrame(dataset["train"])
    
    print(f"Total training samples: {len(df_train)}")
    print(f"Data structure check:")
    print(f"  - Type of labels_info column: {type(df_train['labels_info'].iloc[0])}")
    print(f"  - First 5 labels: {df_train['labels_info'].head().tolist()}")
    
    # Current approach - labels_info is directly the integer label
    current_labels = df_train['labels_info'].tolist()
    
    # Your original code tried to extract from nested structure
    # all_labels = [label for labels_info in df_train['labels_info'] for label in labels_info['label']]
    # But now labels_info IS the label directly
    
    print(f"\nLabel distribution:")
    label_counts = Counter(current_labels)
    for label_idx, count in sorted(label_counts.items()):
        if label_idx < len(CATEGORY_NAMES):
            percentage = (count / len(current_labels)) * 100
            print(f"  {label_idx} ({CATEGORY_NAMES[label_idx]}): {count} samples ({percentage:.1f}%)")
    
    # Calculate weights using original function
    weights = original_calculate_class_weights(current_labels)
    weights_tensor = torch.FloatTensor(weights)
    
    print(f"\nCalculated weights (using original formula):")
    for i, weight in enumerate(weights):
        if i < len(CATEGORY_NAMES):
            print(f"  Class {i} ({CATEGORY_NAMES[i]}): {weight:.4f}")
    
    # Verify the formula
    print(f"\nFormula verification:")
    total = sum(label_counts.values())
    n_classes = len(label_counts)
    print(f"  Total samples: {total}")
    print(f"  Number of classes: {n_classes}")
    print(f"  Formula: weight[i] = total / (n_classes * count[i])")
    
    # Check if weights are inversely proportional to frequency
    print(f"\nWeight vs Frequency check:")
    sorted_by_count = sorted([(idx, count, weights[idx]) 
                             for idx, count in label_counts.items()], 
                            key=lambda x: x[1])
    
    print("  Rank | Class | Count | Weight | Count*Weight")
    print("  " + "-" * 50)
    for rank, (idx, count, weight) in enumerate(sorted_by_count, 1):
        product = count * weight
        print(f"  {rank:4} | {idx:5} | {count:5} | {weight:6.4f} | {product:12.4f}")
    
    # The product should be constant (total/n_classes) for balanced weighting
    expected_product = total / n_classes
    print(f"\n  Expected count*weight product: {expected_product:.4f}")
    
    # Additional check: what if there was a nested structure?
    print("\n" + "=" * 60)
    print("IMPORTANT NOTE:")
    print("In your original code, you had:")
    print("  all_labels = [label for labels_info in df_train['labels_info'] for label in labels_info['label']]")
    print("\nThis suggests labels_info was a nested structure (dict/list).")
    print("In the current implementation, labels_info is directly the integer label.")
    print("This difference might affect the weight distribution if the original")
    print("data structure had multiple labels per sample or a different format.")

if __name__ == "__main__":
    main()