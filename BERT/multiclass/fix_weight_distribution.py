#!/usr/bin/env python3
"""
Fix weight distribution to match the original approach
"""

import sys
sys.path.append('/workspace/BERT/multiclass')
from collections import Counter
import pandas as pd
import numpy as np
import torch

# Import the dataset
from contextual_abuse_dataset4_improved import ContextualAbuseRedditDataset, CATEGORY_NAMES

def load_original_format_data():
    """Load data in the original format to understand the structure"""
    # Import the ORIGINAL dataset class (not improved)
    from contextual_abuse_dataset4 import ContextualAbuseRedditDataset as OriginalDataset
    
    print("Loading ORIGINAL dataset format...")
    dataset_builder = OriginalDataset(level=3)
    dataset_builder.download_and_prepare()
    dataset = dataset_builder.as_dataset()
    
    df_train = pd.DataFrame(dataset["train"])
    
    print(f"\nOriginal data structure:")
    print(f"Total samples: {len(df_train)}")
    print(f"Type of labels_info: {type(df_train['labels_info'].iloc[0])}")
    print(f"First labels_info entry: {df_train['labels_info'].iloc[0]}")
    
    # Check the structure more deeply
    if len(df_train) > 0:
        first_entry = df_train['labels_info'].iloc[0]
        print(f"Structure details:")
        print(f"  - Is list: {isinstance(first_entry, list)}")
        if isinstance(first_entry, list) and len(first_entry) > 0:
            print(f"  - First element type: {type(first_entry[0])}")
            if isinstance(first_entry[0], dict):
                print(f"  - Keys in first element: {first_entry[0].keys()}")
                if 'label' in first_entry[0]:
                    print(f"  - Type of 'label' value: {type(first_entry[0]['label'])}")
    
    return df_train

def calculate_class_weights_original_way(df_train):
    """Calculate weights exactly as in the original code"""
    print("\nCalculating weights using ORIGINAL method...")
    
    # Method 1: Using the nested list comprehension (your original approach)
    try:
        all_labels = [label for labels_info in df_train['labels_info'] for label in labels_info['label']]
        print(f"Method 1 (nested comprehension) extracted {len(all_labels)} labels")
    except:
        print("Method 1 failed - trying alternative extraction")
        # If the structure is different, try extracting differently
        all_labels = []
        for labels_info in df_train['labels_info']:
            if isinstance(labels_info, list):
                for item in labels_info:
                    if isinstance(item, dict) and 'label' in item:
                        all_labels.append(item['label'])
        print(f"Alternative extraction got {len(all_labels)} labels")
    
    # Count labels
    label_counts = Counter(all_labels)
    print(f"\nLabel distribution (original method):")
    for label, count in sorted(label_counts.items()):
        if label < len(CATEGORY_NAMES):
            print(f"  {label} ({CATEGORY_NAMES[label]}): {count}")
    
    # Calculate weights
    total_count = sum(label_counts.values())
    class_weights = {label: total_count / (len(label_counts) * count) 
                    for label, count in label_counts.items()}
    weights = [class_weights[i] for i in range(len(class_weights))]
    
    print(f"\nWeights (original calculation):")
    for i, weight in enumerate(weights):
        if i < len(CATEGORY_NAMES):
            print(f"  {CATEGORY_NAMES[i]}: {weight:.4f}")
    
    return weights, label_counts

def calculate_class_weights_current_way():
    """Calculate weights using the current improved dataset"""
    print("\n" + "="*60)
    print("Calculating weights using CURRENT method...")
    
    # Load current dataset
    dataset_builder = ContextualAbuseRedditDataset(level=3)
    dataset_builder.download_and_prepare()
    dataset = dataset_builder.as_dataset()
    
    df_train = pd.DataFrame(dataset["train"])
    
    # Current approach - labels_info is directly the integer
    all_labels = df_train['labels_info'].tolist()
    
    # Count labels
    label_counts = Counter(all_labels)
    print(f"\nLabel distribution (current method):")
    for label, count in sorted(label_counts.items()):
        if label < len(CATEGORY_NAMES):
            print(f"  {label} ({CATEGORY_NAMES[label]}): {count}")
    
    # Calculate weights
    total_count = sum(label_counts.values())
    class_weights = {label: total_count / (len(label_counts) * count) 
                    for label, count in label_counts.items()}
    weights = [class_weights[i] for i in range(len(class_weights))]
    
    print(f"\nWeights (current calculation):")
    for i, weight in enumerate(weights):
        if i < len(CATEGORY_NAMES):
            print(f"  {CATEGORY_NAMES[i]}: {weight:.4f}")
    
    return weights, label_counts

def main():
    print("COMPARING WEIGHT DISTRIBUTIONS")
    print("="*60)
    
    # First, try to load and analyze the original format
    try:
        df_train_original = load_original_format_data()
        weights_original, counts_original = calculate_class_weights_original_way(df_train_original)
    except Exception as e:
        print(f"\nCouldn't load original format: {e}")
        weights_original = None
        counts_original = None
    
    # Load and analyze current format
    weights_current, counts_current = calculate_class_weights_current_way()
    
    # Compare if both were successful
    if weights_original is not None:
        print("\n" + "="*60)
        print("COMPARISON:")
        print("="*60)
        
        print("\nWeight differences:")
        for i in range(min(len(weights_original), len(weights_current))):
            if i < len(CATEGORY_NAMES):
                diff = weights_current[i] - weights_original[i]
                print(f"  {CATEGORY_NAMES[i]}:")
                print(f"    Original: {weights_original[i]:.4f}")
                print(f"    Current:  {weights_current[i]:.4f}")
                print(f"    Diff:     {diff:+.4f}")
    
    # Provide recommendation
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("="*60)
    print("\nThe weight calculation formula is the same in both approaches.")
    print("Any differences come from:")
    print("1. How labels are extracted from the data structure")
    print("2. The actual distribution of labels in the dataset")
    print("\nTo match your original results exactly, ensure that:")
    print("- The same samples are included (no filtering differences)")
    print("- The label mapping is consistent (Slur/CounterSpeech -> Neutral)")
    print("- The data split is the same")

if __name__ == "__main__":
    main()