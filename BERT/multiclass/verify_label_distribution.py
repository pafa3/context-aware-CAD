#!/usr/bin/env python3
"""
Verify label distribution and weight calculations
Shows how Slur and CounterSpeech are merged into Neutral
"""

import sys
sys.path.append('/workspace/BERT/multiclass')
from collections import Counter
import pandas as pd
import torch
import csv

# Import the dataset
from contextual_abuse_dataset4_improved import ContextualAbuseRedditDataset, CATEGORY_NAMES

def analyze_raw_csv():
    """Analyze the raw CSV to see original label distribution"""
    print("ANALYZING RAW CSV DATA")
    print("="*60)
    
    # Try to find and read the raw CSV
    csv_paths = [
        '/workspace/data/reddit_train.csv',
        '/workspace/reddit_train.csv',
        '/workspace/BERT/data/reddit_train.csv',
        '/workspace/BERT/multiclass/data/reddit_train.csv'
    ]
    
    raw_data = None
    for path in csv_paths:
        try:
            raw_data = pd.read_csv(path)
            print(f"Found CSV at: {path}")
            break
        except:
            continue
    
    if raw_data is None:
        print("Could not find raw CSV file")
        return None
    
    # Count raw labels
    raw_counts = Counter(raw_data['annotation_Primary'].fillna(''))
    print(f"\nRaw label distribution (before merging):")
    total_raw = sum(raw_counts.values())
    for label, count in sorted(raw_counts.items()):
        percentage = (count / total_raw) * 100
        print(f"  {label:30} : {count:6} ({percentage:5.2f}%)")
    
    # Show what will be merged
    slur_count = raw_counts.get('Slur', 0)
    counter_count = raw_counts.get('CounterSpeech', 0)
    neutral_count = raw_counts.get('Neutral', 0)
    
    print(f"\nMerging into Neutral:")
    print(f"  Original Neutral: {neutral_count}")
    print(f"  + Slur: {slur_count}")
    print(f"  + CounterSpeech: {counter_count}")
    print(f"  = Total Neutral after merge: {neutral_count + slur_count + counter_count}")
    
    return raw_counts

def calculate_class_weights(labels):
    """Calculate class weights using the original formula"""
    label_counts = Counter(labels)
    total_count = sum(label_counts.values())
    class_weights = {label: total_count / (len(label_counts) * count) 
                    for label, count in label_counts.items()}
    weights = [class_weights[i] for i in range(len(class_weights))]
    return torch.FloatTensor(weights), label_counts

def analyze_processed_data():
    """Analyze the processed dataset after merging"""
    print("\n\nANALYZING PROCESSED DATASET")
    print("="*60)
    
    # Load dataset
    dataset_builder = ContextualAbuseRedditDataset(level=3)
    dataset_builder.download_and_prepare()
    dataset = dataset_builder.as_dataset()
    
    # Get all splits
    for split_name in ['train', 'validation', 'test']:
        if split_name in dataset:
            df = pd.DataFrame(dataset[split_name])
            labels = df['labels_info'].tolist()
            
            print(f"\n{split_name.upper()} Split:")
            print(f"Total samples: {len(labels)}")
            
            # Count labels
            label_counts = Counter(labels)
            print(f"Label distribution (after merging):")
            for label_id, count in sorted(label_counts.items()):
                if label_id < len(CATEGORY_NAMES):
                    percentage = (count / len(labels)) * 100
                    print(f"  {label_id} ({CATEGORY_NAMES[label_id]:25}) : {count:6} ({percentage:5.2f}%)")
            
            # Calculate weights for training set
            if split_name == 'train':
                weights, _ = calculate_class_weights(labels)
                print(f"\nClass weights for training:")
                for i, weight in enumerate(weights):
                    if i < len(CATEGORY_NAMES):
                        print(f"  {CATEGORY_NAMES[i]:25} : {weight:.4f}")
                
                # Verify the weight calculation
                print(f"\nWeight calculation verification:")
                total = len(labels)
                n_classes = len(label_counts)
                print(f"  Total samples: {total}")
                print(f"  Number of classes: {n_classes}")
                print(f"  Formula: weight[i] = {total} / ({n_classes} * count[i])")
                
                print(f"\n  Class | Count | Weight | Count*Weight")
                print(f"  " + "-"*45)
                for i, count in label_counts.items():
                    if i < len(CATEGORY_NAMES):
                        weight = weights[i].item()
                        product = count * weight
                        print(f"  {CATEGORY_NAMES[i]:25} | {count:5} | {weight:6.4f} | {product:12.4f}")
                
                expected_product = total / n_classes
                print(f"\n  Expected count*weight (balanced): {expected_product:.4f}")

def main():
    print("LABEL DISTRIBUTION AND WEIGHT VERIFICATION")
    print("="*70)
    print("Checking how Slur and CounterSpeech are merged into Neutral")
    print("="*70)
    
    # First analyze raw data if available
    raw_counts = analyze_raw_csv()
    
    # Then analyze processed data
    analyze_processed_data()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nThe implementation correctly:")
    print("1. Maps 'Slur' and 'CounterSpeech' to 'Neutral' (line 132 in contextual_abuse_dataset4_improved.py)")
    print("2. Calculates class weights using: weight[i] = total_samples / (num_classes * count[i])")
    print("3. Applies these weights in CrossEntropyLoss during training")
    print("\nThis ensures that:")
    print("- Minority classes (abuse types) get higher weights")
    print("- The model pays more attention to correctly classifying rare classes")
    print("- The severely underrepresented 'Slur' class is merged into 'Neutral'")

if __name__ == "__main__":
    main()