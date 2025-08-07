"""
Test script to verify data loading works with actual CSV structure
"""

import contextual_abuse_dataset4_improved
from contextual_abuse_dataset4_improved import ContextualAbuseRedditDataset
import pandas as pd

def test_data_loading():
    print("Testing data loading with actual CSV files...")
    
    # Test loading the dataset
    try:
        dataset_builder = ContextualAbuseRedditDataset(level=3)
        dataset_builder.download_and_prepare()
        dataset = dataset_builder.as_dataset()
        
        print("\n✓ Dataset loaded successfully!")
        
        # Convert to DataFrames
        df_train = pd.DataFrame(dataset["train"])
        df_validation = pd.DataFrame(dataset["validation"]) 
        df_test = pd.DataFrame(dataset["test"])
        
        print(f"\nDataset sizes:")
        print(f"  Train: {len(df_train)} samples")
        print(f"  Validation: {len(df_validation)} samples")
        print(f"  Test: {len(df_test)} samples")
        
        # Check data structure
        print(f"\nColumns in dataset: {list(df_train.columns)}")
        
        # Check label distribution
        print(f"\nLabel distribution in training set:")
        label_counts = df_train['labels_info'].value_counts()
        for label_id, count in label_counts.items():
            label_name = contextual_abuse_dataset4_improved.CATEGORY_NAMES[label_id]
            print(f"  {label_name} (id={label_id}): {count}")
        
        # Show a sample
        print(f"\nSample from training set:")
        sample = df_train.iloc[0]
        print(f"  ID: {sample['id']}")
        print(f"  Text: {sample['text'][:100]}...")
        print(f"  Parent text: {sample['parent_text'][:100]}..." if sample['parent_text'] else "  Parent text: (empty)")
        print(f"  Label: {contextual_abuse_dataset4_improved.CATEGORY_NAMES[sample['labels_info']]}")
        
        # Test that labels are integers
        assert all(isinstance(label, int) for label in df_train['labels_info']), "Labels should be integers"
        assert all(0 <= label < 4 for label in df_train['labels_info']), "Labels should be in range 0-3"
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_loading()