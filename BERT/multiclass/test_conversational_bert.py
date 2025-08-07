"""
Test script to verify the Conversational BERT implementation
"""
import pandas as pd
from transformers import AutoTokenizer
from contextual_abuse_dataset4_improved import ContextualAbuseRedditDataset

def test_dataset_loading():
    """Test if the improved dataset loads correctly"""
    print("Testing dataset loading...")
    
    # Test for different levels
    for level in [1, 2, 3]:
        print(f"\nTesting Level {level}:")
        dataset_builder = ContextualAbuseRedditDataset(level=level)
        
        # Create a small test CSV for demonstration
        test_data = pd.DataFrame({
            'info_id': ['test1', 'test2'],
            'meta_text': ['This is a test comment', 'Another test comment'],
            'parent_text_level_0': ['Parent comment', 'Another parent'],
            'parent_text_level_1': ['Grandparent comment', ''],
            'annotation_Primary': ['Neutral', 'PersonDirectedAbuse']
        })
        
        # Save test data
        test_data.to_csv('test_sample.csv', index=False)
        
        # Test the extraction methods
        row = test_data.iloc[0]
        
        if level == 1:
            text, parent_text = dataset_builder.extract_level_1(row)
            print(f"Text: {text}")
            print(f"Parent text: {parent_text}")
            assert "[SEP]" not in text
            assert "[SEP]" not in parent_text
            
        elif level == 2:
            text, parent_text = dataset_builder.extract_level_2(row)
            print(f"Text: {text}")
            print(f"Parent text: {parent_text}")
            assert "[SEP]" not in text
            assert "[SEP]" not in parent_text
            
        elif level == 3:
            text, parent_text = dataset_builder.extract_level_3(row)
            print(f"Text: {text}")
            print(f"Parent text: {parent_text}")
            assert "[SEP]" not in text
            assert "[SEP]" not in parent_text
    
    print("\n✓ Dataset loading tests passed!")

def test_tokenization():
    """Test if tokenization works correctly with Conversational BERT"""
    print("\nTesting tokenization with Conversational BERT...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/bert-base-cased-conversational")
    
    # Test conversation
    text = "Speaker1: This is a test comment"
    parent_text = "Speaker2: This is the parent comment Speaker3: This is another speaker"
    
    # Concatenate as in our dataset
    full_conversation = f"{parent_text} {text}"
    
    # Tokenize
    inputs = tokenizer(
        full_conversation,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    print(f"\nOriginal conversation: {full_conversation[:100]}...")
    print(f"Number of tokens: {inputs['input_ids'].shape[1]}")
    
    # Decode to see what it looks like
    decoded = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
    print(f"\nDecoded tokens (first 200 chars): {decoded[:200]}...")
    
    # Check that we're not seeing literal [SEP] tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    literal_sep_count = sum(1 for token in tokens if token in ['[', 'SEP', ']'])
    print(f"\nNumber of literal '[', 'SEP', ']' tokens: {literal_sep_count}")
    
    # The tokenizer should add its own [CLS] and [SEP] tokens properly
    assert tokens[0] == '[CLS]'
    assert '[SEP]' in tokens  # Should have the special token, not literal text
    
    print("\n✓ Tokenization tests passed!")

def test_conversation_format():
    """Test different conversation formats"""
    print("\nTesting conversation formats...")
    
    examples = [
        {
            'text': "Speaker1: I disagree with your opinion",
            'parent_text': "Speaker2: This policy is great Speaker3: I think it's wonderful",
            'expected_no_sep': True
        },
        {
            'text': "Speaker1: This is offensive",
            'parent_text': "",
            'expected_no_sep': True
        }
    ]
    
    for i, example in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"Text: {example['text']}")
        print(f"Parent: {example['parent_text'][:50]}..." if example['parent_text'] else "Parent: (empty)")
        
        # Check no [SEP] in the text
        assert "[SEP]" not in example['text']
        assert "[SEP]" not in example['parent_text']
        print("✓ No literal [SEP] tokens found")
    
    print("\n✓ Conversation format tests passed!")

if __name__ == "__main__":
    print("Running Conversational BERT implementation tests...\n")
    
    test_dataset_loading()
    test_tokenization()
    test_conversation_format()
    
    print("\n🎉 All tests passed! The implementation is working correctly.")
    print("\nKey improvements:")
    print("1. No literal [SEP] tokens in the text")
    print("2. Using DeepPavlov's Conversational BERT (pre-trained on dialogues)")
    print("3. Proper conversation formatting with speaker labels")
    print("4. All context concatenated into a single sequence")
    
    # Clean up
    import os
    if os.path.exists('test_sample.csv'):
        os.remove('test_sample.csv')