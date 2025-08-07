import torch
import random
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from datasets import load_dataset
import contextual_abuse_dataset4_improved
from contextual_abuse_dataset4_improved import ContextualAbuseRedditDataset

# Set seed for reproducibility
def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)

class ConversationalDatasetWithSegments(Dataset):
    """Dataset that properly uses BERT's two-segment architecture"""
    
    def __init__(self, dataframe, tokenizer, max_len, num_classes):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_classes = num_classes

    def __getitem__(self, index):
        text = str(self.data.text.iloc[index])
        parent_text = str(self.data.parent_text.iloc[index])
        
        # Parse the conversation to separate speakers
        current_speaker_utterances = []
        other_speaker_utterances = []
        
        # Current text is always from Speaker1
        current_speaker_utterances.append(text.replace("Speaker1: ", ""))
        
        # Parse parent text to separate by speakers
        if parent_text and parent_text.strip():
            # Split by "Speaker" to get individual utterances
            parts = parent_text.split("Speaker")
            for part in parts:
                if part.strip():
                    if part.startswith("1: "):
                        # This is from Speaker1 (same as current)
                        current_speaker_utterances.append(part[3:].strip())
                    elif part.startswith("2: ") or part.startswith("3: "):
                        # This is from other speakers
                        other_speaker_utterances.append(part[3:].strip())
        
        # Join utterances by speaker
        # Segment A: Other speakers' utterances (context)
        # Segment B: Current speaker's utterances (including the target)
        segment_a = " ".join(other_speaker_utterances) if other_speaker_utterances else ""
        segment_b = " ".join(reversed(current_speaker_utterances))  # Most recent first
        
        # If no context, just use the current text
        if not segment_a:
            segment_a = segment_b
            segment_b = None
        
        # Tokenize with proper segment separation
        inputs = self.tokenizer(
            text=segment_a,
            text_pair=segment_b,  # This will be None if no context
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,  # Important! This creates the segment IDs
            return_tensors='pt'
        )
        
        # Get label
        labels_info = self.data.labels_info.iloc[index]
        label = labels_info['label'][0]
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs['token_type_ids'].flatten(),  # Include segment IDs
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def __len__(self):
        return self.len

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted'),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted')
    }

def prepare_data(level=3):
    """Prepare datasets for training"""
    dataset_builder = ContextualAbuseRedditDataset(level=level)
    dataset_builder.download_and_prepare()
    dataset = dataset_builder.as_dataset()
    
    df_train = pd.DataFrame(dataset["train"])
    df_validation = pd.DataFrame(dataset["validation"]) 
    df_test = pd.DataFrame(dataset["test"])
    
    # Filter out samples with no labels
    df_train = df_train[df_train['labels_info'].apply(lambda x: len(x['label']) > 0)]
    df_validation = df_validation[df_validation['labels_info'].apply(lambda x: len(x['label']) > 0)]
    df_test = df_test[df_test['labels_info'].apply(lambda x: len(x['label']) > 0)]
    
    return df_train, df_validation, df_test

def main():
    # Configuration
    MODEL_NAME = "DeepPavlov/bert-base-cased-conversational"
    MAX_LEN = 512
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 5
    WARMUP_STEPS = 500
    LEVEL = 3  # Use all conversation context
    
    print(f"Loading conversational BERT model with proper segment handling: {MODEL_NAME}")
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Prepare data
    print("Preparing datasets...")
    df_train, df_validation, df_test = prepare_data(level=LEVEL)
    
    print(f"Train samples: {len(df_train)}")
    print(f"Validation samples: {len(df_validation)}")
    print(f"Test samples: {len(df_test)}")
    
    # Get number of classes
    n_classes = 4  # Based on CATEGORY_NAMES
    
    # Create datasets with proper segment handling
    train_dataset = ConversationalDatasetWithSegments(
        df_train.reset_index(drop=True), 
        tokenizer, 
        MAX_LEN, 
        n_classes
    )
    
    val_dataset = ConversationalDatasetWithSegments(
        df_validation.reset_index(drop=True), 
        tokenizer, 
        MAX_LEN, 
        n_classes
    )
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=n_classes,
        output_attentions=False,
        output_hidden_states=False
    )
    
    # Show example of how data is formatted
    print("\nExample of data formatting:")
    example = train_dataset[0]
    print(f"Input shape: {example['input_ids'].shape}")
    print(f"Token type IDs shape: {example['token_type_ids'].shape}")
    print(f"Unique token type IDs: {torch.unique(example['token_type_ids'])}")
    
    # Decode to show what the model sees
    decoded = tokenizer.decode(example['input_ids'], skip_special_tokens=False)
    print(f"\nDecoded example (first 200 chars): {decoded[:200]}...")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results_conversational_bert_segments',
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        logging_dir='./logs_segments',
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to="none"
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    print("\nStarting training with proper speaker segmentation...")
    trainer.train()
    
    # Save the best model
    print("Saving model...")
    trainer.save_model('./best_conversational_bert_segments_model')
    tokenizer.save_pretrained('./best_conversational_bert_segments_model')
    
    # Evaluate
    print("Evaluating on validation set...")
    eval_results = trainer.evaluate()
    print(f"Validation results: {eval_results}")
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = main()