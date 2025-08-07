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

# Check device
def get_torch_device(verbose: bool = True, gpu_ix: int = 0) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print(f'We will use the GPU: {torch.cuda.get_device_name(gpu_ix)}')
    else:
        if verbose: 
            print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device

device = get_torch_device()

class ConversationalDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, num_classes):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_classes = num_classes

    def __getitem__(self, index):
        text = str(self.data.text.iloc[index])
        parent_text = str(self.data.parent_text.iloc[index])
        
        # For conversational BERT, we can concatenate everything into one sequence
        # The model is pre-trained on conversational data and understands this format
        if parent_text and parent_text.strip():
            # Combine context and current text
            full_conversation = f"{parent_text} {text}"
        else:
            full_conversation = text
        
        # Tokenize the full conversation
        inputs = self.tokenizer(
            full_conversation,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get label
        labels_info = self.data.labels_info.iloc[index]
        label = labels_info['label'][0]
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
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
    # Load the dataset using the improved dataset class
    dataset_builder = ContextualAbuseRedditDataset(level=level)
    dataset_builder.download_and_prepare()
    dataset = dataset_builder.as_dataset()
    
    # Convert to pandas DataFrames
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
    
    print(f"Loading conversational BERT model: {MODEL_NAME}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Prepare data
    print("Preparing datasets...")
    df_train, df_validation, df_test = prepare_data(level=LEVEL)
    
    # Get number of classes
    n_classes = 4  # Based on CATEGORY_NAMES
    
    # Create datasets
    train_dataset = ConversationalDataset(
        df_train.reset_index(drop=True), 
        tokenizer, 
        MAX_LEN, 
        n_classes
    )
    
    val_dataset = ConversationalDataset(
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
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results_conversational_bert',
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        logging_dir='./logs',
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
    print("Starting training...")
    trainer.train()
    
    # Save the best model
    print("Saving model...")
    trainer.save_model('./best_conversational_bert_model')
    tokenizer.save_pretrained('./best_conversational_bert_model')
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    eval_results = trainer.evaluate()
    print(f"Validation results: {eval_results}")
    
    # Create test dataset and evaluate
    test_dataset = ConversationalDataset(
        df_test.reset_index(drop=True), 
        tokenizer, 
        MAX_LEN, 
        n_classes
    )
    
    print("Evaluating on test set...")
    test_results = trainer.predict(test_dataset)
    test_metrics = compute_metrics((test_results.predictions, test_results.label_ids))
    print(f"Test results: {test_metrics}")
    
    return model, tokenizer, test_results

if __name__ == "__main__":
    model, tokenizer, test_results = main()