import torch
import torch.nn as nn
import random
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
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

class SpeakerAwareConversationalBERT(nn.Module):
    """
    Combines DeepPavlov's Conversational BERT with speaker embeddings
    for better multi-speaker conversation understanding
    """
    
    def __init__(self, model_name="DeepPavlov/bert-base-cased-conversational", 
                 num_labels=4, max_speakers=50, dropout_rate=0.1):
        super().__init__()
        
        # Load the conversational BERT model
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Add speaker embeddings (same dimension as BERT hidden size)
        self.speaker_embeddings = nn.Embedding(
            num_embeddings=max_speakers,
            embedding_dim=self.config.hidden_size
        )
        
        # Initialize speaker embeddings
        nn.init.normal_(self.speaker_embeddings.weight, mean=0.0, std=0.02)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification head
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, speaker_ids, labels=None):
        # Get word embeddings from BERT
        word_embeddings = self.bert.embeddings.word_embeddings(input_ids)
        
        # Get position embeddings
        position_ids = torch.arange(input_ids.size(1), device=input_ids.device)
        position_embeddings = self.bert.embeddings.position_embeddings(position_ids)
        
        # Get token type embeddings (for segment IDs if needed)
        token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.bert.embeddings.token_type_embeddings(token_type_ids)
        
        # Get speaker embeddings
        speaker_embeddings = self.speaker_embeddings(speaker_ids)
        
        # Combine all embeddings
        embeddings = word_embeddings + position_embeddings + token_type_embeddings + speaker_embeddings
        embeddings = self.bert.embeddings.LayerNorm(embeddings)
        embeddings = self.bert.embeddings.dropout(embeddings)
        
        # Pass through BERT encoder
        outputs = self.bert.encoder(
            embeddings,
            attention_mask=attention_mask.unsqueeze(1).unsqueeze(2)
        )
        
        # Get the pooled output (CLS token)
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0]  # Take CLS token
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

class SpeakerAwareDataset(Dataset):
    """Dataset that tracks speaker IDs for each token"""
    
    def __init__(self, dataframe, tokenizer, max_len, num_classes):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_classes = num_classes
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = str(self.data.text.iloc[index])
        parent_text = str(self.data.parent_text.iloc[index])
        
        # Parse speakers and create token-speaker mapping
        tokens = []
        speaker_ids = []
        
        # Add CLS token (speaker 0)
        tokens.append(self.tokenizer.cls_token)
        speaker_ids.append(0)
        
        # Process parent text (conversation history)
        if parent_text and parent_text.strip():
            parts = parent_text.split("Speaker")
            for part in parts:
                if part.strip():
                    # Extract speaker number and text
                    if part[0].isdigit() and part[1:3] == ": ":
                        speaker_num = int(part[0])
                        utterance = part[3:].strip()
                        
                        # Tokenize utterance
                        utterance_tokens = self.tokenizer.tokenize(utterance)
                        tokens.extend(utterance_tokens)
                        speaker_ids.extend([speaker_num] * len(utterance_tokens))
        
        # Process current text (always Speaker1)
        current_text = text.replace("Speaker1: ", "").strip()
        current_tokens = self.tokenizer.tokenize(current_text)
        tokens.extend(current_tokens)
        speaker_ids.extend([1] * len(current_tokens))
        
        # Add SEP token
        tokens.append(self.tokenizer.sep_token)
        speaker_ids.append(0)
        
        # Truncate if needed
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len-1] + [self.tokenizer.sep_token]
            speaker_ids = speaker_ids[:self.max_len-1] + [0]
        
        # Convert tokens to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Pad to max length
        padding_length = self.max_len - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
        speaker_ids = speaker_ids + [0] * padding_length
        
        # Create attention mask
        attention_mask = [1] * len(tokens) + [0] * padding_length
        
        # Get label
        labels_info = self.data.labels_info.iloc[index]
        label = labels_info['label'][0]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'speaker_ids': torch.tensor(speaker_ids, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

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

# Custom Trainer to handle our model's output format
class SpeakerAwareTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

def main():
    # Configuration
    MODEL_NAME = "DeepPavlov/bert-base-cased-conversational"
    MAX_LEN = 512
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 5
    WARMUP_STEPS = 500
    LEVEL = 3  # Use all conversation context
    
    print("Loading Speaker-Aware Conversational BERT...")
    print(f"Base model: {MODEL_NAME}")
    print("Enhancement: Speaker embeddings for multi-speaker understanding")
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Prepare data
    print("\nPreparing datasets...")
    df_train, df_validation, df_test = prepare_data(level=LEVEL)
    
    print(f"Train samples: {len(df_train)}")
    print(f"Validation samples: {len(df_validation)}")
    print(f"Test samples: {len(df_test)}")
    
    # Create model
    model = SpeakerAwareConversationalBERT(
        model_name=MODEL_NAME,
        num_labels=4,
        max_speakers=50
    )
    model.to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Speaker embedding parameters: {sum(p.numel() for p in model.speaker_embeddings.parameters()):,}")
    
    # Create datasets
    train_dataset = SpeakerAwareDataset(
        df_train.reset_index(drop=True), 
        tokenizer, 
        MAX_LEN, 
        4
    )
    
    val_dataset = SpeakerAwareDataset(
        df_validation.reset_index(drop=True), 
        tokenizer, 
        MAX_LEN, 
        4
    )
    
    # Show example
    print("\nExample of speaker-aware encoding:")
    example = train_dataset[0]
    print(f"Input shape: {example['input_ids'].shape}")
    print(f"Speaker IDs shape: {example['speaker_ids'].shape}")
    print(f"Unique speakers in example: {torch.unique(example['speaker_ids']).tolist()}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results_speaker_aware_conversational_bert',
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        logging_dir='./logs_speaker_aware',
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
    trainer = SpeakerAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    print("\nStarting training with speaker-aware conversational BERT...")
    print("This combines:")
    print("- DeepPavlov's conversational pre-training")
    print("- Speaker embeddings for multi-speaker understanding")
    print("- Proper handling of Reddit conversation structure")
    
    trainer.train()
    
    # Save the best model
    print("\nSaving model...")
    trainer.save_model('./best_speaker_aware_conversational_bert')
    tokenizer.save_pretrained('./best_speaker_aware_conversational_bert')
    
    # Save speaker embeddings separately for analysis
    torch.save(model.speaker_embeddings.state_dict(), './best_speaker_aware_conversational_bert/speaker_embeddings.pt')
    
    # Evaluate
    print("\nEvaluating on validation set...")
    eval_results = trainer.evaluate()
    print(f"Validation results: {eval_results}")
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = main()