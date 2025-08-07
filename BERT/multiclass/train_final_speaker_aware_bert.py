"""
Final production-ready script for Speaker-Aware Conversational BERT
Combines all improvements and saves comprehensive results
"""

import torch
import torch.nn as nn
import random
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    precision_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Set seed for reproducibility
def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed
SEED = 42
set_seed(SEED)

# Configuration
CONFIG = {
    "model_name": "DeepPavlov/bert-base-cased-conversational",
    "max_len": 300,  # Based on your 90% coverage analysis
    "batch_size": 16,
    "learning_rate": 3e-5,
    "num_epochs": 2,  # As you correctly noted, 2 is fine for BERT
    "warmup_steps": 100,  # Your calculation was correct
    "weight_decay": 0.01,  # BERT paper recommendation
    "max_speakers": 50,
    "num_labels": 4,
    "dropout_rate": 0.1,
    "gradient_accumulation_steps": 1,
    "fp16": True,  # Mixed precision training
    "seed": SEED,
    "output_dir": "./results_final_speaker_aware_bert",
    "logging_dir": "./logs_final",
    "save_results_dir": "./final_results"
}

# Category names for your task
CATEGORY_NAMES = ['Neutral', 'IdentityDirectedAbuse', 'AffiliationDirectedAbuse', 'PersonDirectedAbuse']

class SpeakerAwareConversationalBERT(nn.Module):
    """
    Combines DeepPavlov's Conversational BERT with speaker embeddings
    """
    
    def __init__(self, config_dict):
        super().__init__()
        
        # Load the conversational BERT model
        self.config = AutoConfig.from_pretrained(config_dict["model_name"])
        self.bert = AutoModel.from_pretrained(config_dict["model_name"])
        
        # Add speaker embeddings
        self.speaker_embeddings = nn.Embedding(
            num_embeddings=config_dict["max_speakers"],
            embedding_dim=self.config.hidden_size
        )
        
        # Initialize speaker embeddings
        nn.init.normal_(self.speaker_embeddings.weight, mean=0.0, std=0.02)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config_dict["dropout_rate"])
        
        # Classification head
        self.classifier = nn.Linear(self.config.hidden_size, config_dict["num_labels"])
        
    def forward(self, input_ids, attention_mask, speaker_ids, labels=None):
        # Get word embeddings from BERT
        word_embeddings = self.bert.embeddings.word_embeddings(input_ids)
        
        # Get position embeddings
        position_ids = torch.arange(input_ids.size(1), device=input_ids.device)
        position_embeddings = self.bert.embeddings.position_embeddings(position_ids)
        
        # Get token type embeddings
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
        pooled_output = sequence_output[:, 0]
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Use class weights if provided
            if hasattr(self, 'class_weights'):
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

class SpeakerAwareDataset(Dataset):
    """Dataset that properly tracks speaker IDs for each token"""
    
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
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
        if parent_text and parent_text.strip() and parent_text != 'nan':
            parts = parent_text.split("Speaker")
            for part in parts:
                if part.strip():
                    # Extract speaker number and text
                    if len(part) > 3 and part[0].isdigit() and part[1:3] == ": ":
                        speaker_num = int(part[0])
                        utterance = part[3:].strip()
                        
                        # Tokenize utterance
                        utterance_tokens = self.tokenizer.tokenize(utterance)
                        tokens.extend(utterance_tokens)
                        speaker_ids.extend([speaker_num] * len(utterance_tokens))
        
        # Process current text (always Speaker1)
        if text.startswith("Speaker1: "):
            current_text = text.replace("Speaker1: ", "").strip()
        else:
            current_text = text.strip()
            
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
        
        # Get label - it's now just an integer
        label = self.data.labels_info.iloc[index]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'speaker_ids': torch.tensor(speaker_ids, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    """Fixed compute metrics function"""
    predictions, labels = eval_pred
    # For multi-class classification, just use argmax (no > 0.5!)
    preds = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    accuracy = accuracy_score(labels, preds)
    
    # Also calculate per-class metrics
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }
    
    # Add per-class F1 scores
    for i, class_name in enumerate(CATEGORY_NAMES):
        if i < len(per_class_f1):
            metrics[f'f1_{class_name.lower()}'] = per_class_f1[i]
    
    return metrics

def calculate_class_weights(labels):
    """Calculate class weights for imbalanced dataset"""
    from collections import Counter
    label_counts = Counter(labels)
    total_count = sum(label_counts.values())
    class_weights = {label: total_count / (len(label_counts) * count) 
                    for label, count in label_counts.items()}
    weights = [class_weights[i] for i in range(len(class_weights))]
    return torch.FloatTensor(weights)

def load_data():
    """Load and prepare the datasets"""
    print("Loading datasets...")
    
    # Import the improved dataset class
    import contextual_abuse_dataset4_improved
    from contextual_abuse_dataset4_improved import ContextualAbuseRedditDataset
    
    # Load with level 3 (all context)
    dataset_builder = ContextualAbuseRedditDataset(level=3)
    dataset_builder.download_and_prepare()
    dataset = dataset_builder.as_dataset()
    
    # Convert to DataFrames
    df_train = pd.DataFrame(dataset["train"])
    df_validation = pd.DataFrame(dataset["validation"]) 
    df_test = pd.DataFrame(dataset["test"])
    
    # No need to filter - the dataset class already handles this
    
    print(f"Train samples: {len(df_train)}")
    print(f"Validation samples: {len(df_validation)}")
    print(f"Test samples: {len(df_test)}")
    
    return df_train, df_validation, df_test

# Custom Trainer to handle our model's output format
class SpeakerAwareTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

def save_results(trainer, model, tokenizer, df_test, config):
    """Save all results comprehensively"""
    print("\nSaving results...")
    
    # Create results directory
    os.makedirs(config["save_results_dir"], exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save configuration
    with open(f"{config['save_results_dir']}/config_{timestamp}.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # 2. Create test dataset and get predictions
    test_dataset = SpeakerAwareDataset(df_test, tokenizer, config["max_len"])
    predictions = trainer.predict(test_dataset)
    
    # 3. Save raw predictions
    pred_labels = np.argmax(predictions.predictions, axis=1)
    pred_probs = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
    
    # Create results DataFrame
    results_df = df_test.copy()
    results_df['predicted_label'] = pred_labels
    results_df['predicted_class'] = [CATEGORY_NAMES[i] for i in pred_labels]
    
    # Add prediction probabilities
    for i, class_name in enumerate(CATEGORY_NAMES):
        results_df[f'prob_{class_name}'] = pred_probs[:, i]
    
    # Save to CSV
    results_df.to_csv(f"{config['save_results_dir']}/predictions_{timestamp}.csv", index=False)
    
    # 4. Calculate and save metrics
    true_labels = df_test['labels_info'].tolist()
    
    # Classification report
    report = classification_report(true_labels, pred_labels, 
                                 target_names=CATEGORY_NAMES, 
                                 output_dict=True)
    
    with open(f"{config['save_results_dir']}/classification_report_{timestamp}.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=CATEGORY_NAMES))
    
    # 5. Save confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CATEGORY_NAMES, 
                yticklabels=CATEGORY_NAMES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{config['save_results_dir']}/confusion_matrix_{timestamp}.png")
    plt.close()
    
    # 6. Save model and tokenizer
    model_save_path = f"{config['save_results_dir']}/model_{timestamp}"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # 7. Save training history
    if trainer.state.log_history:
        with open(f"{config['save_results_dir']}/training_history_{timestamp}.json", 'w') as f:
            json.dump(trainer.state.log_history, f, indent=2)
    
    # 8. Create summary report
    summary = {
        "timestamp": timestamp,
        "model_name": config["model_name"],
        "test_metrics": predictions.metrics,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "speaker_embedding_parameters": sum(p.numel() for p in model.speaker_embeddings.parameters()),
    }
    
    with open(f"{config['save_results_dir']}/summary_{timestamp}.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAll results saved to {config['save_results_dir']}/")
    print(f"Timestamp: {timestamp}")
    
    return results_df, summary

def main():
    """Main training function"""
    print("=" * 70)
    print("SPEAKER-AWARE CONVERSATIONAL BERT FOR HATE SPEECH DETECTION")
    print("=" * 70)
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    
    # Load data
    df_train, df_validation, df_test = load_data()
    
    # Calculate class weights
    print("\nCalculating class weights...")
    all_labels = df_train['labels_info'].tolist()
    class_weights = calculate_class_weights(all_labels)
    print(f"Class weights: {class_weights}")
    
    # Create model
    print("\nInitializing model...")
    model = SpeakerAwareConversationalBERT(CONFIG)
    model.class_weights = class_weights.to(device)
    model.to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Speaker embedding parameters: {sum(p.numel() for p in model.speaker_embeddings.parameters()):,}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = SpeakerAwareDataset(df_train, tokenizer, CONFIG["max_len"])
    val_dataset = SpeakerAwareDataset(df_validation, tokenizer, CONFIG["max_len"])
    
    # Show example
    print("\nExample of speaker-aware encoding:")
    example = train_dataset[0]
    print(f"Input shape: {example['input_ids'].shape}")
    print(f"Speaker IDs shape: {example['speaker_ids'].shape}")
    print(f"Unique speakers: {torch.unique(example['speaker_ids']).tolist()}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        warmup_steps=CONFIG["warmup_steps"],
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        logging_dir=CONFIG["logging_dir"],
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=400,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=CONFIG["fp16"] and torch.cuda.is_available(),
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        lr_scheduler_type='linear',  # Simple linear decay as you suggested
        report_to="none",  # Disable wandb
        push_to_hub=False,
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
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Total steps: {len(train_dataset) // CONFIG['batch_size'] * CONFIG['num_epochs']}")
    print("=" * 70)
    
    trainer.train()
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    eval_results = trainer.evaluate()
    print("Validation results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Save all results
    results_df, summary = save_results(trainer, model, tokenizer, df_test, CONFIG)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best F1 score: {eval_results.get('eval_f1', 0):.4f}")
    print(f"Test accuracy: {summary['test_metrics'].get('test_accuracy', 0):.4f}")
    print(f"Results saved to: {CONFIG['save_results_dir']}/")
    print("=" * 70)
    
    return model, tokenizer, results_df

if __name__ == "__main__":
    model, tokenizer, results = main()