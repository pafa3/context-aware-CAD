# Speaker-Aware Conversational BERT for Hate Speech Detection

## Quick Start

### 1. Install Dependencies
```bash
pip install transformers torch pandas numpy scikit-learn matplotlib seaborn datasets
```

### 2. Ensure Dataset Files Exist
Make sure you have these files in the current directory:
- `train.csv`
- `dev.csv` 
- `test.csv`
- `contextual_abuse_dataset4_improved.py` (our improved dataset class)

### 3. Run Training
```bash
python train_final_speaker_aware_bert.py
```

## What This Script Does

1. **Loads DeepPavlov's Conversational BERT** (pre-trained on Reddit/Twitter)
2. **Adds Speaker Embeddings** to track multi-speaker conversations
3. **Fixes all the issues** we identified:
   - No literal [SEP] tokens in text
   - Proper metrics calculation (no `preds > 0.5` bug)
   - Uses cased model for better performance
   - Class weights for imbalanced data
   - Your optimized settings (2 epochs, max_len=300, etc.)

## Results Saved

All results are saved to `./final_results/` with timestamp:

```
final_results/
├── config_20240115_143022.json          # Training configuration
├── predictions_20240115_143022.csv      # Test set predictions
├── classification_report_20240115_143022.json
├── confusion_matrix_20240115_143022.png
├── training_history_20240115_143022.json
├── summary_20240115_143022.json
└── model_20240115_143022/              # Saved model
    ├── pytorch_model.bin
    ├── config.json
    └── tokenizer files...
```

## Key Improvements Over Original

1. **Speaker Awareness**: Each token knows which speaker said it
2. **Better Base Model**: Conversational BERT instead of general BERT
3. **Fixed Bugs**: Metrics calculation and [SEP] token issues
4. **Comprehensive Results**: Saves everything you need for analysis

## Expected Performance

Based on the improvements:
- Should reduce "confusion due to context" errors
- Better handling of multi-speaker conversations
- More accurate on borderline cases

## GPU Requirements

- Memory: ~8-12GB
- Time: ~1-1.5 hours for 2 epochs
- Works with mixed precision (fp16) for faster training

## Customization

Edit the `CONFIG` dictionary in the script to adjust:
- `num_epochs`: Change from 2 if needed
- `batch_size`: Reduce if GPU memory issues
- `max_len`: Already optimized at 300
- `learning_rate`: 3e-5 is good default