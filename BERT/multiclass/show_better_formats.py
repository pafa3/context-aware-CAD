"""
Better ways to format multi-speaker conversations for BERT
"""

def show_better_formatting_options():
    print("=== BETTER CONVERSATION FORMATTING OPTIONS ===\n")
    
    # Example conversation
    current_text = "Stay in China and milk your inflated income for as long as you can."
    parent_0 = "how jeow BOO jenn"
    parent_1 = "What does this even mean?"
    parent_2 = "It's a transliteration of a Chinese phrase"
    
    print("OPTION 1: Using Special Tokens (Best if you can add them to tokenizer)")
    print("-" * 70)
    print("Add custom tokens to the tokenizer: [SPEAKER1], [SPEAKER2], [TURN]")
    formatted_1 = f"[SPEAKER2] {parent_2} [TURN] [SPEAKER1] {parent_1} [TURN] [SPEAKER2] {parent_0} [TURN] [SPEAKER1] {current_text}"
    print(f"Format: {formatted_1}")
    print("✅ Clear speaker boundaries with special tokens")
    print("✅ Model learns these tokens during training")
    print("❌ Requires modifying tokenizer vocabulary\n")
    
    print("\nOPTION 2: Using Segment IDs (What BERT was designed for)")
    print("-" * 70)
    print("Use BERT's token_type_ids to distinguish speakers:")
    print("Text A (segment 0): All Speaker2 utterances concatenated")
    print("Text B (segment 1): All Speaker1 utterances concatenated")
    text_a = f"{parent_2} {parent_0}"  # All Speaker2 utterances
    text_b = f"{parent_1} {current_text}"  # All Speaker1 utterances
    print(f"Text A: {text_a}")
    print(f"Text B: {text_b}")
    print("✅ Uses BERT's built-in two-segment architecture")
    print("✅ Clear separation between speakers")
    print("❌ Limited to two speakers\n")
    
    print("\nOPTION 3: Using Punctuation as Natural Boundaries")
    print("-" * 70)
    formatted_3 = f"Speaker2: {parent_2}. Speaker1: {parent_1}. Speaker2: {parent_0}. Speaker1: {current_text}."
    print(f"Format: {formatted_3}")
    print("✅ Natural sentence boundaries with periods")
    print("✅ No special tokens needed")
    print("❌ Still just one text block\n")
    
    print("\nOPTION 4: Structured with Clear Delimiters")
    print("-" * 70)
    formatted_4 = f"Context: {parent_2} | {parent_1} | {parent_0} | Current: {current_text}"
    print(f"Format: {formatted_4}")
    print("✅ Clear structure with | delimiters")
    print("✅ Distinguishes context from current")
    print("❌ Model needs to learn what | means\n")
    
    print("\nOPTION 5: Dialogue Format (Like a Script)")
    print("-" * 70)
    formatted_5 = f"A: {parent_2}\nB: {parent_1}\nA: {parent_0}\nB: {current_text}"
    print(f"Format:\n{formatted_5}")
    print("✅ Natural dialogue format")
    print("✅ Clear turn-taking with newlines")
    print("✅ Similar to training data format\n")
    
    print("\nRECOMMENDED APPROACH FOR YOUR CASE:")
    print("-" * 70)
    print("Since you're using DeepPavlov's Conversational BERT:")
    print("1. Use Option 2 (segment IDs) if you have mainly 2-party conversations")
    print("2. Use Option 5 (dialogue format) for multi-party conversations")
    print("3. Consider fine-tuning with special tokens if you have time\n")
    
    # Show how to implement Option 2 with tokenizer
    print("\nIMPLEMENTATION EXAMPLE (Option 2 - Using Segments):")
    print("-" * 70)
    print("# When tokenizing:")
    print("inputs = tokenizer(")
    print("    text=speaker1_utterances,  # First segment")
    print("    text_pair=speaker2_utterances,  # Second segment") 
    print("    add_special_tokens=True,")
    print("    max_length=512,")
    print("    truncation=True")
    print(")")
    print("\nThis creates: [CLS] speaker1_text [SEP] speaker2_text [SEP]")
    print("With token_type_ids: [0, 0, ..., 0, 0, 1, 1, ..., 1, 1]")
    print("                      └─ segment A ─┘ └─ segment B ─┘")

if __name__ == "__main__":
    show_better_formatting_options()