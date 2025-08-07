import csv
import pandas as pd
import re
import datasets
from transformers import AutoTokenizer

# Updated to match actual labels in the data
CATEGORY_NAMES = ['Neutral', "IdentityDirectedAbuse", "AffiliationDirectedAbuse", "PersonDirectedAbuse"]

# Paths for the dataset
DATASET_TRAIN_PATH = "train.csv"
DATASET_DEV_PATH = "dev.csv"
DATASET_TEST_PATH = "test.csv"

def get_label_map():
    label_map = {label: i for i, label in enumerate(CATEGORY_NAMES)}
    inv_label_map = {v: k for k, v in label_map.items()}
    return label_map, inv_label_map

def replace_subreddits_usernames(text):
    text = re.sub(r'\/r\/\w+', '[subreddit]', text)
    text = re.sub(r'\/u\/\w+', '[user]', text)
    return text

def replace_urls(text):
    text = re.sub(r"\[([^\[\]]+)\]\((https:\/\/(.*?))\)", r"\1", text)
    text = re.sub(r"\[([^\[\]]+)\]\((\/message\/compose(.*?))\)", r"\1", text)
    text = re.sub(r"\[([^\[\]]+)\]\((\/r\/(.*?))\)", r"\1", text)
    text = re.sub(r'http(s?):\/\/\S+', '[LINK]', text)
    text = re.sub(r'www\.\S+', '[LINK]', text)
    return text

def ignore_entry(s):
    return pd.isna(s) or len(str(s).strip()) == 0 or s in ["[removed]", "[deleted]"]

class ContextualAbuseRedditDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def __init__(self, level=1, *args, **kwargs):
        super(ContextualAbuseRedditDataset, self).__init__(*args, **kwargs)
        self.level = level

    def _info(self):
        return datasets.DatasetInfo(
            description="Reddit Dataset for Contextual Abuse Detection",
            features=datasets.Features({
                "text": datasets.Value("string"),
                "parent_text": datasets.Value("string"),
                "id": datasets.Value("string"),
                "labels_info": datasets.features.ClassLabel(names=CATEGORY_NAMES)
            }),
            supervised_keys=("text", "labels_info")
        )

    def extract_level_1(self, row):
        # Level 1: Current comment only, no context
        text = f"Speaker1: {row['meta_text']}"
        return text, ""

    def extract_level_2(self, row):
        # Level 2: Current comment with its immediate parent
        text = f"Speaker1: {row['meta_text']}"
        parent_text = f"Speaker2: {row.get('parent_text_level_0', '')}" if row.get('parent_text_level_0', '') else ""
        return text, parent_text

    def extract_level_3(self, row):
        # Level 3: Current comment with all preceding comments
        text = f"Speaker1: {row['meta_text']}"
        
        # Build conversation history from oldest to newest
        conversation_parts = []
        for i in range(14, -1, -1):  # Start from oldest (level_14) to newest (level_0)
            parent_text_key = f'parent_text_level_{i}'
            parent_text = row.get(parent_text_key, '')
            if parent_text and not pd.isna(parent_text) and str(parent_text).strip():
                # Alternate speaker labels based on depth
                speaker_num = 2 + (i % 2)
                conversation_parts.append(f"Speaker{speaker_num}: {parent_text}")
        
        # Join all parts with space (no [SEP] tokens!)
        parent_text = " ".join(conversation_parts) if conversation_parts else ""
        return text, parent_text

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": DATASET_TRAIN_PATH},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": DATASET_DEV_PATH},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": DATASET_TEST_PATH},
            ),
        ]

    def _generate_examples(self, filepath):
        label_map = get_label_map()[0]
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for idx, row in enumerate(reader):
                # Skip entries that should be ignored
                if ignore_entry(row.get('meta_text', '')):
                    continue
                
                # Get the info_id as unique identifier
                info_id = row.get('info_id', f'unknown_{idx}')
                
                # Process text based on the level
                if self.level == 1:
                    text, parent_text = self.extract_level_1(row)
                elif self.level == 2:
                    text, parent_text = self.extract_level_2(row)
                else:
                    text, parent_text = self.extract_level_3(row)
                
                # Apply preprocessing
                text = replace_subreddits_usernames(text).replace('[linebreak]', "\n").strip()
                text = replace_urls(text)
                parent_text = replace_subreddits_usernames(parent_text).replace('[linebreak]', "\n").strip()
                parent_text = replace_urls(parent_text)
                
                # Handle labels - map to our categories
                annotation = row.get('annotation_Primary', 'Neutral')
                
                # Map Slur and CounterSpeech to Neutral as in original code
                if annotation in ['Slur', 'CounterSpeech', '']:
                    annotation = 'Neutral'
                
                # Only yield if we have a valid label
                if annotation in CATEGORY_NAMES:
                    label_id = label_map[annotation]
                    
                    yield info_id, {
                        'text': text,
                        'parent_text': parent_text,
                        'id': info_id,
                        'labels_info': label_id,
                    }