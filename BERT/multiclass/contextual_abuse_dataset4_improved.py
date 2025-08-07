import csv
import pandas as pd
import re
import datasets
from transformers import AutoTokenizer

CATEGORY_NAMES = ['Neutral', "IdentityDirectedAbuse", "AffiliationDirectedAbuse", "PersonDirectedAbuse"]

# Paths for the dataset
DATASET_FULL_PATH = "cad_v1_1.tsv"
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
    return pd.isna(s) or len(s.strip()) == 0 or s in ["[removed]", "[deleted]"]

class ContextualAbuseRedditDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def __init__(self, level=1, *args, **kwargs):
        super(ContextualAbuseRedditDataset, self).__init__(*args, **kwargs)
        self.level = level

    def _info(self):
        label_map_dict = get_label_map()[0]
        return datasets.DatasetInfo(
            description="Reddit Dataset for Contextual Abuse Detection",
            features=datasets.Features({
                "text": datasets.Value("string"),
                "parent_text": datasets.Value("string"),
                "id": datasets.Value("string"),
                "labels_info": datasets.features.Sequence({
                    "label": datasets.ClassLabel(names=list(label_map_dict.keys()))
                })
            }),
            supervised_keys=("text", "labels_info")
        )

    def extract_level_1(self, row):
        # Level 1: Current comment only, no context
        # Just the current speaker's text
        text = f"Speaker1: {row['meta_text']}"
        return text, ""

    def extract_level_2(self, row):
        # Level 2: Current comment with its immediate parent
        # Format: current text, parent as context
        text = f"Speaker1: {row['meta_text']}"
        parent_text = f"Speaker2: {row.get('parent_text_level_0', '')}" if row.get('parent_text_level_0', '') else ""
        return text, parent_text

    def extract_level_3(self, row):
        # Level 3: Current comment with all preceding comments
        # We'll concatenate all context into parent_text with proper speaker labels
        text = f"Speaker1: {row['meta_text']}"
        
        # Build conversation history from oldest to newest
        conversation_parts = []
        for i in range(14, -1, -1):  # Start from oldest (level_14) to newest (level_0)
            parent_text_key = f'parent_text_level_{i}'
            parent_text = row.get(parent_text_key, '')
            if parent_text:
                # Alternate speaker labels based on depth
                # Even levels = Speaker2, Odd levels = Speaker1 (to show alternating conversation)
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
        df = pd.read_csv(filepath, sep=',')
        df = df.fillna('')

        id_to_data = {}
        for id_, row in df.iterrows():
            info_id = row['info_id']

            # Process text based on the level
            if self.level == 1:
                text, parent_text = self.extract_level_1(row)
            elif self.level == 2:
                text, parent_text = self.extract_level_2(row)
            else:
                text, parent_text = self.extract_level_3(row)

            # Apply preprocessing to both text and parent_text
            text = replace_subreddits_usernames(text).replace('[linebreak]', "\n").replace("\n ", "\n").strip()
            text = replace_urls(text)
            parent_text = replace_subreddits_usernames(parent_text).replace('[linebreak]', "\n").replace("\n ", "\n").strip()
            parent_text = replace_urls(parent_text)

            # Skip entries that should be ignored
            if ignore_entry(row['meta_text']):
                continue

            if info_id not in id_to_data:
                id_to_data[info_id] = {
                    'text': text,
                    'parent_text': parent_text,
                    'labels': set(),
                }

            # Handle labels
            labels = []
            if pd.isna(row['annotation_Primary']) or row['annotation_Primary'] in ['Slur', 'CounterSpeech']:
                labels.append('Neutral')
            else:
                labels.append(row['annotation_Primary'])

            id_to_data[info_id]['labels'].update(labels)

        label_map = get_label_map()[0]
        for info_id, data in id_to_data.items():
            labels_info = [{'label': label_map[label]} for label in data['labels']]
            yield info_id, {
                'text': data['text'],
                'parent_text': data['parent_text'],
                'id': info_id,
                'labels_info': labels_info,
            }