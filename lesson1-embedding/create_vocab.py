import os
import numpy as np
import json
import hebrew_tokenizer as ht
from collections import Counter

EMBEDDING_DIM = 600
TRAIN_DATA_PATH = "train.json"
def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
    return data

def create_vocab(data, vocab_size=10000, file_path="vocabulary.json"):
    print("Tokenizing data...")
    all_tokens = []
    for i, row in enumerate(data):
        text = row.get('text', '')
        if text:
            tokens = ht.tokenize(text)
            word_tokens = [token[1] for token in tokens if token[0] == 'HEBREW'] # Keep only Hebrew words
            all_tokens.extend(word_tokens)
        if (i + 1) % 1000 == 0:
            print(f"Processed {i+1}/{len(data)} rows...")
    

    print(f"Finished tokenizing. Total tokens: {len(all_tokens)}")
    word_counts = Counter(all_tokens)
    most_common_words = [word for word, count in word_counts.most_common(vocab_size - 1)]
    word_to_ix = {word: i for i, word in enumerate(most_common_words)}
    print(f"Vocabulary size: {len(word_to_ix)}")

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(word_to_ix, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    print("Loading data...")
    data = load_data(TRAIN_DATA_PATH)
    create_vocab(data)
