from pathlib import Path
import ast
import json
import random

import numpy as np
import torch

from genrec.ghost.utils import infer_tokenizer_kind, tokens_key


def load_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def tokens_to_item_id(
    tokens_sequence, 
    tokens_to_item_map
):
    if not tokens_to_item_map:
        return None
    if torch.is_tensor(tokens_sequence):
        tokens_list = tokens_sequence.tolist()
    else:
        tokens_list = tokens_sequence

    tokens_tuple = tokens_key(tokens_list)
    return tokens_to_item_map.get(tokens_tuple, None)


def tokenizer_artifacts_ready(save_path: str, tokenizer_kind: str | None = None) -> bool:
    item2tokens_path = Path(save_path)
    if not item2tokens_path.exists():
        return False

    kind = infer_tokenizer_kind(tokenizer_kind or "")
    if kind != "ghost":
        return True

    metadata_path = Path(str(item2tokens_path).replace("item2tokens.json", "ghost_metadata.json"))
    undesired_collection_path = Path(str(item2tokens_path).replace(".json", "_undesired_collection.json"))
    return metadata_path.exists() and undesired_collection_path.exists()



def create_token_to_item_mapping(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    token_to_item_map = {}
    
    for key, value in data.items():
        try:
            tokens_tuple = ast.literal_eval(key)
            tokens_list = list(tokens_tuple)
        except:
            cleaned_key = key.strip('()').replace(' ', '')
            tokens_list = [int(x) for x in cleaned_key.split(',') if x]
        

        token_to_item_map[tuple(tokens_list)] = value

    return token_to_item_map

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
