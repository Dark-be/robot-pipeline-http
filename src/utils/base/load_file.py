import pickle
import yaml
import os
import json
from pathlib import Path

def load_yaml(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_pkl(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def load_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    p = Path(file_path)
    with p.open("r", encoding="utf-8") as f:
        s = f.read().strip()
        if s == "":
            return {}
        return json.loads(s)

