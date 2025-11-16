import os
import json
import logging
from typing import List


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj):
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'w', encoding='utf8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)