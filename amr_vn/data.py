
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Iterable
import json

from .utils.penman import parse_amr_blocks

class AMRCorpusLoader:
    """
    Load one or many AMR training files and build samples for training.
    Each sample is a dict with keys: {"input": <sentence>, "output": <amr>}
    """
    def __init__(self, paths: List[Path]):
        self.paths = paths

    def load(self) -> List[Dict[str, str]]:
        samples: List[Dict[str, str]] = []
        for p in self.paths:
            with open(p, "r", encoding="utf-8") as f:
                for sent, amr in parse_amr_blocks(f):
                    samples.append({"input": sent, "output": amr})
        return samples

    @staticmethod
    def save_json(samples: List[Dict[str, str]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fo:
            json.dump(samples, fo, ensure_ascii=False, indent=2)

class PublicTestLoader:
    """
    Load public_test file. Treat each non-empty line as a test item.
    """
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> List[str]:
        items: List[str] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if t:
                    items.append(t)
        return items
