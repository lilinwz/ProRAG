import torch
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class CustomDataCollator:
    def __call__(self, features: list[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {
            'input_ids': torch.tensor([f['input_ids'] for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f['attention_mask'] for f in features], dtype=torch.long),
            'labels_full': torch.tensor([f['labels_full'] for f in features], dtype=torch.long),
            'labels_special': torch.tensor([f['labels_special'] for f in features], dtype=torch.long),
        }
        return batch