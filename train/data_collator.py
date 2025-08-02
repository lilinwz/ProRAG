import torch
from typing import Any, Dict, List

class CustomDataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = torch.stack([torch.tensor(f["input_ids"]) for f in features])
        attention_mask = torch.stack([torch.tensor(f["attention_mask"]) for f in features])

        labels_full = torch.stack([torch.tensor(f["labels_full"]) for f in features])
        labels_special = torch.stack([torch.tensor(f["labels_special"]) for f in features])
        
        labels = torch.stack([labels_full, labels_special], dim=1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }