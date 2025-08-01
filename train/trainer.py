# trainer.py
from torch.nn import CrossEntropyLoss
from transformers import Trainer
import torch # 确保导入 torch

class CustomTrainer(Trainer):
    def __init__(self, *args, special_token_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.special_token_weight = special_token_weight
        # CrossEntropyLoss 默认的 reduction 是 'mean'
        self.loss_fct = CrossEntropyLoss(ignore_index=-100)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_full = inputs.pop("labels_full")
        labels_spe = inputs.pop("labels_special")

        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss_full = self.loss_fct(logits.view(-1, logits.size(-1)), labels_full.view(-1))
        loss_spe = self.loss_fct(logits.view(-1, logits.size(-1)), labels_spe.view(-1))

        if torch.isnan(loss_spe):
            loss_spe = torch.tensor(0.0, device=logits.device)
            
        total_loss = loss_full + (self.special_token_weight * loss_spe)
        
        return (total_loss, outputs) if return_outputs else total_loss