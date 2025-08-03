import torch
from torch.nn import CrossEntropyLoss
from transformers import Trainer
import numpy as np

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.special_token_weight = kwargs.pop('special_token_weight', 1.0)
        super().__init__(*args, **kwargs)
        self.loss_fct = CrossEntropyLoss(ignore_index=-100)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        stacked_labels = inputs.pop("labels")
        labels_full = stacked_labels[:, 0, :]
        labels_special = stacked_labels[:, 1, :]
        
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_labels_full = labels_full.reshape(-1)
        flat_labels_special = labels_special.reshape(-1)

        loss_full = self.loss_fct(flat_logits, flat_labels_full)
        loss_spe = self.loss_fct(flat_logits, flat_labels_special)

        if torch.isnan(loss_spe):
            loss_spe = torch.tensor(0.0, device=logits.device)

        total_loss = loss_spe
        total_loss = loss_full + (self.special_token_weight * loss_spe)
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        output = self.prediction_loop(
            self.get_eval_dataloader(eval_dataset),
            description="Evaluation",
            prediction_loss_only=False,
            ignore_keys=ignore_keys,
        )

        metrics = output.metrics if output.metrics is not None else {}
        
        logits = output.predictions
        stacked_labels = output.label_ids
        labels_special = stacked_labels[:, 1, :]
        preds = np.argmax(logits, axis=-1)

        structurally_complete_samples = 0
        total_samples_with_special_tokens = 0

        for i in range(labels_special.shape[0]):
            sample_labels = labels_special[i]
            sample_preds = preds[i]
            
            special_token_mask = sample_labels != -100

            if not np.any(special_token_mask):
                continue
            
            total_samples_with_special_tokens += 1
            
            true_tokens = sample_labels[special_token_mask]
            pred_tokens = sample_preds[special_token_mask]
            
            if np.array_equal(true_tokens, pred_tokens):
                structurally_complete_samples += 1
        
        structural_completion_rate = 0.0
        if total_samples_with_special_tokens > 0:
            structural_completion_rate = structurally_complete_samples / total_samples_with_special_tokens

        metrics[f"{metric_key_prefix}_accuracy_special"] = structural_completion_rate
        
        self.log(metrics)
        return metrics