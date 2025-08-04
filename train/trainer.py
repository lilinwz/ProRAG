import torch
from torch.nn import CrossEntropyLoss
from transformers import Trainer
import numpy as np

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.special_token_weight = kwargs.pop('special_token_weight', 1.0)
        super().__init__(*args, **kwargs)

        weights = torch.ones(self.model.config.vocab_size).to(self.model.device)
        custom_tokens = [
            "<think>", "</think>", "<subquery>", "</subquery>",
            "<retrieval>", "</retrieval>", "<subanswer>", "</subanswer>",
            "<answer>", "</answer>"
        ]
        special_token_ids = self.tokenizer.convert_tokens_to_ids(custom_tokens)
        for token_id in special_token_ids:
            if token_id != self.tokenizer.unk_token_id:
                weights[token_id] = self.special_token_weight

        self.loss_fct = CrossEntropyLoss(weight=weights.to(self.model.device), ignore_index=-100)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_for_eval = inputs.pop("labels_for_eval", None)
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss = self.loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        output = self.prediction_loop(
            self.get_eval_dataloader(eval_dataset),
            description="Evaluation",
            prediction_loss_only=False,
            ignore_keys=ignore_keys,
        )

        metrics = {}
        if output.metrics is not None:
             metrics[f"{metric_key_prefix}_loss"] = output.metrics.get('eval_loss')
        
        logits = output.predictions
        labels_special = np.array(eval_dataset['labels_for_eval'])
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