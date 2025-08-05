import torch
from torch.nn import CrossEntropyLoss
from transformers import Trainer
import numpy as np
from collections import defaultdict

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.special_token_weight_config = kwargs.pop('special_token_weight', None)
        super().__init__(*args, **kwargs)

        default_weight = 1.0
        self.token_weights = {
            "<answer>": 20.0, 
            "</answer>": 20.0,
            "<subquery>": 10.0, 
            "</subquery>": 10.0,
            "<subanswer>": 10.0, 
            "</subanswer>": 10.0,
            "<step>": 4.0, 
            "</step>": 4.0,
            "<retrieval>": 10.0, 
            "</retrieval>": 10.0,
        }

        weights = torch.full((self.model.config.vocab_size,), default_weight).to(self.model.device)

        special_token_ids = []
        for token_str, weight in self.token_weights.items():
            token_id = self.tokenizer.convert_tokens_to_ids(token_str)
            if token_id != self.tokenizer.unk_token_id:
                weights[token_id] = weight
                special_token_ids.append(token_id)
        
        self.loss_fct = CrossEntropyLoss(weight=weights, ignore_index=-100)

        self.special_token_map = {
            self.tokenizer.convert_tokens_to_ids(token): token for token in self.token_weights.keys()
        }

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
        labels = np.array(eval_dataset['labels']) 
        preds = np.argmax(logits, axis=-1)

        true_counts = defaultdict(int)
        pred_counts = defaultdict(int)

        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                label_token_id = labels[i, j]
                pred_token_id = preds[i, j]

                if label_token_id in self.special_token_map:
                    true_counts[label_token_id] += 1
                    
                    if pred_token_id == label_token_id:
                        pred_counts[label_token_id] += 1
        
        print("\n--- Special Token Accuracy Report ---")
        for token_id, count in sorted(true_counts.items()):
            token_str = self.special_token_map[token_id]
            accuracy = (pred_counts[token_id] / count) if count > 0 else 0
            metrics[f"{metric_key_prefix}_acc_{token_str}"] = accuracy
            print(f"Token: {token_str:<12} | Accuracy: {accuracy:>7.2%} | Count: {count}")
        print("-------------------------------------\n")

        self.log(metrics)
        return metrics