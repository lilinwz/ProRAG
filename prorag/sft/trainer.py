import torch
from torch.nn import CrossEntropyLoss
from transformers import Trainer
import numpy as np
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
import re

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        default_weight = 1.0
        self.token_weights = {
            "<answer>": 5.0, 
            "</answer>": 5.0,
            "<subquery>": 3.0, 
            "</subquery>": 3.0,
            "<subanswer>": 3.0, 
            "</subanswer>": 3.0,
            "<step>": 1.0, 
            "</step>": 2.0,
            "<retrieval>": 1.0, 
            "</retrieval>": 1.0,
        }
        vocab_size = len(self.processing_class)
        self.cost_weights = torch.full((vocab_size,), default_weight)

        self.special_token_map = {}
        for token_str, weight in self.token_weights.items():
            token_id = self.processing_class.convert_tokens_to_ids(token_str)
            if token_id != self.processing_class.unk_token_id:
                self.cost_weights[token_id] = weight
                self.special_token_map[token_id] = token_str
        
        self.loss_fct = CrossEntropyLoss(weight=self.cost_weights, ignore_index=-100)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        if self.loss_fct.weight.device != shift_logits.device:
            self.loss_fct.weight = self.loss_fct.weight.to(shift_logits.device)
        if self.loss_fct.weight.dtype != shift_logits.dtype:
            self.loss_fct.weight = self.loss_fct.weight.to(shift_logits.dtype)

        loss = self.loss_fct(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
    @torch.no_grad()
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        dataloader = self.get_eval_dataloader(eval_dataset)
        model = self.model
        model.eval()

        print(f"\n[Starting Evaluation] Special Token Check...")
        
        MAX_DEBUG_BATCH = 20
        count = 0
        preds_texts, losses = [], []
        for batch in dataloader:
            if count >= MAX_DEBUG_BATCH:
                break
            count += 1

            batch = self._prepare_inputs(batch)
            labels = batch["labels"]
            outputs = model(**batch)
            logits = outputs.get("logits")
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = self.loss_fct(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))
            losses.append(loss.item())

            input_ids = batch["input_ids"]
            
            prompt_texts = []
            decoded_inputs = self.processing_class.batch_decode(input_ids, skip_special_tokens=False)
            for text in decoded_inputs:
                prompt_len = text.rfind("<step>\n")
                prompt = text[:prompt_len + len("<step>\n")]
                prompt_texts.append(prompt)

            gen_inputs = self.processing_class(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096
            ).to(model.device)

            gen_outputs = model.generate(
                **gen_inputs,
                max_new_tokens=4096,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=self.processing_class.eos_token_id,
                pad_token_id=self.processing_class.pad_token_id
            )

            gen_texts = self.processing_class.batch_decode(gen_outputs, skip_special_tokens=False)
            pure_generations = [text[len(prompt):] for prompt, text in zip(prompt_texts, gen_texts)]
            preds_texts.extend(pure_generations)
        
        avg_loss = np.mean(losses)
        metrics = {f"{metric_key_prefix}_loss": avg_loss}

        def format_check(completion):
            tags = re.findall(r"</?[a-zA-Z]+>", completion)

            if tags[:3] == ["</step>", "<subquery>", "</subquery>"]:
                return True
            if tags[:3] == ["</step>", "<subanswer>", "</subanswer>"]:
                return True
            if tags[:3] == ["</step>", "<answer>", "</answer>"]:
                return True

            return False

        total = len(preds_texts)
        correct = sum(1 for text in preds_texts if format_check(text))
        format_acc = correct / total if total > 0 else 0.0

        metrics[f"{metric_key_prefix}_format_acc"] = format_acc

        report = (
            "\n--- Format Evaluation Report ---\n"
            f"Samples: {total} Correct format: {correct}/{total} ({format_acc:.2%})\n"
            f"Sample: {preds_texts[0]}\n"
            "--------------------------------\n"
        )
        # print(report)

        self.log(metrics)
        return metrics