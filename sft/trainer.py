import torch
from torch.nn import CrossEntropyLoss
from transformers import Trainer
import numpy as np
from collections import defaultdict
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
            "<step>": 2.0, 
            "</step>": 2.0,
            "<retrieval>": 3.0, 
            "</retrieval>": 3.0,
        }
        vocab_size = len(self.processing_class)
        weights = torch.full((vocab_size,), default_weight, device=self.model.device)

        self.special_token_map = {}
        for token_str, weight in self.token_weights.items():
            token_id = self.processing_class.convert_tokens_to_ids(token_str)
            if token_id != self.processing_class.unk_token_id:
                weights[token_id] = weight
                self.special_token_map[token_id] = token_str
        
        self.loss_fct = CrossEntropyLoss(weight=weights, ignore_index=-100)

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

        preds_texts, prompt_texts, losses = [], [], []
        for batch in dataloader:
            batch = self._prepare_inputs(batch)
            labels = batch["labels"]
            outputs = model(**batch)
            logits = outputs.get("logits")

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_fct(
                shift_logits.view(-1, self.model.config.vocab_size),
                shift_labels.view(-1)
            )
            losses.append(loss.item())

            input_ids = batch["input_ids"]
            decoded_inputs = self.processing_class.batch_decode(input_ids, skip_special_tokens=False)
            for text in decoded_inputs:
                match = re.search(r"(<\|im_start\|>assistant\n<think>\n</think>\n)", text)
                if match:
                    prompt = text[:match.end()]
                else:
                    prompt = text
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
                pad_token_id=self.processing_class.eos_token_id
            )

            gen_texts = self.processing_class.batch_decode(gen_outputs, skip_special_tokens=False)
            preds_texts.extend(gen_texts)

        avg_loss = np.mean(losses)
        metrics = {f"{metric_key_prefix}_loss": avg_loss}

        def check_format_correctness(text: str) -> bool:
            tag_pattern = re.compile(r"</?\s*(step|subquery|retrieval|subanswer|answer)\s*>", flags=re.IGNORECASE)
            tags_matches = list(tag_pattern.finditer(text))
            if not tags_matches:
                return False

            tag_seq = []
            for m in tags_matches:
                raw = m.group(0)
                name = m.group(1).lower()
                is_closing = raw.strip().startswith("</")
                tag_seq.append((is_closing, name))

            stack = []
            for is_closing, name in tag_seq:
                if not is_closing:
                    stack.append(name)
                else:
                    if not stack:
                        return False
                    top = stack.pop()
                    if top != name:
                        return False
            if stack:
                return False

            opening_tags = [name for is_closing, name in tag_seq if not is_closing]
            if not opening_tags:
                return False

            i = 0
            n = len(opening_tags)
            while i < n:
                if opening_tags[i] != "step":
                    return False
                i += 1

                if i < n and opening_tags[i] == "answer":
                    return i == n - 1
                expected_seq = ["subquery", "retrieval", "step", "subanswer"]
                for expected in expected_seq:
                    if i >= n or opening_tags[i] != expected:
                        return False
                    i += 1

            return True

        total = len(preds_texts)
        correct = sum(1 for text in preds_texts if check_format_correctness(text))
        format_acc = correct / total if total > 0 else 0.0

        metrics[f"{metric_key_prefix}_format_acc"] = format_acc

        report = (
            "\n--- Format Evaluation Report ---\n"
            f"Samples: {total} Correct format: {correct}/{total} ({format_acc:.2%})\n"
            f"Sample: {preds_texts[0]}\n"
            "--------------------------------\n"
        )
        print(report)

        self.log(metrics)
        return metrics