import torch
import torch.nn.functional as F
from trl import GRPOTrainer
from trl.trainer.utils import pad, nanmin, nanstd, nanmax
from trl.models.utils import disable_gradient_checkpointing
from accelerate.utils import gather_object
from typing import Any
import copy

class RAGTrainer(GRPOTrainer):
    def __init__(self, reward_model=None, prm_beta=0.5, retriever=None, **kwargs):
        super().__init__(**kwargs)
        self.prm_model = reward_model
        self.prm_beta = prm_beta
        self.retriever = retriever
        
        if self.prm_model:
            self.prm_model.eval()
            self.prm_model.to(self.accelerator.device)

    def get_prm_score(self, text):
        if not self.prm_model:
            return 0.0
        
        device = self.accelerator.device
        tokenizer = self.reward_processing_classes[0] if isinstance(self.reward_processing_classes, list) else self.reward_processing_classes
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)
        with torch.no_grad():
            outputs = self.prm_model(**inputs)
            score = torch.sigmoid(outputs.logits[0]).item()
        return score
    
    def _generate(self, prompts: list):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = copy.deepcopy(prompts)
        prompt_ids, completion_ids, logprobs, extra_fields = self._generate_single_turn(prompts)
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=False)
        
        tool_mask = None
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()
        total_completion_tokens = agg_completion_lengths.sum()

        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        return (
            prompt_ids,
            completion_ids,
            tool_mask,
            completions,
            total_completion_tokens,
            logprobs,
            extra_fields,
        )
    
    def _generate_and_score_completions(self, inputs: list[dict[str, torch.Tensor | Any]]) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        (
            prompt_ids_list,
            completion_ids_list,
            tool_mask_list,
            completions,
            num_items_in_batch,
            sampling_per_token_logps_list,
            extra_fields,
        ) = self._generate(prompts)

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")

        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.tensor(ids, device=device) for ids in extra_fields["rag_mask"]]
        completion_attn_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        completion_attn_mask = pad(completion_attn_mask, padding_value=0, padding_side="right")

        prm_scores = [torch.tensor(ids, device=device) for ids in extra_fields["prm_scores"]]
        prm_scores = pad(prm_scores, padding_value=0.0, padding_side="right")

        sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
        sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")

        completion_ids = self.accelerator.pad_across_processes(completion_ids, dim=1, pad_index=self.pad_token_id)
        completion_mask = self.accelerator.pad_across_processes(completion_mask, dim=1, pad_index=0)
        prm_scores = self.accelerator.pad_across_processes(prm_scores, dim=1, pad_index=0.0)
        sampling_per_token_logps = self.accelerator.pad_across_processes(sampling_per_token_logps, dim=1, pad_index=0.0)

        all_prm_scores = self.accelerator.gather(prm_scores)
        all_completion_mask = self.accelerator.gather(completion_mask)

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_attn_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size
                )
            else:
                old_per_token_logps = None

            if self.use_vllm and self.vllm_importance_sampling_correction:
                per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * completion_mask
                # vllm_importance_sampling_ratio = torch.exp(per_token_logps_diff)
                # vllm_importance_sampling_ratio = torch.clamp(vllm_importance_sampling_ratio, min=0.5, max=2.0)
                # vllm_importance_sampling_ratio = vllm_importance_sampling_ratio * completion_mask

                logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)
                vllm_importance_sampling_ratio = torch.exp(logps_diff)
                vllm_importance_sampling_ratio = vllm_importance_sampling_ratio.masked_fill(
                    vllm_importance_sampling_ratio > self.vllm_importance_sampling_cap, value=0.0
                )

            ref_per_token_logps = None

        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=False)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=False)

        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
        outcome_advantages = rewards - mean_grouped_rewards

        std_rewards = rewards.view(-1, num_generations).std(dim=1)
        std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        outcome_advantages = outcome_advantages / (std_rewards + 1e-4)

        if self.prm_beta > 0:
            prm_scores_reshaped = all_prm_scores.view(-1, num_generations, all_prm_scores.shape[-1])
            mask_reshaped = all_completion_mask.view(-1, num_generations, all_completion_mask.shape[-1])
            valid_mask = mask_reshaped.bool()

            # Group Normalization: 统计每个 Group 中所有 Generation 的所有 Valid Token
            group_counts = valid_mask.sum(dim=(1, 2), keepdim=True).float()
            group_sums = (prm_scores_reshaped * mask_reshaped).sum(dim=(1, 2), keepdim=True)
            group_means = group_sums / group_counts.clamp(min=1.0)

            squared_diffs = (prm_scores_reshaped - group_means).pow(2) * mask_reshaped
            group_vars = squared_diffs.sum(dim=(1, 2), keepdim=True) / group_counts.clamp(min=1.0)
            group_stds = (group_vars + 1e-8).sqrt()
            group_stds = torch.where(group_counts <= 1, torch.ones_like(group_stds), group_stds)

            normalized_prm = (prm_scores_reshaped - group_means) / group_stds
            
            step_adv = normalized_prm.view(all_prm_scores.shape) * all_completion_mask
            
            # [Core logic] Broadcast Outcome (scalar) to Vector and add Step Adv
            # outcome_advantages: (Global_Batch) -> unsqueeze -> (Global_Batch, 1)
            # step_adv: (Global_Batch, Max_Len)
            advantages = outcome_advantages.unsqueeze(1) + self.prm_beta * step_adv
        else:
            advantages = outcome_advantages.unsqueeze(1)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
            
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())

        valid_mask_float = all_completion_mask.float()
        sum_adv = (all_process_advantages * valid_mask_float).sum(dim=1)
        valid_len = valid_mask_float.sum(dim=1).clamp(min=1.0)
        avg_adv_for_log = sum_adv / valid_len
        self._logs["advantages"].extend(avg_adv_for_log.tolist())

        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            mask = completion_mask.bool()
            delta = delta[mask]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            flat_is_ratio = vllm_importance_sampling_ratio.flatten()
            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "attention_mask": attention_mask,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = vllm_importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        return output
    
    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = inputs["attention_mask"]
        logits_to_keep = completion_ids.size(1) 

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            mask = completion_mask if not self.tools else completion_mask * inputs["tool_mask"]
            entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the loss
        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            mask = completion_mask if not self.tools else completion_mask * inputs["tool_mask"]
            log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            # Importance sampling correction for the KL divergence
            if self.args.use_bias_correction_kl:
                per_token_kl = per_token_kl * coef_1

        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)
        if self.loss_type == "cispo":
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages * per_token_logps
        elif self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            # Two-sided clipping
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)

            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        elif self.loss_type == "sapo":
            per_token_loss = torch.empty_like(coef_1)
            positive_advantages_mask = advantages.repeat([1, coef_1.shape[1]]) > 0
            per_token_loss[positive_advantages_mask] = self.get_sapo_token_loss(
                coef_1[positive_advantages_mask], self.args.sapo_temperature_pos
            )
            per_token_loss[~positive_advantages_mask] = self.get_sapo_token_loss(
                coef_1[~positive_advantages_mask], self.args.sapo_temperature_neg
            )
            per_token_loss = -per_token_loss * advantages
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        mask = completion_mask if not self.tools else completion_mask * inputs["tool_mask"]
        if self.loss_type in ["grpo", "sapo"]:
            loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type in ["cispo", "dapo"]:
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        completion_token_count = mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            # Compute the clipped probability ratios
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())

            gathered_low_clip = self.accelerator.gather(low_clip)
            self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        elif self.loss_type == "cispo":
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            cispo_clip_ratio = masked_batch_mean(is_cispo_clipped.float())
            gathered_cispo_clip_ratio = self.accelerator.gather(cispo_clip_ratio)
            self._metrics[mode]["cispo_clip_ratio"].append(gathered_cispo_clip_ratio.nanmean().item())

        return loss