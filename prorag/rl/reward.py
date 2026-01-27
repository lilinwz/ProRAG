import re
from prorag.utils.metric import calculate_f1_score

TAG_MAP = {
    "<step>":      ("</step>",      "S"),
    "<subquery>":  ("</subquery>",  "Q"),
    "<retrieval>": ("</retrieval>", "R"),
    "<subanswer>": ("</subanswer>", "A"),
    "<answer>":    ("</answer>",    "F")
}
CYCLE_PATTERN = ["S", "Q", "R", "S", "A"]
END_PATTERN = ["S", "F"]

def format_reward(completion):
    completion = "<step>\n" + completion
    tags = re.findall(r"</?[a-zA-Z]+>", completion)
    tags = [t for t in tags if t != "<|endodtext|>" and t != "<|im_end|>"]

    symbols = []
    is_valid_pairing = True
    i = 0
    while i < len(tags):
        open_tag = tags[i]
        if open_tag not in TAG_MAP:
            is_valid_pairing = False
            break
            
        expected_close, symbol = TAG_MAP[open_tag]
        if i + 1 >= len(tags) or tags[i+1] != expected_close:
            is_valid_pairing = False
            break

        symbols.append(symbol)
        i += 2

    if not is_valid_pairing:
        return -1.0
        
    if len(symbols) < len(END_PATTERN):
        return -1.0
        
    if symbols[-len(END_PATTERN):] != END_PATTERN:
        return -1.0
        
    remaining = symbols[:-len(END_PATTERN)]

    if len(remaining) == 0:
        return 1.0
        
    if len(remaining) % len(CYCLE_PATTERN) != 0:
        return -1.0
        
    is_structure_valid = True
    for k in range(0, len(remaining), len(CYCLE_PATTERN)):
        chunk = remaining[k : k + len(CYCLE_PATTERN)]
        if chunk != CYCLE_PATTERN:
            is_structure_valid = False
            break

    if is_structure_valid:
        return 1.0
    return -1.0

def outcome_reward(completions: list[str], answer: list[str], **kwargs) -> list[float]:
    rewards = []
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
    for generated_content, ground_truth in zip(completions, answer):
        match = pattern.search(generated_content)
        acc = 0.0
        if match:
            pred = match.group(1).strip()
            acc = calculate_f1_score(pred, ground_truth)
            if acc < 0.2: acc = 0.0
            else: acc = acc * 2

        fmt = format_reward(generated_content)
        rewards.append(acc+fmt)
            
    return rewards

def format_reward_step(completion):
    completion = "<step>\n" + completion
    tags = re.findall(r"</?[a-zA-Z]+>", completion)
    tags = [t for t in tags if t != "<|endodtext|>" and t != "<|im_end|>"]
    if tags == ["</step>", "<subquery>", "</subquery>", "<retrieval>", "</retrieval>"]:
        return 1.0
    if tags == ["</step>", "<subanswer>", "</subanswer>"]:
        return 1.0
    if tags == ["</step>", "<answer>", "</answer>"]:
        return 1.0
    return -2.0