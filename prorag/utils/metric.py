import collections

def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = prediction.lower().split()
    
    gt_tokens = ground_truth.lower().split()
    if not prediction_tokens or not gt_tokens:
        return 0.0
    common = collections.Counter(prediction_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(prediction_tokens)
    rec = num_same / len(gt_tokens)
    f1 = 2 * prec * rec / (prec + rec)

    return f1