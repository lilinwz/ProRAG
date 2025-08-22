import json
import re
import numpy as np
import collections
from typing import List, Dict, Any, Tuple

# --- 辅助函数 (与之前版本相同) ---
def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = prediction.lower().split()
    ground_truth_tokens = ground_truth.lower().split()
    if not prediction_tokens or not ground_truth_tokens: return 0.0
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_full_state_from_path(path: List[Dict]) -> str:
    state_parts = [node.get('action') for node in path if node.get('action')]
    return "".join(state_parts)

def find_all_terminal_paths(node: Dict, current_path: List[Dict], all_paths: List[List[Dict]]):
    current_path.append(node)
    if not node.get('children'):
        all_paths.append(list(current_path))
    else:
        for child in node['children']:
            find_all_terminal_paths(child, current_path, all_paths)
    current_path.pop()

def analyze_diversity_recursively(node: Dict) -> Tuple[float, float, int]:
    if not node.get('children') or len(node.get('children', [])) < 2:
        return 0.0, 0.0, 0
    total_q_n_variance, total_n_entropy, nodes_with_siblings_count = 0.0, 0.0, 1
    children = node['children']
    sibling_q_n = [child['q'] / child['n'] if child['n'] > 0 else 0 for child in children]
    sibling_n = [child['n'] for child in children]
    total_q_n_variance += np.var(sibling_q_n)
    total_visits = sum(sibling_n)
    if total_visits > 0:
        probs = np.array([n / total_visits for n in sibling_n if n > 0])
        total_n_entropy += -np.sum(probs * np.log2(probs))
    for child in children:
        var, entropy, count = analyze_diversity_recursively(child)
        total_q_n_variance += var
        total_n_entropy += entropy
        nodes_with_siblings_count += count
    return total_q_n_variance, total_n_entropy, nodes_with_siblings_count

def analyze_tree_quality(tree: Dict, ground_truth_answers: List[str]) -> Dict[str, Any]:
    if not tree or not tree.get('children'):
        return {"error": "Tree is empty or has no children."}
    all_paths = []
    find_all_terminal_paths(tree, [], all_paths)
    best_path, max_f1 = None, -1.0
    for path in all_paths:
        full_state = get_full_state_from_path(path)
        answer_match = re.search(r"<answer>(.*?)</answer>", full_state, re.DOTALL)
        if answer_match:
            extracted_answer = answer_match.group(1).strip()
            scores = [calculate_f1_score(extracted_answer, ans) for ans in ground_truth_answers if ans]
            current_f1 = max(scores) if scores else 0.0
            if current_f1 > max_f1:
                max_f1 = current_f1
                best_path = path
    metrics = {'max_f1': max_f1}
    total_var, total_ent, count = analyze_diversity_recursively(tree)
    metrics['avg_sibling_q_n_variance'] = total_var / count if count > 0 else 0.0
    return metrics

# --- 新增的核心筛选逻辑 ---

def apply_filters(metrics: Dict, tree_root: Dict, thresholds: Dict) -> Tuple[bool, str]:
    """
    根据预设的阈值对树的分析指标进行筛选。
    返回一个元组 (是否通过, 失败原因)。
    """
    # 规则 1: 最终答案质量必须达标
    if metrics['max_f1'] < thresholds['MIN_F1_SCORE']:
        return False, f"F1分数过低 ({metrics['max_f1']:.2f})"

    # 规则 2: 必须有最基本的探索，不能是一条路走到黑
    if len(tree_root.get('children', [])) < thresholds['MIN_ROOT_CHILDREN']:
        return False, "根节点探索不足 (子节点 < 2)"

    # 规则 3: 树必须能有效区分不同选择的优劣
    if metrics['avg_sibling_q_n_variance'] < thresholds['MIN_AVG_Q_N_VARIANCE']:
        return False, f"Q/N方差过低 ({metrics['avg_sibling_q_n_variance']:.4f})"

    # 如果所有检查都通过
    return True, "通过筛选"

def main(input_path: str, output_path: str):
    """主函数，加载、分析、筛选并保存结果。"""
    
    # --- 在这里配置你的筛选标准 ---
    FILTER_THRESHOLDS = {
        # 规则1: 最高F1分数必须高于此值，否则认为搜索失败。
        "MIN_F1_SCORE": 0.1,
        
        # 规则2: 根节点的孩子数量必须大于等于此值，以确保进行了基本的探索。
        "MIN_ROOT_CHILDREN": 2,
        
        # 规则3: 兄弟节点Q/N值的平均方差必须高于此值，确保树具有分辨能力。
        # 一个非常低的值意味着所有选项看起来都差不多，模型没学到什么。
        "MIN_AVG_Q_N_VARIANCE": 0.01
    }

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_path}")
        return
    except json.JSONDecodeError:
        print(f"错误：文件 {input_path} 不是有效的 JSON 格式。")
        return

    passed_items = []
    failure_reasons = []

    for item in data:
        if 'mcts_tree' in item and item['mcts_tree']:
            metrics = analyze_tree_quality(item['mcts_tree'], item.get('answer', []))
            
            if "error" in metrics:
                failure_reasons.append("树结构错误或为空")
                continue

            is_passed, reason = apply_filters(metrics, item['mcts_tree'], FILTER_THRESHOLDS)
            
            if is_passed:
                passed_items.append(item)
            else:
                failure_reasons.append(reason)

    # --- 打印总结报告 ---
    total_count = len(data)
    passed_count = len(passed_items)
    failed_count = total_count - passed_count
    
    print("=" * 80)
    print("MCTS 树筛选报告")
    print("=" * 80)
    print(f"总共处理树的数量: {total_count}")
    print(f"通过筛选的数量:   {passed_count} ({passed_count/total_count:.2%})")
    print(f"被淘汰的数量:     {failed_count} ({failed_count/total_count:.2%})")
    print("-" * 80)
    
    if failed_count > 0:
        print("淘汰原因分析:")
        reason_counts = collections.Counter(failure_reasons)
        for reason, count in reason_counts.most_common():
            print(f"  - {reason:<30}: {count:>5} 次")
    print("=" * 80)

    # --- 保存通过筛选的数据 ---
    if passed_items:
        print(f"\n正在将 {passed_count} 个通过筛选的样本保存到: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(passed_items, f, ensure_ascii=False, indent=4)
        print("保存完成。")
    else:
        print("\n没有样本通过筛选，不生成输出文件。")


if __name__ == "__main__":
    input_file = '/home/v-zhaowan/zhaowang/rag/sample/data_mcts_500.json'
    output_file = '/home/v-zhaowan/zhaowang/rag/sample/data_mcts_500_filtered.json'
    main(input_file, output_file)