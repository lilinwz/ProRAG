## Environment
```bash
conda create -n rag python=3.10
conda activate rag
pip install -r requirements.txt
```

Changed your local data path in download.py and run it:
```bash
python download.py
```

## Data
SFT: MulSiQue answerable train dataset (about 19k) -> (Sampling) 7.8k

MC: MulSiQue answerable dev dataset (about 2.5k) -> (Sampling) 1.5k

PPO: 

## Experiment setup

### baseline

### benchmark
Hotpot、2WikiMultiHopQA、Frames

MulSiQue(Main): using paragraph in the data as retrieval

### Ablation
sft model, DPO model, PPO model(, GRPO model)

## Log
### 7.27 Start
    Search for the dataset: MulSiQue(phase 1 train), frames, 2wikimultihopQA, HotpotQA
### 7.28 Prepare data
    preprocess for problems in MulSiQue (non-question)
    prepare for training data: generate 2048 items(findings: 32 batch could be faster)
### 7.29 Prepare data and SFT
    mistake: carelessly delete 2048 items generated yesterday
    regenerate 5245 items (update to huggingface) (not all)
    sft qwen3-8b on my data (LoRA without valid set and test set)
    possible problem 1: all train data is 2-hop, which means no single-hop and multi(3+)-hop
        solved: all dataset contains multi-hop questions, and we focus on multihop rather than single-hop (more general)
    possible problem 2: some subanswer could not be retrievaled. (how to say idk)
        1. wait for training result
        2. add some extra training data
        but actually we deal with it in phase 2, so it may don't matter.
### 7.30 Prepare data, SFT and brainstorm
    mistake: the num of dataset (rerun or continue) should be 19k rather than 9k (7k after processing)
    sft qwen3-8b on old data (7k): may be enough
    brainstorm about the Mento Carlo sample on the dev set of MulSiQue
    generate dev data set (about 1.5k)
### 7.31 Prepare data, SFT and test
    prepare train_full.json (13k)
    changed sft.py: add special token
    write test/mulsique.py and sample/gen.py
### 8.1 Prepare data, SFT
    prepare train_full.json (14k) and divide it into train_sft.json (12k) and train_rl.json (2k)
    problem: missing special token in output
        rewrite sft.py and trainer: weight special token when calculating the loss
    thought: PPO / GRPO / SFT, which one is better?
        PPO / GRPO: format reward
        SFT: rewight special token
### 8.1 SFT
    fixed sft.py and search the best hyperparameters
    running (2k) to learn workflow
### 8.2 SFT
    finding: it's difficult for model to learn workflow only by special tokens
    try: loss_full + 100 * loss_special 
        why not 0.01 * loss_full + loss_special? 
        100 means constraint but 0.01 means learn new tasks (may lead to vanishing gradient)
### 8.4 SFT
    fixed cot_post.py and trainer.py
    try: weight = 10
### 8.5 SFT
    problem: model only predict <think> and </think>
        reweight every special token: 2, 10, 20 (not work)
    solved: unlock emb & head layer for special token
### 8.6 SFT
    finish 8-phase Curriculum Learning 
    Token: <step>       | Accuracy: 100.00% | Count: 538
    Token: </step>      | Accuracy:  98.88% | Count: 538
    Token: <subquery>   | Accuracy: 100.00% | Count: 219
    Token: </subquery>  | Accuracy: 100.00% | Count: 219
    Token: <retrieval>  | Accuracy: 100.00% | Count: 219
    Token: </retrieval> | Accuracy:  98.17% | Count: 219
    Token: <subanswer>  | Accuracy: 100.00% | Count: 219
    Token: </subanswer> | Accuracy: 100.00% | Count: 219
    Token: <answer>     | Accuracy: 100.00% | Count: 100
    Token: </answer>    | Accuracy: 100.00% | Count: 100
### 8.7 SFT
    finish final SFT
### 8.12 test
    test the sft model
### 8.14 reSFT
    reSFT
### 8.15 reTest and BrainStorm
    retest the sft model
    brainstorm: LLM-enhanced / MCTS-based and DPO / GRPO
### 8.16 discussion
    discuss with ziliang
### 8.17 retrieval
    rewrite retrieval method: need too much time
### 8.18 sample
    write mcts.py and debug
### 8.19 sample
    begin hg-mcts
    rewrite retrieval (e5) in test
    consider baseline (R1-Searcher++, search-r1, ReasonRAG, search-o1)
### 8.20-21 sample
    continue hg-mcts
    write rm.py and grpo.py
### 8.22 sample
    finish hg-mcts (1.5k in total)
    start training rm
### 8.23-25 training
    continue training rm
### 8.26-29 training
    rewrite ppo (final use trl)