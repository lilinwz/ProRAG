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
    prepare train_full.json
    changed sft.py: add special token
    write test/mulsique.py and sample/gen.py
    