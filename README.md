## Usage


### Environment
```bash
conda create -n rag python=3.10
conda activate rag
pip install -r requirements.txt
```

Changed your local data path in download.py and run it:
```bash
python download.py
```

tmux new -s rag

## Log
### 7.26 Start
    Search for the dataset: MulSiQue(phase 1 train), frames, 2wikimultihopQA, HotpotQA
### 7.27 Prepare data
    preprocess for problems in MulSiQue (non-question)
    prepare for training data: generate 2048 items(findings: 32 batch could be faster)
### 7.28 Prepare data and SFT
    mistake: carelessly delete 2048 items generated yesterday
    regenerate 5245 items (update to huggingface) (not all)
    sft qwen3-8b on my data (LoRA)