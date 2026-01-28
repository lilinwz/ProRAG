## Installation

### Prerequisities
- Python 3.13+
- CUDA 12.x (Recommended)

### ProRAG environment

```bash
# 1. Create and activate conda environment
conda create -n prorag python=3.13.11
conda activate prorag

# 2. Install vLLM (This will install PyTorch 2.8.0 automatically)
pip install vllm==0.11.0

# 3. Install ProRAG (Current package)
pip install -e .

# 4. Install training dependencies (TRL, PEFT, WandB, DeepSpeed)
pip install deepspeed==0.18.4 trl==0.26.0 peft==0.18.1 wandb==0.24.0 

# 5. Install Flash Attention 2
pip install flash-attn==2.8.3 --no-build-isolation
```

### Retriever environment (optional)

If you would like to call a local retriever as the search engine, you can install the environment as follows. (We recommend using a seperate environment.)

```bash
conda create -n retriever python=3.10
conda activate retriever

# we recommend installing torch with conda for faiss-gpu
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini

## install the gpu version faiss to guarantee efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

## API function
pip install uvicorn fastapi
```
