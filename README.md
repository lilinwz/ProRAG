## Installation

### Prerequisities
- Python 3.13+
- CUDA 12.x (Recommended)

### Quick Start

```bash
# 1. Create and activate conda environment
conda create -n prorag python=3.13.11
conda activate prorag

# 2. Install vLLM (This will install PyTorch 2.8.0 automatically)
pip install vllm==0.11.0

# 3. Install ProRAG (Current package)
pip install -e .

# 4. Install training dependencies (TRL, PEFT, WandB, DeepSpeed)
pip install deepspeed==0.18.4 trl==0.27.0 peft==0.18.1 wandb==0.24.0 

# 5. Install Flash Attention 2
pip install flash-attn==2.8.3 --no-build-isolation
```
