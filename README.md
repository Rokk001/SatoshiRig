## BtcSoloMinerGpu

Clean refactor with professional layout, standard naming, and Docker support.

### Structure

```
src/
  BtcSoloMinerGpu/
    __init__.py
    __main__.py
    cli.py
    context.py
    miner.py
tests/
requirements.txt
Dockerfile
.dockerignore
.gitignore
```

### Run locally

1) Install dependencies:
```
pip install -r requirements.txt
```

2) Run with wallet address (optional: custom config path):
```
python -m BtcSoloMinerGpu --wallet YOUR_BTC_ADDRESS --config ./config/config.toml --backend cpu --gpu 0
```

or via environment variable:
```
set WALLET_ADDRESS=YOUR_BTC_ADDRESS  # Windows
set CONFIG_FILE=./config/config.toml
set COMPUTE_BACKEND=cpu
set GPU_DEVICE=0
python -m BtcSoloMinerGpu
```

### Docker

Build image:
```
docker build -t btcsominer-gpu .
```

Run (pass wallet via env):
```
docker run --rm --gpus all -e WALLET_ADDRESS=YOUR_BTC_ADDRESS -e COMPUTE_BACKEND=cuda -e GPU_DEVICE=0 btcsominer-gpu
```

Config file is TOML at `config/config.toml`. Override with `--config` or `CONFIG_FILE`.

Press Ctrl+C to stop gracefully.
