# Negoformer: A Novel Negotiation Strategy based on Transformers for Long-term Opponent Behavior Prediction
## IJCAI - 2024

## Table of Contents
- [Install](#install)
  - [For CUDA](#for-cuda)
- [Usage](#usage)
- [License](#license)

## Install
> **_NOTE:_** This project Python project is tested in [Python 3.8](https://www.python.org/downloads/release/python-387/), [CUDA 11.3](https://developer.nvidia.com/cuda-11.3.0-download-archive), [cuDNN 8](https://developer.nvidia.com/rdp/cudnn-archive), Windows 10 and Ubuntu 18.04. 
> Creating a [virtual environment](https://docs.python.org/3.8/library/venv.html) is recommended. 

You can install this Python project from the scratch, by following steps:

1. Download whole project from [GitHub](https://github.com/negoformer/NegoFormer-IJCAI-2024).
2. Install [Python 3](https://www.python.org/downloads/release/python-387/)
3. Download & install required Python libraries via `pip`, as shown below:
    ```bash
    pip install -r requirements.txt 
    ```
4. You can run a tournament as described in [Usage](#usage)

### For CUDA
You can utilize your GPU to run Autoformer model in Negoformer, by following steps:
1. Download & install [NVIDIA GPU Drivers](https://www.nvidia.com/download/index.aspx)
2. Download & install [CUDA](https://developer.nvidia.com/cuda-11.3.0-download-archive)
3. Download & install [cuDNN 8](https://developer.nvidia.com/rdp/cudnn-archive)
4. Download & install proper [torch](https://pytorch.org/get-started/previous-versions/) version. For example:
    ```bash
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
    ```
## Usage
You can start a negotiation tournament in `run.py` Python script by providing a tournament configuration. Tournament configuration is a [YAML](https://yaml.org/). For example:
```bash
python run.py tournament_negoformer.yaml
```

> **_NOTE:_** You can create or edit your own [YAML](https://yaml.org/) file to customize a tournament.
>

Also, you can run pre-defined tournament configurations, as in the paper:
- Negoformer Tournament:
    ```bash
    python run.py tournament_negoformer.yaml
    ```
- ParetoWalker Tournament:
    ```bash
    python run.py tournament_paretowalker.yaml
    ```

- Negoformer & ParetoWalker Tournament:
    ```bash
    python run.py tournament_mix.yaml
    ```

- Data Collection Tournament:
    ```bash
    python run.py tournament_data_collection.yaml
    ```

## License
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)