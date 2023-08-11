# Self-Supervised Learning for Human Activity Recognition using Transformer

## Requirements
If you would like to develop the model for your own use, you need to follow the instructions below.
### Installation
```bash
conda create -n ssl_env python=3.7
conda activate ssl_env
pip install -r req.txt
```
You may get an error regarding cuda, please install cuda with [pytorch](https://pytorch.org/get-started/locally/).

## Training
Pre-training can be performed with the following command. the data path to `/home/SHL/ssl-data/...` is set in `conf/config.py`, so you should be able to run it. If you want to change the configuration, please change `conf/config.py` as needed.
### pre-train learning
```bash
python pre-train.py
```

## downstream
Please download the data first. 
```bash
bash wget_oppodata.sh
```
then, opportunity data for downstream will be downloaded. Downstreaming can be performed with the following code.
```bash
python3 downstream.py
```
Please change the configuration in `conf/config_eva.py` as needed.
### Results
after the above code is executed, the result is output to `experiment_log/downstream/`.

## References
[Multi-task self-supervised learning for wearables](https://github.com/OxWearables/ssl-wearables)
