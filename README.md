# Mixture-of-depths replication

This repo contains code to replicate the paper [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258).


# Setup

1. Run `sh setup_miniconda.sh` to setup `conda`
2. `conda activate mod`
3. `databricks configure --token` you need to configure your DBX URI and PAT in order to track model training with `mlflow`


# Flops for GPT2 models
- 32B stands for batch size of 32
- FP stands for forward pass

|Metric|gpt2|gpt2-medium|gpt2-large|gpt2-xl|
|---|---|---|---|---|
|Total FLOPs per FP|6.2e+08|1.8e+09|3.9e+09|7.8e+09|
|# of Transformer Blocks|12|24|36|48|
|FLOPs for Standard Block|3.5e+07|6.3e+07|9.8e+07|1.5e+08|
|FLOPs for MOT Block|4.4e+06|7.9e+06|1.2e+07|1.9e+07|
|FLOPs for All Non-Transformer Layers|1.9e+08|2.6e+08|3.2e+08|4.0e+08|
|# of 32B FP w/o MOT|51,729,631,974|18,081,665,480|8,278,940,692|4,111,749,117|
|# of 32B FP w/ MOT|74,005,002,738|28,879,031,512|13,823,343,203|7,027,382,194|