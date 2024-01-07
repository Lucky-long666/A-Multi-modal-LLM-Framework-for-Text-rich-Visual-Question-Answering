# Using a Simple Multimodal LLM to Enhance the Processing of Text-heavy Visual Questions.



## Installation

1. Creating conda environment

```bash
conda create -n bliva python=3.9
conda activate bliva
```

2. build from source

```bash
git clone https://github.com/mlpc-ucsd/BLIVA
cd BLIVA
pip install -e .
```

## Prepare Weight

1. BLIVA Vicuna 7B

    Our Vicuna version model is released at [here](https://huggingface.co/mlpc-lab/BLIVA_Vicuna). Download our model weight and specify the path in the model config [here](bliva/configs/models/bliva_vicuna7b.yaml#L8) at line 8. 

    The LLM we used is the v0.1 version from Vicuna-7B. To prepare Vicuna's weight, please refer to our instruction [here](PrepareVicuna.md). Then, set the path to the vicuna weight in the model config file [here](bliva/configs/models/bliva_vicuna7b.yaml#L21) at Line 21.

2. BLIVA FlanT5 XXL (Available for Commercial Use)

    The FlanT5 version model is released at [here](https://huggingface.co/mlpc-lab/BLIVA_FlanT5). Download our model weight and specify the path in the model config [here](bliva/configs/models/bliva_flant5xxl.yaml#L8) at line 8. 

    The LLM weight for Flant5 will automatically begin to download from huggingface when running our inference code. 

## Inference 

To answer one question from the image, run the following evaluation code. For example,

```Shell
python evaluate.py --answer_qs \
        --model_name bliva_vicuna \
        --img_path images/example.jpg \
        --question "what is this image about?"
```

We also support answer multiple choice question, which is the same as we used for evaluation tasks in paper. To provide a list of chioce, it should be a string split by comma. For example,

```Shell
python evaluate.py --answer_mc \
        --model_name bliva_vicuna \
        --img_path images/mi6.png \
        --question "Which genre does this image belong to?" \
        --candidates "play, tv show, movie"
```
## Demo

To run our demo locally on your machine. Run:

```Shell
python demo.py
```

## Train

After downloading the training datasets and specify their path in [dataset configs](bliva/configs/datasets/), we are ready for training. We utilized 8x A6000 Ada in our experiments. Please adjust hyperparamters according to your GPU resources. It may take transformers around 2 minutes to load the model, give some time for the model to start training. Here we give an example of traning BLIVA Vicuna version, the Flant5 version follows the same format.

1. Pretraining of BLIVA's visual assistant branch 

```Shell
torchrun --nnodes=1 --nproc_per_node=8 \
    train.py \
    --cfg-path train_configs/pretrain_bliva_vicuna.yaml
```

2. Instruction Finetuning BLIVA

```Shell
torchrun --nnodes=1 --nproc_per_node=8 \
    train.py \
    --cfg-path train_configs/finetune_bliva_vicuna.yaml
```

Or, we also support training Vicuna7b together with BLIVA using LoRA during the second step, by default we don't use this version. 

```Shell
torchrun --nnodes=1 --nproc_per_node=8 \
    train.py \
    --cfg-path train_configs/finetune_bliva_and_vicuna.yaml
```


## 

## License
This repository's code is under [BSD 3-Clause License](LICENSE.md).
Many codes are based on [Lavis](https://github.com/salesforce/LAVIS) with 
BSD 3-Clause License [here](LICENSE_LAVIS.md).

For our model parameters of BLIVA Vicuna Version, it's should be used under LLaMA's [model license](https://github.com/facebookresearch/llama/blob/llama_v1/LICENSE). 
For the model weight of BLIVA FlanT5, it's under [Apache 2.0 License](LICENSE_BLIVA_FLANT5_WEIGHT.md). 
For our YTTB-VQA data, it's under [CC BY NC 4.0](LICENSE_DATA.md)

[![Code License](https://img.shields.io/badge/Code%20License-BSD_3--Clause-blue.svg)](LICENSE.md)
[![BLIVA Vicuna Weight License](https://img.shields.io/badge/BLIVA%20Vicuna%20Weight%20License-Non_commercial_bespoke_license-orange.svg)](https://github.com/facebookresearch/llama/blob/llama_v1/LICENSE)
[![BLIVA FLANT5 Weight License](https://img.shields.io/badge/BLIVA%20FLANT5%20Weight%20License-Apache_2.0-green.svg)](LICENSE_BLIVA_FLANT5_WEIGHT.md)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](LICENSE_DATA.md)

