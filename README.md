# Fine-tuning for Vietnamese Sentiment Analysis

## 1. Purpose
This project aims to explore and evaluate various fine-tuning techniques for the task of **Vietnamese sentiment analysis**.

## 2. Objectives
* Gain a comprehensive understanding of fundamental fine-tuning techniques, including Full Model Fine-tuning and Parameter-Efficient Fine-Tuning (PEFT) methods such as Adapter, LoRA, and QLoRA.
* Learn to utilize multiple GPUs for fine-tuning large-scale models (e.g., ViT5).
* Achieve results suitable for submission to an international scientific conference.

## 3. Datasets
The following datasets are used in this project:
* [UIT-VSFC](https://nlp.uit.edu.vn/datasets#h.p_4Brw8L-cbfTe)
* [AIVIVN-2019](https://www.kaggle.com/datasets/mcocoz/aivivn-2019/code)

## 4. Large Language Models
The project utilizes the following Vietnamese pre-trained language models:
* [PhoBERT-base-v2](https://huggingface.co/vinai/phobert-base-v2)
* [PhoBERT-large](https://huggingface.co/vinai/phobert-large)
* [BARTpho-word](https://huggingface.co/vinai/bartpho-word)
* [ViT5-large](https://huggingface.co/VietAI/vit5-large)

## 5. Fine-tuning Techniques
1. Full Model Fine-tuning
2. LoRA (Low-Rank Adaptation)
3. QLoRA
4. Adapter

## 6. Install dependencies
The following libraries need to be installed:
* [Pytorch](https://pytorch.org/get-started/locally/)
* [Jupyter Notebook](https://jupyter.org/install)

```
pip install pandas
pip install transformers datasets adapters peft bitsandbytes
pip install lightning torchmetrics
pip install underthesea
pip install gputil
pip install matplotlib seaborn
```

## 7. Experiment
All scripts include an argument parser, so you can run any Python script with the ```--help``` parameter to see usage instructions. See the example below.

```
python vsfc_full_fine_tune.py --help
```