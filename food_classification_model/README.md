---
license: apache-2.0
base_model: google/vit-base-patch16-224-in21k
tags:
- generated_from_trainer
datasets:
- ethz/food101
metrics:
- accuracy
model-index:
- name: google/vit-base-patch16-224-in21k-v2-finetuned
  results:
  - task:
      name: Image Classification
      type: image-classification
    dataset:
      name: food101
      type: ethz/food101
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.7968976897689769
language:
- en
pipeline_tag: image-classification
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# google/vit-base-patch16-224-in21k-v2-finetuned

This model is a fine-tuned version of [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) on the food101 dataset.
It achieves the following results on the evaluation set:
- Loss: 1.0612
- Accuracy: 0.7969

## Model description
- Model type: Language model
- Language(s) (NLP): English
- License: Apache 2.0
- Related Model: google/vit-base-patch16-224-in21k
- Original Checkpoints: google/vit-base-patch16-224-in21k
- Resources for more information:
[Research paper](https://arxiv.org/pdf/2210.11416.pdf)


## Intended uses & limitations

This model can be used to classify what type of food in the image provided.

## Training and evaluation data

The model was trained on food101 dataset with 80:20 train-test-split.

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 64
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 1.9201        | 1.0   | 947  | 1.9632          | 0.7297   |
| 1.2002        | 2.0   | 1894 | 1.2327          | 0.7805   |
| 0.9561        | 3.0   | 2841 | 1.0612          | 0.7969   |


### Framework versions

- Transformers 4.40.2
- Pytorch 2.4.0+cu121
- Datasets 2.21.0
- Tokenizers 0.19.1