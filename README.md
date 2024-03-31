### LLM-Prompt Recovery Kaggle Competition


### https://www.kaggle.com/competitions/llm-prompt-recovery/overview


# Steps taking for this competition

1) use Phi-2 model and QLora train on one open source dataset. Tune hyper parameters
2) Clean the dataset first and see if any changes happen. Tune hyper parameters
2) Combine multiple open source datasets
3) Create own dataset
4) hyperparameter tuning along all steps

## Datasets

1) https://www.kaggle.com/datasets/mozhiwenmzw/llmpr-public-10k-unique

2) https://www.kaggle.com/datasets/dschettler8845/llm-prompt-recovery-synthetic-datastore

3) https://www.kaggle.com/datasets/dipamc77/3000-rewritten-texts-prompt-recovery-challenge

4) https://www.kaggle.com/datasets/thedrcat/llm-prompt-recovery-data

### command: 
```python train.py --save_path="phi2/exp_1_lr_2.5e-4 --lr=2.5e4```

# Phi-2 Experiments with public_10k_unique_rewrite_prompt dataset

### exp0 baseline lr=1e-4

| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 0     | 1.228200      | 1.134236        |
| 1     | 1.035700      | 1.000741        |
| 2     | 0.886600      | 0.953702        |
| 3     | 0.836000      | 0.937026        |
| 4     | 0.843800      | 0.934122        |

TrainOutput(global_step=1320, training_loss=1.059759164578987, metrics={'train_runtime': 7885.3166, 'train_samples_per_second': 2.679, 'train_steps_per_second': 0.167, 'total_flos': 1.309273021254144e+17, 'train_loss': 1.059759164578987, 'epoch': 5.0})


### exp1 baseline lr=2.5e-4
kserver running

### exp2 baseline lr=5e-4
kserver running

### exp3 baseline lr=1e-3
kserver running


### exp4 baseline lr=2.5e-4


### exp1 baseline lr=2.5e-4



