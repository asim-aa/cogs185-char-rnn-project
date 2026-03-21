# char-rnn.pytorch

A PyTorch implementation of [char-rnn](https://github.com/karpathy/char-rnn) for character-level text generation, based on the [Practical PyTorch series](https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb).

This repository was extended for a COGS 185 final project at UC San Diego, conducting a systematic empirical study of GRU language model architecture, optimizer choice, and cross-domain transfer across two literary datasets.

---

## Project Overview

**Research question:** How do hidden size, network depth, optimizer, and sampling temperature affect character-level language modeling quality?

**Datasets:** Tiny Shakespeare, Sherlock Holmes (Project Gutenberg)

**Key findings:**
- Increasing hidden size improves perplexity but with diminishing returns (4.80 → 4.19 → 4.02)
- Combining depth + width yields the best model (PPL 4.01)
- SGD fails catastrophically (PPL 26.45) vs Adam (PPL 4.01)
- The model learns domain-specific structure (verse vs. prose) without explicit supervision
- Temperature τ=1.0 gives the best balance of coherence and diversity

---

## Requirements
```
pip install -r requirements.txt
```

Requires Python 3.12+ and PyTorch 2.x.

---

## Training

### Basic usage
```
python train.py shakespeare.txt
```

### Reproduce all experiments

**Baseline** (1 layer, hidden 128, Adam):
```
python train.py shakespeare.txt --n_epochs 300 --hidden_size 128 --n_layers 1 --learning_rate 0.005 --batch_size 100 --chunk_len 200 --save_dir experiments/baseline
```

**Bigger** (1 layer, hidden 256, Adam):
```
python train.py shakespeare.txt --n_epochs 300 --hidden_size 256 --n_layers 1 --learning_rate 0.005 --batch_size 100 --chunk_len 200 --save_dir experiments/bigger
```

**Biggest** (1 layer, hidden 512, Adam):
```
python train.py shakespeare.txt --n_epochs 300 --hidden_size 512 --n_layers 1 --learning_rate 0.005 --batch_size 100 --chunk_len 200 --save_dir experiments/biggest
```

**Deeper** (2 layers, hidden 128, Adam):
```
python train.py shakespeare.txt --n_epochs 300 --hidden_size 128 --n_layers 2 --learning_rate 0.005 --batch_size 100 --chunk_len 200 --save_dir experiments/deeper
```

**deeper_512** (2 layers, hidden 512, Adam) — best model:
```
python train.py shakespeare.txt --n_epochs 300 --hidden_size 512 --n_layers 2 --learning_rate 0.005 --batch_size 100 --chunk_len 200 --save_dir experiments/deeper_512
```

**sgd_512** (2 layers, hidden 512, SGD):
```
python train.py shakespeare.txt --n_epochs 300 --hidden_size 512 --n_layers 2 --learning_rate 0.005 --batch_size 100 --chunk_len 200 --optimizer sgd --save_dir experiments/sgd_512
```

**sherlock_512** (2 layers, hidden 512, Adam — Sherlock Holmes):
```
python train.py sherlock.txt --n_epochs 300 --hidden_size 512 --n_layers 2 --learning_rate 0.005 --batch_size 100 --chunk_len 200 --save_dir experiments/sherlock_512
```

---

## Generation

### Basic usage
```
python generate.py experiments/deeper_512/model.pt -p "The" -l 300 -t 0.8
```

### Temperature comparison (best Shakespeare model)
```
python generate.py experiments/deeper_512/model.pt -p "The" -l 300 -t 0.5
python generate.py experiments/deeper_512/model.pt -p "The" -l 300 -t 1.0
python generate.py experiments/deeper_512/model.pt -p "The" -l 300 -t 1.3
```

### Temperature comparison (Sherlock Holmes model)
```
python generate.py experiments/sherlock_512/model.pt -p "Holmes" -l 300 -t 0.5
python generate.py experiments/sherlock_512/model.pt -p "Holmes" -l 300 -t 1.0
python generate.py experiments/sherlock_512/model.pt -p "Holmes" -l 300 -t 1.3
```

---

## Compare All Experiments

After training, run the comparison script to print a perplexity summary table and generate an overlay loss curve:
```
python compare_experiments.py
```

This loads `loss_history.json` from each experiment folder and outputs `comparison_loss_curve.png`.

---

## Results Summary

| Model | Hidden | Layers | Optimizer | Loss | PPL |
|---|---|---|---|---|---|
| baseline | 128 | 1 | Adam | 1.5695 | 4.80 |
| bigger | 256 | 1 | Adam | 1.4322 | 4.19 |
| biggest | 512 | 1 | Adam | 1.3902 | 4.02 |
| deeper | 128 | 2 | Adam | 1.4869 | 4.42 |
| **deeper_512** | **512** | **2** | **Adam** | **1.3890** | **4.01** |
| sgd_512 | 512 | 2 | SGD | 3.2752 | 26.45 |
| sherlock_512 | 512 | 2 | Adam | 1.2811 | 3.60 |

---

## Project Structure
```
char-rnn.pytorch/
├── experiments/
│   ├── baseline/
│   ├── bigger/
│   ├── biggest/
│   ├── deeper/
│   ├── deeper_512/
│   ├── sgd_512/
│   └── sherlock_512/
├── samples/
├── train.py
├── generate.py
├── model.py
├── helpers.py
├── compare_experiments.py
├── shakespeare.txt
├── sherlock.txt
├── requirements.txt
└── README.md
```

---

## Training Options
```
Usage: train.py [filename] [options]

Options:
--model            Whether to use LSTM or GRU units    gru
--n_epochs         Number of epochs to train           2000
--print_every      Log learning rate at this interval  100
--hidden_size      Hidden size of GRU                  50
--n_layers         Number of GRU layers                2
--learning_rate    Learning rate                       0.01
--chunk_len        Length of training chunks           200
--batch_size       Number of examples per batch        100
--cuda             Use CUDA
```

## Generation Options
```
Usage: generate.py [filename] [options]

Options:
-p, --prime_str      String to prime generation with
-l, --predict_len    Length of prediction
-t, --temperature    Temperature (higher is more chaotic)
--cuda               Use CUDA
```

---

## Citation

Based on the char-rnn.pytorch implementation by [@spro](https://github.com/spro/char-rnn.pytorch).
Original char-rnn by [Andrej Karpathy](https://github.com/karpathy/char-rnn).