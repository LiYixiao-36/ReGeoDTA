# ReGeoDTA

This repository contains the main ReGeoDTA training and evaluation code, together with a separate module for generating the pretrained input data required by the model.

## 1. Main Code

The main code is used to train and evaluate ReGeoDTA on the Davis, KIBA, and BindingDB datasets.

### Main Files

- `main.py`: command-line entry point for training and evaluation.
- `model.py`: ReGeoDTA model definition.
- `engine.py`: training and evaluation workflow.
- `data.py`: dataset registry.
- `datasets_davis.py`: Davis dataset loader.
- `datasets_kiba.py`: KIBA dataset loader.
- `datasets_bindingdb.py`: BindingDB dataset loader.
- `datasets_common.py`: shared dataset utilities.
- `metrics.py`: evaluation metrics.
- `utils.py`: runtime utilities.

### Train

```bash
python main.py train \
  --dataset <davis|kiba|bindingdb> \
  --data-root <dataset-path> \
  --pretrained-root <pretrained-data-path>
```

### Evaluate

```bash
python main.py evaluate \
  --dataset <davis|kiba|bindingdb> \
  --data-root <dataset-path> \
  --pretrained-root <pretrained-data-path> \
  --checkpoint <checkpoint-path>
```

## 2. Pretrained Data Generator

This folder is used to generate the pretrained input data required by ReGeoDTA for the Davis, KIBA, and BindingDB datasets.

### Programs

#### `generate_drug_graph.py`

Generates molecular graph data from drug SMILES strings, including atom node features and graph edge indices.

```bash
python generate_drug_graph.py \
  --dataset <davis|kiba|bindingdb> \
  --data-root <dataset-path> \
  --output-root <output-path>
```

#### `generate_esm.py`

Uses ESMFold to predict protein structures from amino acid sequences and saves the results as PDB files.

```bash
python generate_esm.py \
  --dataset <davis|kiba|bindingdb> \
  --data-root <dataset-path> \
  --output-root <output-path> \
  --model-path <esmfold-model-path>
```

#### `generate_prottrans.py`

Uses ProtTrans to generate residue-level protein embeddings and saves them as tensor files.

```bash
python generate_prottrans.py \
  --dataset <davis|kiba|bindingdb> \
  --data-root <dataset-path> \
  --output-root <output-path> \
  --model-path <prottrans-model-path>
```
