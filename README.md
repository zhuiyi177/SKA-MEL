# SKA-MEL

This repository provides the implementation of **SKA-MEL: A Structural Knowledge-Augmented Method for Multimodal Entity Linking**.

SKA-MEL introduces structural semantic evidence into multimodal entity linking and jointly models textual, visual, cross-modal, and structural evidence through constrained dynamic arbitration and multi-branch self-distillation.

## Environment

It is recommended to use Conda to manage the environment.

The reference environment uses:

- Python 3.7
- PyTorch 1.11
- CUDA 11.3

Create and activate the environment:

```bash
conda create -n skamel python=3.7 -y
conda activate skamel
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Data

Benchmark datasets are prepared following [MMoE](https://github.com/zhiweihu1103/MEL-MMoE).

Update the dataset and resource paths in `configs/` according to your local environment.

Candidate Relation–Tail units can be constructed from entity metadata using:

```bash
python scripts/build_entity_triples.py --config configs/wikimel.yaml
```

Replace the configuration file as needed for RichpediaMEL or WikiDiverse.

## Training

Train SKA-MEL on WikiMEL:

```bash
python main.py --config configs/wikimel.yaml
```

For the other datasets:

```bash
python main.py --config configs/richpediamel.yaml
python main.py --config configs/wikidiverse.yaml
```

## Evaluation

Evaluate a trained checkpoint:

```bash
python main.py \
  --config configs/wikimel.yaml \
  --test-only-ckpt /path/to/model.ckpt
```

The same evaluation procedure applies to RichpediaMEL and WikiDiverse by changing the configuration file.

## Acknowledgements

We thank the authors of MMoE for making their code publicly available.
