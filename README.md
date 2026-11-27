# FOUL: Computation and Communication Efficient Federated Unlearning via On-server Gradient Conflict Mitigation and Expression

Protocolo de pesquisa com CNN treinável, controles de pruning, validação separada e diagnósticos: [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md). Os novos comandos usam checkpoints específicos por experimento.

Protocolo de remoção completa dos exemplos de uma raça: [EXAMPLE_REMOVAL_PROTOCOL.md](EXAMPLE_REMOVAL_PROTOCOL.md). A referência retreinada também exclui a raça da tarefa cachorro/gato.

<div align="center">

[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue?style=for-the-badge)](https://cvpr.thecvf.com/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green?style=for-the-badge&logo=python)](https://www.python.org/)
[![License: GPL v2](https://img.shields.io/badge/License-GPLv2-yellow?style=for-the-badge)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
[![Stars](https://img.shields.io/github/stars/skydvn/FOUL?style=for-the-badge)](https://github.com/skydvn/FOUL/stargazers)

**[Paper](#) | [Project Page](#) | [BibTeX](#citation)**

</div>

---

## 📖 Overview

**FOUL** (**F**ederated **U**n**l**earning via **O**n-server Gradient Conflict Mitigation and **E**xpression) is a novel federated unlearning framework that addresses the "right to be forgotten" problem in federated learning (FL) systems — without incurring the full cost of retraining from scratch.

Standard federated unlearning approaches require either full model retraining or heavy client-side computation, making them impractical at scale. FOUL decouples the unlearning burden from clients entirely by resolving **gradient conflicts on the server side**, dramatically reducing both communication overhead and computational cost while preserving model performance on remaining clients.

<div align="center">
  <img src="FOUL Architecture.png" alt="FOUL Framework Overview" width="80%"/>
  <p><em>Overview of the FOUL framework architecture.</em></p>
</div>

---

## ✨ Key Contributions

- **On-server Gradient Conflict Mitigation:** Identifies and resolves conflicts between the gradients of data to be forgotten and data to be retained — all on the server, without requiring clients to participate in the unlearning round.
- **Communication Efficiency:** Clients send updates only once; no additional unlearning rounds are pushed back to clients, slashing communication rounds compared to prior work.
- **Computation Efficiency:** Eliminates costly client-side retraining or fine-tuning, making FOUL scalable to large federated networks.
- **Flexible Unlearning Granularity:** Supports client-wise, class-wise, and domain-wise unlearning (e.g., selective domain removal in PACS).
- **Strong Empirical Performance:** Achieves unlearning efficacy comparable to exact retraining while maintaining high accuracy on retained data.

---

## 🏗️ Repository Structure

```
FOUL/
├── baselines/          # Competing federated unlearning methods
├── dataset/            # Dataset loading and preprocessing utilities
├── results/            # Experiment output logs and metrics
├── system/             # Core FOUL implementation
│   └── main.py         # Main training and unlearning entry point
├── splits.zip          # Pre-generated data splits
├── structure.png       # Framework architecture diagram
├── requirements.txt    # Python dependencies
└── env_cuda_latest.yaml  # Conda environment specification
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommend)

### Option 1: Conda (Recommended)

```bash
git clone https://github.com/skydvn/FOUL.git
cd FOUL
conda env create -f env_cuda_latest.yaml
conda activate foul
```

### Option 2: pip

```bash
git clone https://github.com/skydvn/FOUL.git
cd FOUL
pip install -r requirements.txt
```

---

## 📂 Dataset Preparation

### PACS (Multi-Domain)

PACS contains images across four domains: **Photo, Art Painting, Cartoon, Sketch**. In FOUL, clients are allocated based on domain-specific segmentation, enabling selective domain unlearning.

To generate IID and balanced data splits across clients:

```bash
python generate_PACS1.py iid balance -
```

> Pre-generated splits are also Can wavailable in [`splits.zip`](splits.zip).

---

## 🚀 Running Experiments

### Oxford Pet: pretrained breed model and semantic unlearning

#### Research comparison: breed forgetting and pruning

Use an annotated breed available in the generated dataset, such as `Chihuahua`.
First train one reference checkpoint with `-lm learn`; run each treatment with
`-lm unlearn` and a distinct `-go` value. The unlearning treatments all load
`system/models/OxfordPet/FOUL_server.pt`. They save their output under a
treatment-specific name, leaving that trained reference checkpoint intact.

From `system/`, with a generated Oxford dataset under `../dataset/OxfordPet224`:

```bash
python main.py -data OxfordPet -algo FOUL -nc 20 -lm learn -lmr 40 -gr 40 \
  -fb Chihuahua --dataset-root ../dataset/OxfordPet224 -go breed_train

python main.py -data OxfordPet -algo FOUL -nc 20 -lm unlearn -lmr 40 -gr 50 \
  -fb Chihuahua --dataset-root ../dataset/OxfordPet224 \
  --pruning-rate 0 -go breed_unlearning_only

python main.py -data OxfordPet -algo FOUL -nc 20 -lm unlearn -lmr 40 -gr 50 \
  -fb Chihuahua --dataset-root ../dataset/OxfordPet224 \
  --pruning-rate 0.5 --pet-pruning-only -go breed_pruning_only

python main.py -data OxfordPet -algo FOUL -nc 20 -lm unlearn -lmr 40 -gr 50 \
  -fb Chihuahua --dataset-root ../dataset/OxfordPet224 \
  --pruning-rate 0.5 -go breed_combined

python main.py -data OxfordPet -algo FOUL -nc 20 -lm learn -lmr 40 -gr 40 \
  -fb Chihuahua --dataset-root ../dataset/OxfordPet224 \
  --pet-retrain-reference -go breed_reference
```

Keep the same backbone, seed, dataset partition, checkpoint and unlearning
rounds when comparing treatments. Training must exceed
`--min-target-accuracy` on the target breed before a treatment starts. The
`results/*_hierarchy.json` files report target, other breeds in the same
family, and the other family separately, along with parameter counts, tensor
bytes, and post-training times. A lower target-breed accuracy alone is not
evidence of complete removal of learned information. The current ResNet18
path prunes only its projection; its frozen encoder remains full size.
The retraining reference excludes target images from the breed loss while
retaining them in the dog/cat family loss, matching the stated research target.
Use the same seed and training budget for the original and reference runs.

Generate a new 224-pixel partition from the official Oxford source. Keep the
existing 128-pixel partition for comparison; the generator will not overwrite
an existing output directory.

```bash
cd dataset
python generate_pet_datasets.py --dataset OxfordPet --root source/OxfordPet \
  --output OxfordPet224/OxfordPet --num-clients 20 --image-size 224
cd ../system
python main.py -data OxfordPet -m CNN -algo FOUL -nc 20 -gr 40 -lmr 30 \
  -ls 2 -lbs 32 -lm joint -beta_foul 3.5 -dev cpu -fb Chihuahua \
  --pet-backbone resnet18 --pet-learning-rate 0.001 \
  --pruning-rate 0.25 --min-target-accuracy 0.3 \
  --dataset-root ../dataset/OxfordPet224
```

The ImageNet-pretrained ResNet18 is frozen by default. Each client caches its
512-dimensional features, trains two output heads, and prunes channels in the
trainable projection using breed-specific activation scores before unlearning.
This keeps the residual encoder intact; pruning `fc1` reduces the trainable
model and the cost of local unlearning updates, but does **not** shrink the
ResNet18 encoder. The `--pruning-rate` option supports comparisons at 0.1,
0.25, and 0.5. Different runs should use a different `-go` goal to keep their
result files separate; the server checkpoint under `system/models/` is
overwritten by each run.

To fine-tune the last ResNet18 block after the heads have trained, add
`--pet-unfreeze-round 20 --pet-backbone-lr 0.00001`. This phase processes raw
images again and is much slower on CPU. The encoder is frozen before pruning
and unlearning, using its fine-tuned weights.

The experiment stops before pruning if Chihuahua test accuracy is below 30%,
since a model that has not learned the target breed cannot demonstrate its
forgetting. The result JSON reports family, target-breed, and non-target-breed
accuracy and separate family/breed test losses. ImageNet pretraining may
already encode information about Chihuahua, so the experiment measures removal
of the Oxford training contribution, not complete removal of all prior breed
knowledge.

All experiment commands are run from the `system/` directory:

```bash
cd system/
```

### 1. Train the Global Model (FedAvg)

```bash
python main.py --learn False --learn_round 50
```

### 2. Run FOUL Unlearning

```bash
python main.py --learn unlearn --learn_round 50 --algo FOUL
```

---

## 🔬 Baseline Methods

### Naive Retraining (Exact Unlearning Baseline)

Full retraining from scratch on retained clients — the gold standard for unlearning efficacy.

```bash
python main.py --learn joint -gr 150 --learn_round 100 \
               --algo Retrain --local_epochs 2 -lbs 32 --log
```

### FOUL (Ours)

```bash
python main.py --learn joint -gr 150 --learn_round 100 \
               --algo FOUL --local_epochs 2 -lbs 32 \
               --beta_foul 3.5 --log
```

### FUSED

```bash
python main.py -data PACS -m CNN -algo FUSED \
               -gr 20 -ls 3 -lr 0.01 \
               -nc 20 -nb 7 \
               -ncl 6 -sr 0.9 \
               -ut domain -ud 0 \
               -lm learn -lrnd 15 \
               -wb \
               -did 0
```

---

## ⚙️ Key Arguments

| Argument | Description | Default |
|---|---|---|
| `--algo` | Algorithm to run (`FOUL`, `Retrain`, `FUSED`, `FedAvg`) | `FedAvg` |
| `--learn` | Learning mode (`False`, `unlearn`, `joint`) | `False` |
| `--learn_round` / `-lrnd` | Number of unlearning rounds | `50` |
| `-gr` | Total global communication rounds | `150` |
| `--local_epochs` / `-ls` | Local training epochs per client | `3` |
| `-lbs` | Local batch size | `32` |
| `-lr` | Learning rate | `0.01` |
| `-nc` | Number of clients | `20` |
| `--beta_foul` | FOUL gradient conflict mitigation coefficient | `3.5` |
| `-data` | Dataset name (e.g., `PACS`) | — |
| `-m` | Model architecture (e.g., `CNN`) | — |
| `-ut` | Unlearning type (`domain`, `client`, `class`) | — |
| `-ud` | Unlearning target ID | `0` |
| `--log` | Enable logging | `False` |
| `-wb` | Enable Weights & Biases logging | `False` |
| `-did` | CUDA device ID | `0` |

---

## 📊 Results

FOUL achieves unlearning performance on par with full retraining while reducing communication rounds and eliminating client-side unlearning computation. Detailed results across datasets, unlearning granularities, and baselines are available in the `results/` directory and in our CVPR 2026 paper.

---

## 📄 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{foul2026cvpr,
  title     = {Computation and Communication Efficient Federated Unlearning via On-server Gradient Conflict Mitigation and Expression},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

---

## 📜 License

This project is licensed under the [GNU General Public License v2.0](LICENSE).

---

## 🙏 Acknowledgements

This codebase builds upon [PFLlib](https://github.com/TsingZ0/PFLlib). We thank the authors for their excellent open-source contribution to the federated learning community.

---

<div align="center">
  <sub>⭐ If you find FOUL useful, please consider starring this repository!</sub>
</div>
