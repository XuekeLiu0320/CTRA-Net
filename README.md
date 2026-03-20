# CTRA-Net: Cross-Temporal Relational Alignment Network for Knowledge Tracing

CTRA-Net (Cross-Temporal Relational Alignment Network) is a unified deep learning framework for knowledge tracing that jointly models temporal dynamics and relational evolution in student learning processes.

Unlike conventional methods that rely on static graph structures or decoupled temporal modeling, CTRA-Net dynamically reconstructs knowledge relations conditioned on evolving learning states, enabling more stable and interpretable predictions.

The framework integrates temporal encoding, graph-based relational propagation, and memory-driven alignment into a coherent architecture.

---

🔍 Core Architecture

CTRA-Net consists of three key components:

Interaction & Temporal Encoder
Encodes student interaction sequences using embedding + GRU to capture learning dynamics.

CT-GAP (Cross-Temporal Graph Attention Propagation)
Dynamically generates time-conditioned relational structures via attention-guided graph propagation.

TRMB (Temporal-Relational Memory Bank)
Stores historical relational prototypes and enforces cross-temporal consistency through memory alignment.
---

✨ Key Features

⏱️ Dynamic coupling of temporal evolution and relational structure

🧩 Time-conditioned graph reconstruction via attention propagation

🧠 Memory-based relational alignment to reduce representation drift

🔄 End-to-end joint learning of prediction + structure + memory

📊 Designed for large-scale knowledge tracing datasets (e.g., ASSISTments, Statics)

---
📊 Evaluation Metrics

Prediction Performance: AUC, Accuracy

Loss Components:

BCE Loss (prediction)

Alignment Loss (relation consistency)

Entropy Regularization (memory diversity)
---

📁 Project Structure
```
CTRA-Net/
├── config.py        # Global configuration :contentReference[oaicite:0]{index=0}
├── data_loader.py   # Data preprocessing & loaders :contentReference[oaicite:1]{index=1}
├── model.py         # Main CTRA-Net model :contentReference[oaicite:2]{index=2}
├── modules.py       # Core modules (CT-GAP, TRMB, Embedding) :contentReference[oaicite:3]{index=3}
├── train.py         # Training pipeline :contentReference[oaicite:4]{index=4}
├── utils.py         # Utilities & metrics :contentReference[oaicite:5]{index=5}
├── data/            # Dataset directory
├── checkpoints/     # Saved models
└── logs/            # Training logs

```
⚙️ Data Processing Pipeline

The data loader performs:

Student-wise sequence construction

ID remapping (continuous indices)

Sequence segmentation (fixed length)

Padding & masking

Question graph construction (co-occurrence adjacency)

Graph is automatically built from training sequences.
---

## Training

### Basic Training
```bash
python train.py

```
### Key Configurations (config.py)
```bash
batch_size = 64
epochs = 100
learning_rate = 1e-3
max_seq_len = 200
embed_dim = 128
hidden_dim = 128
memory_size = 64

```
---
🧠 Model Workflow

At each time step t, CTRA-Net performs:

Encode interaction → temporal state

Generate relation-aware graph via CT-GAP

Retrieve historical relational context from memory

Align current relation with memory prototypes

Predict next response (t+1)
---

---
📈 Loss Function

The total loss consists of:

Prediction Loss

Binary cross-entropy for next-step prediction

Alignment Loss

Aligns current relational representation with memory

Entropy Regularization

Encourages diverse memory usage

Predict next response (t+1)
---
🧪 Evaluation

After training, the model reports:

AUC

Accuracy

Loss breakdown

Best model is saved automatically based on validation AUC.
---
🔧 Reproducibility

Fixed random seed (seed = 42)

Deterministic training setup

Full config control via config.py
---
📌 Notes

Graph adjacency is constructed dynamically (no external graph required)

Supports variable-length sequences with masking

Easily extendable to concept-aware KT via question_concept_map
---


## Citation

If you use this code in your research, please cite:

```bibtex
@article{ctranet2026,
  title={Cross-Temporal Relational Alignment for Unified Knowledge Tracing},
  author={Your Name},
  journal={},
  year={2026}
}
```

## License

This project is released under the MIT License.

