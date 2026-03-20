# config.py

import os
import torch


class Config:
    """
    Global configuration for CTRA-Net
    """

    # ======================
    # 1. Basic Settings
    # ======================
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======================
    # 2. Dataset Settings
    # ======================
    dataset = "ASSISTments2009"   # options: ASSISTments2009, Statics2011, etc.
    data_dir = "./data/"

    # sequence settings
    max_seq_len = 200
    train_ratio = 0.8

    # dataset-specific (will be overwritten dynamically if needed)
    num_students = None
    num_questions = None
    num_concepts = None

    # ======================
    # 3. Model Settings
    # ======================
    # embedding
    embed_dim = 128

    # GRU (temporal encoder)
    hidden_dim = 128
    num_layers = 1

    # Graph (CT-GAP)
    graph_hidden_dim = 128

    # Memory Bank (TRMB)
    memory_size = 64
    memory_dim = 128
    memory_write_lambda = 0.5
    time_decay_lambda = 0.01
    temperature = 0.5

    # ======================
    # 4. Training Settings
    # ======================
    batch_size = 64
    epochs = 100
    learning_rate = 1e-3
    weight_decay = 1e-5

    # optimizer
    optimizer = "adam"

    # dropout
    dropout = 0.2

    # ======================
    # 5. Loss Weights
    # ======================
    lambda_align = 0.1     # alignment loss weight
    lambda_entropy = 0.01  # entropy regularization

    # ======================
    # 6. Paths
    # ======================
    save_dir = "./checkpoints/"
    log_dir = "./logs/"

    # ======================
    # 7. Misc
    # ======================
    num_workers = 4
    pin_memory = True

    # ======================
    # 8. Utility Functions
    # ======================
    @staticmethod
    def create_dirs():
        os.makedirs(Config.save_dir, exist_ok=True)
        os.makedirs(Config.log_dir, exist_ok=True)


def set_seed(seed):
    """
    Fix random seed for reproducibility
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_config():
    """
    Initialize config and environment
    """
    cfg = Config()
    set_seed(cfg.seed)
    cfg.create_dirs()
    return cfg