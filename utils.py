import json
import os
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


class AverageMeter:
    """
    Computes and stores the average and current value.
    Commonly used for tracking training/evaluation losses.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.val = float(value)
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    """
    Create directory if it does not exist.
    """
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def compute_mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Mean Absolute Error.
    """
    return float(np.mean(np.abs(y_pred - y_true)))


def compute_rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Root Mean Square Error.
    """
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def compute_mape(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-6) -> float:
    """
    Mean Absolute Percentage Error (%).
    """
    denominator = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_pred - y_true) / denominator)) * 100.0)


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """
    Compute MAE / RMSE / MAPE together.
    """
    return {
        "mae": compute_mae(y_pred, y_true),
        "rmse": compute_rmse(y_pred, y_true),
        "mape": compute_mape(y_pred, y_true),
    }


def inverse_transform_if_needed(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    target_scaler: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse transform predictions and targets if target_scaler is provided.

    Args:
        y_pred: predicted values, shape [N, pred_len, C] or compatible
        y_true: ground truth values, shape [N, pred_len, C] or compatible
        target_scaler: scaler object with inverse_transform method

    Returns:
        y_pred_inv, y_true_inv
    """
    if target_scaler is None:
        return y_pred, y_true

    pred_shape = y_pred.shape
    true_shape = y_true.shape

    y_pred_2d = y_pred.reshape(-1, pred_shape[-1])
    y_true_2d = y_true.reshape(-1, true_shape[-1])

    y_pred_inv = target_scaler.inverse_transform(y_pred_2d).reshape(pred_shape)
    y_true_inv = target_scaler.inverse_transform(y_true_2d).reshape(true_shape)

    return y_pred_inv, y_true_inv


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Safely convert torch tensor to numpy array.
    """
    return tensor.detach().cpu().numpy()


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_in_millions(model: torch.nn.Module) -> float:
    """
    Count trainable parameters in millions.
    """
    return count_parameters(model) / 1e6


def save_checkpoint(
    save_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    best_metric: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save training checkpoint.
    """
    ensure_dir(os.path.dirname(save_path))

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, save_path)


def load_checkpoint(
    load_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load checkpoint and restore model/optimizer/scheduler states.

    Returns:
        checkpoint dictionary
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def save_json(data: Dict[str, Any], save_path: str) -> None:
    """
    Save dictionary to a json file.
    """
    ensure_dir(os.path.dirname(save_path))
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(load_path: str) -> Dict[str, Any]:
    """
    Load dictionary from a json file.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"JSON file not found: {load_path}")

    with open(load_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def format_metrics(metrics: Dict[str, float], precision: int = 6) -> str:
    """
    Format metric dictionary into a readable string.
    """
    parts = []
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            parts.append(f"{key}: {value:.{precision}f}")
        else:
            parts.append(f"{key}: {value}")
    return " | ".join(parts)


def print_metrics(title: str, metrics: Dict[str, float], precision: int = 6) -> None:
    """
    Pretty print metrics.
    """
    print(f"{title} -> {format_metrics(metrics, precision=precision)}")


def is_better(current: float, best: float, mode: str = "min") -> bool:
    """
    Compare whether current metric is better than best metric.

    Args:
        current: current metric value
        best: best metric value so far
        mode: "min" or "max"
    """
    if mode == "min":
        return current < best
    elif mode == "max":
        return current > best
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def get_monitor_value(metrics: Dict[str, float], key: str) -> float:
    """
    Get monitored metric value from dictionary.
    """
    if key not in metrics:
        raise KeyError(f"Metric key '{key}' not found. Available keys: {list(metrics.keys())}")
    return float(metrics[key])


def print_config(cfg: Any) -> None:
    """
    Pretty print configuration.
    Supports dataclass config or normal object with __dict__.
    """
    print("=" * 80)
    print("Configuration")
    print("=" * 80)

    if hasattr(cfg, "to_dict"):
        cfg_dict = cfg.to_dict()
    elif hasattr(cfg, "__dict__"):
        cfg_dict = vars(cfg)
    else:
        raise TypeError("Unsupported config type for print_config.")

    for key, value in cfg_dict.items():
        print(f"{key}: {value}")

    print("=" * 80)


if __name__ == "__main__":
    # Simple test
    preds = np.array([[10.0], [12.0], [9.5]])
    trues = np.array([[11.0], [10.0], [10.0]])

    metrics = compute_metrics(preds, trues)
    print_metrics("Test Metrics", metrics)

    meter = AverageMeter()
    meter.update(1.2, n=2)
    meter.update(0.8, n=2)
    print(f"AverageMeter avg: {meter.avg:.6f}")