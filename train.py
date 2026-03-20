# train.py

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from config import get_config
from _model import CTRANet
from data_loader import get_dataloaders
from utils import compute_auc, compute_accuracy, AverageMeter


def compute_loss(outputs, labels, cfg):
    """
    Compute total loss:
    1) BCE prediction loss
    2) relational alignment loss
    3) entropy regularization for memory attention
    """
    pred = outputs["pred"]              # [B, T]
    mask = outputs["mask"].float()      # [B, T]
    z_t = outputs["z_t"]                # [B, T, D]
    c_t = outputs["c_t"]                # [B, T, D]
    memory_attn = outputs["memory_attn"]  # [B, T, M]

    # ---------------------------
    # 1. BCE loss
    # ---------------------------
    bce_fn = nn.BCELoss(reduction="none")
    bce_loss = bce_fn(pred, labels.float())   # [B, T]
    bce_loss = (bce_loss * mask).sum() / (mask.sum() + 1e-8)

    # ---------------------------
    # 2. Alignment loss
    # ---------------------------
    align_dist = torch.norm(z_t - c_t, p=2, dim=-1)   # [B, T]
    align_loss = (align_dist * mask).sum() / (mask.sum() + 1e-8)

    # ---------------------------
    # 3. Entropy regularization
    # ---------------------------
    entropy = -torch.sum(memory_attn * torch.log(memory_attn + 1e-8), dim=-1)  # [B, T]
    entropy_loss = -(entropy * mask).sum() / (mask.sum() + 1e-8)
    # negative sign because we want to maximize entropy, but optimize by minimizing loss

    total_loss = (
        bce_loss
        + cfg.lambda_align * align_loss
        + cfg.lambda_entropy * entropy_loss
    )

    loss_dict = {
        "total_loss": total_loss,
        "bce_loss": bce_loss.detach().item(),
        "align_loss": align_loss.detach().item(),
        "entropy_loss": entropy_loss.detach().item(),
    }
    return total_loss, loss_dict


def run_one_epoch(model, loader, optimizer, cfg, training=True):
    """
    Train or evaluate for one epoch
    """
    if training:
        model.train()
    else:
        model.eval()

    loss_meter = AverageMeter()
    bce_meter = AverageMeter()
    align_meter = AverageMeter()
    entropy_meter = AverageMeter()

    all_preds = []
    all_labels = []

    for batch in loader:
        # expected batch dict
        # question_seq: [B, T]
        # response_seq: [B, T]
        # target_seq:   [B, T]
        # mask_seq:     [B, T]
        question_seq = batch["question_seq"].to(cfg.device)
        response_seq = batch["response_seq"].to(cfg.device)
        target_seq = batch["target_seq"].to(cfg.device).float()
        mask_seq = batch["mask_seq"].to(cfg.device).float()

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(
                question_seq=question_seq,
                response_seq=response_seq,
                mask_seq=mask_seq
            )

            # 强制使用 batch 中的 mask，防止模型内部 mask 不一致
            outputs["mask"] = mask_seq

            loss, loss_dict = compute_loss(outputs, target_seq, cfg)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        loss_meter.update(loss.item(), question_seq.size(0))
        bce_meter.update(loss_dict["bce_loss"], question_seq.size(0))
        align_meter.update(loss_dict["align_loss"], question_seq.size(0))
        entropy_meter.update(loss_dict["entropy_loss"], question_seq.size(0))

        pred = outputs["pred"].detach()
        valid_pred = pred[mask_seq > 0].cpu()
        valid_label = target_seq[mask_seq > 0].cpu()

        all_preds.append(valid_pred)
        all_labels.append(valid_label)

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    auc = compute_auc(all_labels, all_preds)
    acc = compute_accuracy(all_labels, all_preds)

    metrics = {
        "loss": loss_meter.avg,
        "bce_loss": bce_meter.avg,
        "align_loss": align_meter.avg,
        "entropy_loss": entropy_meter.avg,
        "auc": auc,
        "acc": acc,
    }
    return metrics


def build_optimizer(model, cfg):
    if cfg.optimizer.lower() == "adam":
        return optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")


def save_checkpoint(state, save_path):
    torch.save(state, save_path)


def main():
    cfg = get_config()

    print("=" * 60)
    print("CTRA-Net Training")
    print(f"Dataset        : {cfg.dataset}")
    print(f"Device         : {cfg.device}")
    print(f"Epochs         : {cfg.epochs}")
    print(f"Batch size     : {cfg.batch_size}")
    print(f"Learning rate  : {cfg.learning_rate}")
    print("=" * 60)

    # dataloaders
    train_loader, val_loader, test_loader, data_info = get_dataloaders(cfg)

    # update dataset-specific config
    cfg.num_questions = data_info["num_questions"]
    cfg.num_concepts = data_info.get("num_concepts", data_info["num_questions"])

    print(f"Num Questions  : {cfg.num_questions}")
    print(f"Num Concepts   : {cfg.num_concepts}")

    # model
    model = CTRANet(cfg).to(cfg.device)

    # optimizer
    optimizer = build_optimizer(model, cfg)

    best_val_auc = 0.0
    best_epoch = 0
    best_model_path = os.path.join(cfg.save_dir, f"{cfg.dataset}_best_model.pth")

    for epoch in range(1, cfg.epochs + 1):
        start_time = time.time()

        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            cfg=cfg,
            training=True
        )

        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            cfg=cfg,
            training=False
        )

        elapsed = time.time() - start_time

        print(
            f"[Epoch {epoch:03d}/{cfg.epochs:03d}] "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train AUC: {train_metrics['auc']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f} | "
            f"Val ACC: {val_metrics['acc']:.4f} | "
            f"Time: {elapsed:.2f}s"
        )

        # save best model
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_auc": best_val_auc,
                    "config": vars(cfg) if hasattr(cfg, "__dict__") else cfg.__class__.__dict__,
                },
                best_model_path
            )

            print(f"Best model saved to: {best_model_path}")

    print("\nTraining finished.")
    print(f"Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")

    # test
    checkpoint = torch.load(best_model_path, map_location=cfg.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = run_one_epoch(
        model=model,
        loader=test_loader,
        optimizer=None,
        cfg=cfg,
        training=False
    )

    print("=" * 60)
    print("Test Results")
    print(f"Test Loss : {test_metrics['loss']:.4f}")
    print(f"Test AUC  : {test_metrics['auc']:.4f}")
    print(f"Test ACC  : {test_metrics['acc']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()