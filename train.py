import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from datasets import build_dataset
from models import build_model
from utils.config import load_config
from utils.losses import BCEDiceLoss
from utils.metrics import add_counts, confusion_counts, metrics_from_counts
from utils.utils import set_seed


def evaluate(model, loader, criterion, device, threshold):
    model.eval()
    counts = {key: 0 for key in ("tp", "tn", "fp", "fn")}
    loss_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for images, masks, _ in loader:
            images, masks = images.to(device), masks.to(device)
            output = model(images)
            loss_sum += criterion(output, masks).item() * images.size(0)
            sample_count += images.size(0)
            predictions = output.cpu().numpy()
            add_counts(counts, confusion_counts(predictions, masks.cpu().numpy(), threshold))
    return loss_sum / sample_count, metrics_from_counts(counts)


def main():
    parser = argparse.ArgumentParser(description="Train a LAMP variant.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, help="Override experiment.seed for repeated runs.")
    parser.add_argument("--output", help="Override experiment.output_dir.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
    set_seed(cfg["experiment"]["seed"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output or cfg["experiment"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    train_set = build_dataset(cfg["dataset"], "train", cfg["augmentation"])
    val_set = build_dataset(cfg["dataset"], "val", False)
    common = dict(num_workers=cfg["dataset"]["num_workers"], pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, batch_size=cfg["training"]["batch_size"], shuffle=True, **common)
    val_loader = DataLoader(val_set, batch_size=cfg["evaluation"]["batch_size"], shuffle=False, **common)
    model = build_model(cfg).to(device)
    train_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"],
                                  weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["t_max"], eta_min=train_cfg["eta_min"])
    criterion = BCEDiceLoss()
    start_epoch, best_val_loss = 1, float("inf")
    if args.resume:
        payload = torch.load(args.resume, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch = payload["epoch"] + 1
        best_val_loss = payload.get("best_val_loss", payload.get("min_loss", float("inf")))
    log_path = output / "training.jsonl"
    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        model.train()
        loss_sum = 0.0
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * images.size(0)
        scheduler.step()
        val_loss, metrics = evaluate(
            model, val_loader, criterion, device, cfg["evaluation"]["threshold"])
        record = {"epoch": epoch, "loss": loss_sum / len(train_set),
                  "val_loss": val_loss,
                  "learning_rate": optimizer.param_groups[0]["lr"], **metrics}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        state = {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "scheduler_state_dict": scheduler.state_dict(),
                 "best_val_loss": min(best_val_loss, val_loss), "config": cfg}
        torch.save(state, output / "latest.pth")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state["best_val_loss"] = best_val_loss
            torch.save(state, output / "best.pth")
        if epoch % train_cfg["save_every"] == 0:
            torch.save(state, output / f"epoch_{epoch:03d}.pth")
        print(f"epoch={epoch:03d} train_loss={record['loss']:.4f} "
              f"val_loss={val_loss:.4f} DSC={metrics['DSC']:.4f}")


if __name__ == "__main__":
    main()
