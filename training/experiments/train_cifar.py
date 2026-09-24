"""
End-to-end CIFAR-10 Training with Virtual Multi-Rank SkipReduce + Domain Transforms.
Simulates N-rank distributed data parallel training on a single GPU.
"""

import argparse
import json
import os
import sys
import time

# Ensure repo root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from training.core.ring import simulate_skipreduce_ring, ErrorFeedbackBuffer
from training.core.sparsity import GradientSparsityTracker
from training.models import get_cifar_resnet50





def get_cifar10_loaders(data_dir: str, micro_batch_size: int, num_ranks: int):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    # Total step batch size = micro_batch_size * num_ranks
    step_batch_size = micro_batch_size * num_ranks
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=step_batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2
    )

    return trainloader, testloader


def create_cifar_resnet50():
    """Builds a CIFAR-10 adapted ResNet-50 (standard 3x3 conv1, identity maxpool)."""
    return get_cifar_resnet50(num_classes=10)


def evaluate(model, testloader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return val_loss / total, 100.0 * correct / total


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    trainloader, testloader = get_cifar10_loaders(args.data_dir, args.micro_batch_size, args.ranks)
    model = create_cifar_resnet50().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ef_buffer = ErrorFeedbackBuffer(num_ranks=args.ranks) if args.use_ef else None

    # Gradient sparsity tracker for future-proof profiling
    tracker = (
        GradientSparsityTracker(model, sample_per_epoch=args.sample_per_epoch)
        if getattr(args, "track_sparsity", True)
        else None
    )

    # Run metadata & tracking
    run_id = f"ranks{args.ranks}_skip{args.skip}_{args.transform}_r{args.retention}_{'ef' if args.use_ef else 'noef'}_{int(time.time())}"
    history = {
        "run_id": run_id,
        "ranks": args.ranks,
        "skip": args.skip,
        "transform": args.transform,
        "retention_ratio": args.retention,
        "use_ef": args.use_ef,
        "epochs": args.epochs,
        "lr": args.lr,
        "micro_batch_size": args.micro_batch_size,
        "step_batch_size": args.micro_batch_size * args.ranks,
        "epochs_data": []
    }


    print("\n" + "=" * 80)
    print(f"Run ID: {run_id}")
    print(f"Ranks: {args.ranks} | Skip: {args.skip} | Transform: {args.transform} | Ret: {args.retention} | EF: {args.use_ef}")
    print("=" * 80)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        epoch_cos_sims = []
        epoch_rel_errors = []

        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Split the step batch into N virtual micro-batches
            micro_inputs = torch.chunk(inputs, args.ranks, dim=0)
            micro_targets = torch.chunk(targets, args.ranks, dim=0)

            # Collect gradients from each virtual rank
            rank_grads = {name: [] for name, p in model.named_parameters() if p.requires_grad}

            for r in range(args.ranks):
                model.zero_grad()
                outputs = model(micro_inputs[r])
                loss = criterion(outputs, micro_targets[r])
                loss.backward()

                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        rank_grads[name].append(param.grad.detach().clone())

                running_loss += loss.item() * micro_targets[r].size(0)
                total_samples += micro_targets[r].size(0)

            # Simulate SkipReduce collective across ranks for each parameter
            step_cos_sims = []
            step_rel_errors = []

            for p_idx, (name, param) in enumerate(model.named_parameters()):
                if not param.requires_grad or len(rank_grads[name]) == 0:
                    continue

                reduced_grad, stats = simulate_skipreduce_ring(
                    grads=rank_grads[name],
                    num_ranks=args.ranks,
                    s=args.skip,
                    transform_type=args.transform,
                    retention_ratio=args.retention,
                    param_id=p_idx,
                    ef_buffer=ef_buffer
                )

                param.grad = reduced_grad
                step_cos_sims.append(stats["cos_sim"])
                step_rel_errors.append(stats["rel_l2_error"])

            if tracker and tracker.should_sample(batch_idx):
                tracker.record_step(batch_idx)

            optimizer.step()

            epoch_cos_sims.append(sum(step_cos_sims) / len(step_cos_sims))
            epoch_rel_errors.append(sum(step_rel_errors) / len(step_rel_errors))

        scheduler.step()
        epoch_time = time.time() - start_time

        train_loss = running_loss / total_samples
        val_loss, val_acc = evaluate(model, testloader, criterion, device)
        mean_cos_sim = sum(epoch_cos_sims) / len(epoch_cos_sims)
        mean_rel_error = sum(epoch_rel_errors) / len(epoch_rel_errors)

        print(
            f"Epoch {epoch+1:02d}/{args.epochs:02d} [{epoch_time:.1f}s] | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | CosSim: {mean_cos_sim:.4f} | RelErr: {mean_rel_error:.4f}"
        )

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "mean_cos_sim": round(mean_cos_sim, 4),
            "mean_rel_error": round(mean_rel_error, 4),
            "epoch_time_s": round(epoch_time, 2),
        }
        if tracker:
            epoch_record["gradient_sparsity"] = tracker.finish_epoch(epoch + 1)
        history["epochs_data"].append(epoch_record)

    if tracker:
        history["gradient_sparsity_summary"] = tracker.get_summary()

    # Save output log
    os.makedirs(args.log_dir, exist_ok=True)
    out_file = os.path.join(args.log_dir, f"{run_id}.json")
    with open(out_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved experiment record to: {out_file}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 on CIFAR-10 with Simulated SkipReduce")
    parser.add_argument("--ranks", type=int, default=4, help="Number of virtual ranks")
    parser.add_argument("--skip", type=int, default=0, help="Number of skipped steps s")
    parser.add_argument("--transform", type=str, default="none", choices=["none", "dct", "hadamard"])
    parser.add_argument("--retention", type=float, default=0.0, help="Retention ratio r in [0, 1]")
    parser.add_argument("--use-ef", action="store_true", help="Enable Error Feedback buffer")
    parser.add_argument("--track-sparsity", action=argparse.BooleanOptionalAction, default=True, help="Track and log mathematical gradient sparsity metrics (E10, Hoyer) per epoch")
    parser.add_argument("--sample-per-epoch", type=int, default=5, help="Number of gradient sample batches per epoch")
    parser.add_argument("--epochs", type=int, default=20, help="Total training epochs")

    parser.add_argument("--micro-batch-size", type=int, default=32, help="Batch size per virtual rank")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-dir", type=str, default="./data", help="CIFAR-10 dataset path")
    parser.add_argument("--log-dir", type=str, default="training/logs", help="Directory for JSON run logs")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
