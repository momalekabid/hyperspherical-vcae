import argparse
import yaml
from pathlib import Path
from datetime import datetime
from itertools import product
import json
import math
from typing import Dict, Any, Tuple, List, Optional

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.vcae import VAE


_DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Using device: {_DEVICE}")

def load_config(config_path: Path) -> Dict[str, Any]:
    """ YAML config file."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_dataset(name: str, batch_size: int):
    transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])

    dataset_map = {
        "fashionmnist": (datasets.FashionMNIST, 1),
        "cifar10": (datasets.CIFAR10, 3),
    }

    ds_cls, in_channels = dataset_map[name]

    train_ds = ds_cls(root="./data", train=True, download=True, transform=transform)
    test_ds = ds_cls(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, in_channels


def encode_dataset(model: VAE, dl: DataLoader, device: torch.device):
    model.eval()
    latents, labels = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            mu, _ = model.encoder(x)
            latents.append(mu.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(latents), np.concatenate(labels)


def plot_fft_spectrum(model: VAE, test_loader: DataLoader, save_path: Path):
    """
    Generate and save plots of FFT properties from encoded vectors.
    
    Args:
        model: The VAE model
        test_loader: DataLoader for test data
        save_path: Directory to save plots
    """
    model.eval()
    all_vectors = []
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(_DEVICE)
            mu, _ = model.encoder(data)
            all_vectors.append(mu.cpu())
    
    vectors = torch.cat(all_vectors, dim=0)
    
    fft_vectors = torch.fft.fft(vectors, dim=1)
    fft_magnitudes = torch.abs(torch.fft.fftshift(fft_vectors, dim=1))
    mean_fft_magnitude = torch.mean(fft_magnitudes, dim=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mean_fft_magnitude.numpy())
    plt.axhline(y=1.0/math.sqrt(vectors.shape[1]), color='r', linestyle='--', 
                label=f'Uniform value (1/√{vectors.shape[1]})')
    plt.title("Average FFT Magnitude Spectrum of Encoded Vectors")
    plt.xlabel("Frequency Component")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.savefig(save_path / "encoder_fft_spectrum.png", dpi=300)
    plt.close()


def knn_eval(
    model: VAE,
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_samples: int,
    n_neighbors: int,
    n_runs: int,
    device: torch.device,
):
    train_latent, train_labels = encode_dataset(model, train_loader, device)
    test_latent, test_labels = encode_dataset(model, test_loader, device)

    accuracies, f1s, cms = [], [], []

    metric = "cosine" if model.distribution in ["powerspherical", "clifford"] else "euclidean"

    for _ in range(n_runs):
        if n_samples < len(test_labels):
            idxs = np.random.choice(len(test_labels), n_samples, replace=False)
            tl = test_latent[idxs]
            lb = test_labels[idxs]
        else:
            tl, lb = test_latent, test_labels

        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        knn.fit(train_latent, train_labels)
        pred = knn.predict(tl)

        accuracies.append(accuracy_score(lb, pred))
        f1s.append(f1_score(lb, pred, average="macro"))
        cms.append(confusion_matrix(lb, pred, labels=np.arange(lb.max()+1)))

    return {
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "n_samples": n_samples,
        "n_neighbors": n_neighbors,
        "n_runs": n_runs,
        "confusion_matrix": np.mean(cms, axis=0).tolist(),
    }


########################
# training 
########################


def save_checkpoint(
    model: VAE,
    optimizer: optim.Optimizer,
    epoch: int,
    ckpt_path: Path,
):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
    }
    torch.save(ckpt, ckpt_path)


def load_checkpoint(model: VAE, optimizer: optim.Optimizer, ckpt_path: Path) -> int:
    ckpt = torch.load(ckpt_path, map_location=_DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    return ckpt["epoch"]


def train_one_experiment(
    cfg: Dict[str, Any],
    dataset_name: str,
    distribution: str,
    beta: float,
    latent_dim: int,
    run_id: int,
    base_output: Path,
    resume: bool = False,
) -> Dict[str, Any]:
    """Trains/evaluates for a single configuration"""

    train_loader, test_loader, in_channels = get_dataset(dataset_name, cfg["batch_size"])

    exp_name = f"{dataset_name}_dist{distribution}_beta{beta}_latent{latent_dim}_run{run_id}"
    exp_dir = base_output / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = exp_dir / "metrics.json"
    if resume and metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)

    model = VAE(latent_dim=latent_dim, in_channels=in_channels, distribution=distribution).to(_DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))

    ckpt_path = exp_dir / "checkpoint.pt"
    start_epoch = 0
    if resume and ckpt_path.exists():
        start_epoch = load_checkpoint(model, optimizer, ckpt_path) + 1

    print(f"[{exp_name}] starting (epochs={cfg['epochs']}) on {_DEVICE}")

    for epoch in tqdm(range(start_epoch, cfg["epochs"]), desc=exp_name, leave=False):
        model.train()
        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(_DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar, q_z, p_z = model(x)
            loss, _, _, _ = model.compute_loss(
                x, recon, mu, logvar, q_z, p_z, beta, cfg.get("gamma", 0.0)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % cfg.get("log_every", 1) == 0:
            print(f"[{exp_name}] epoch {epoch+1}/{cfg['epochs']}   loss={total_loss/len(train_loader):.4f}")

        # basic checkpointing every N epochs
        if (epoch + 1) % cfg.get("ckpt_interval", 5) == 0:
            save_checkpoint(model, optimizer, epoch, ckpt_path)

    save_checkpoint(model, optimizer, cfg["epochs"] - 1, ckpt_path)
    
    plot_fft_spectrum(model, test_loader, exp_dir)
    
    tsne_path = exp_dir / "tsne_visualization.png"
    generate_tsne_plot(
        model,
        test_loader,
        tsne_path,
        n_samples=min(1000, cfg.get("tsne_samples", 1000)),
        perplexity=cfg.get("tsne_perplexity", 30),
        n_iter=cfg.get("tsne_iterations", 1000)
    )

    knn_metrics = knn_eval(
        model,
        train_loader,
        test_loader,
        cfg["knn_samples"],
        cfg["n_neighbors"],
        cfg["knn_runs"],
        _DEVICE,
    )

    # persist metrics
    with open(metrics_file, "w") as f:
        json.dump(knn_metrics, f, indent=4)

    # save confusion matrix plot
    plot_confusion_matrix(
        np.array(knn_metrics["confusion_matrix"]),
        exp_dir / "confusion_matrix.png",
    )

    return knn_metrics


##################################
# plotting
##################################
def generate_tsne_plot(
    model: VAE,
    test_loader: DataLoader,
    save_path: Path,
    n_samples: int = 1000,
    perplexity: int = 30,
    n_iter: int = 1000
):
    """
    Generate and save a t-SNE plot of the latent space.
    
    Args:
        model: The VAE model
        test_loader: DataLoader for test data
        save_path: Path to save the t-SNE plot
        n_samples: Number of samples to use for t-SNE
        perplexity: Perplexity parameter for t-SNE
        n_iter: Number of iterations for t-SNE
    """
    model.eval()
    latents, labels = [], []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(_DEVICE)
            mu, _ = model.encoder(data)
            latents.append(mu.cpu().numpy())
            labels.append(target.numpy())
            
            if len(np.concatenate(labels)) >= n_samples:
                break
    
    latents = np.concatenate(latents)[:n_samples]
    labels = np.concatenate(labels)[:n_samples]
    
    # t-SNE params passed through 
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    tsne_results = tsne.fit_transform(latents)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10', alpha=0.8)
    plt.colorbar(scatter, label='Class')
    plt.title(f"t-SNE Visualization of Latent Space ({model.distribution} distribution)")
    plt.xlabel("c1")
    plt.ylabel("c2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_results(results: Dict[Tuple[str, float], Dict[str, List[Any]]], save_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    styles = {
        ("gaussian", 1.0): ("green", "-", "Gaussian (β=1)"),
        ("gaussian", 0.1): ("green", "--", "Gaussian (β=0.1)"),
        ("powerspherical", 1.0): ("blue", "-", "PowerSpherical (β=1)"),
        ("powerspherical", 0.1): ("blue", "--", "PowerSpherical (β=0.1)"),
        ("clifford", 1.0): ("red", "-", "Clifford (β=1)"),
        ("clifford", 0.1): ("red", "--", "Clifford (β=0.1)"),
    }

    all_latent_dims = set()
    has_data = False

    for key, metrics in results.items():
        latent_dims = metrics.get("latent_dims", [])
        if not latent_dims:
            continue
            
        has_data = True
        all_latent_dims.update(latent_dims)

        # line style info 
        color, style, label = styles.get(key, ("black", "-", str(key)))
        
        sorted_indices = sorted(range(len(latent_dims)), key=lambda i: latent_dims[i])
        sorted_dims = [latent_dims[i] for i in sorted_indices]
        sorted_acc_mean = [metrics["acc_mean"][i] for i in sorted_indices]
        sorted_acc_std = [metrics["acc_std"][i] for i in sorted_indices]
        sorted_f1_mean = [metrics["f1_mean"][i] for i in sorted_indices]
        sorted_f1_std = [metrics["f1_std"][i] for i in sorted_indices]
        
        # plot on both axes
        ax1.errorbar(sorted_dims, sorted_acc_mean, yerr=sorted_acc_std, 
                    color=color, linestyle=style, marker="o", capsize=4, label=label)
        ax2.errorbar(sorted_dims, sorted_f1_mean, yerr=sorted_f1_std, 
                    color=color, linestyle=style, marker="o", capsize=4, label=label)

    if has_data:
        for ax in (ax1, ax2):
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted(all_latent_dims))
            ax.set_xlabel("Latent Dimension")
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax1.set_ylabel("Accuracy")
        ax2.set_ylabel("F1 Score")
        ax1.set_title("Accuracy vs Latent Dim")
        ax2.set_title("F1 Score vs Latent Dim")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, "No data available yet", 
                   horizontalalignment='center', verticalalignment='center')
            ax.set_xticks([])
            ax.set_yticks([])
        plt.savefig(save_path, dpi=300)
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Latent dimension vs KNN evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoints/metrics")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    base_output = Path(cfg.get("output_dir", "results"))
    base_output.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results: Dict[Tuple[str, float], Dict[str, List[Any]]] = {}

    combinations = list(product(cfg["distributions"], cfg["betas"], cfg["latent_dims"], range(cfg["runs"])))

    seed = cfg.get("seed", 0)
    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        for dist, beta, ld, run in combinations:
            key = (dist, beta)
            results.setdefault(key, {
                "latent_dims": [], 
                "acc_mean": [], 
                "acc_std": [], 
                "f1_mean": [], 
                "f1_std": []
            })

            m = train_one_experiment(
                cfg,
                cfg["dataset"],
                dist,
                beta,
                ld,
                run,
                base_output,
                resume=args.resume,
            )

            results[key]["latent_dims"].append(ld)
            results[key]["acc_mean"].append(m["accuracy_mean"])
            results[key]["acc_std"].append(m["accuracy_std"])
            results[key]["f1_mean"].append(m["f1_mean"])
            results[key]["f1_std"].append(m["f1_std"])

            try:
                plot_results(results, base_output / f"plot_{timestamp}_intermediate.png")
                
                # save intermediate results
                with open(base_output / f"results_{timestamp}_intermediate.json", "w") as f:
                    serializable_results = {f"{k[0]}_{k[1]}": v for k, v in results.items()}
                    json.dump(serializable_results, f, indent=2)
            except Exception as e:
                print(f"Warning: Intermediate saving/plotting failed: {e}")
    except KeyboardInterrupt:
        plot_results(results, base_output / f"plot_{timestamp}.png")
    
    # plot final results
    plot_results(results, base_output / f"plot_{timestamp}.png")


if __name__ == "__main__":
    main()