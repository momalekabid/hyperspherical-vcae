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


################################################################################
# Utility functions
################################################################################

_DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"Using device: {_DEVICE}")

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


################################################################################
# Dataset helpers
################################################################################


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


################################################################################
# Latent evaluation helpers
################################################################################


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


################################################################################
# Unitary check functions
################################################################################

def check_unitary_property(vectors: torch.Tensor) -> Dict[str, float]:
    """
    Check if encoded vectors have unitary properties:
    1. Unit norm (normalized vectors)
    2. Flat frequency spectrum
    3. Orthogonality of vectors (if applicable)
    
    Args:
        vectors: Tensor of shape (batch_size, latent_dim)
    
    Returns:
        Dict with metrics about unitary properties
    """
    results = {}
    
    norms = torch.norm(vectors, dim=1)
    results["mean_norm"] = float(torch.mean(norms).item())
    results["norm_std"] = float(torch.std(norms).item())
    results["norm_deviation"] = float(torch.mean(torch.abs(norms - 1.0)).item())
    
    fft_vectors = torch.fft.fft(vectors, dim=1)
    fft_magnitudes = torch.abs(torch.fft.fftshift(fft_vectors, dim=1))
    
    mean_fft_magnitude = torch.mean(fft_magnitudes, dim=0)
    target_magnitude = torch.ones_like(mean_fft_magnitude) / math.sqrt(vectors.shape[1])
    
    results["fft_uniformity"] = float(F.mse_loss(mean_fft_magnitude, target_magnitude).item())
    results["fft_std"] = float(torch.std(mean_fft_magnitude).item())
    
    if vectors.shape[0] > 1 and vectors.shape[0] <= 1000 and vectors.shape[1] <= 1000:
        normalized_vectors = F.normalize(vectors, p=2, dim=1)
        dot_products = torch.mm(normalized_vectors, normalized_vectors.t())
        
        mask = ~torch.eye(dot_products.shape[0], dtype=torch.bool, device=dot_products.device)
        off_diag_dots = dot_products[mask]
        
        results["mean_abs_correlation"] = float(torch.mean(torch.abs(off_diag_dots)).item())
        results["max_abs_correlation"] = float(torch.max(torch.abs(off_diag_dots)).item())
    
    return results


def plot_unitary_properties(model: VAE, test_loader: DataLoader, save_path: Path):
    """
    Generate and save plots of unitary properties from encoder vectors.
    
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
            all_vectors.append(mu.cpu())  # must move to CPU before collecting
    
    vectors = torch.cat(all_vectors, dim=0)
    
    norms = torch.norm(vectors, dim=1)
    plt.figure(figsize=(10, 6))
    plt.hist(norms.numpy(), bins=50) 
    plt.axvline(x=1.0, color='r', linestyle='--', label='Unit norm')
    plt.title(f"Distribution of Encoder Vector Norms (mean={norms.mean():.4f}, std={norms.std():.4f})")
    plt.xlabel("L2 Norm")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(save_path / "encoder_vector_norms.png", dpi=300)
    plt.close()
    
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
    
    if vectors.shape[0] <= 100: 
        normalized_vectors = F.normalize(vectors[:100], p=2, dim=1)  # take at most 100 samples
        dot_products = torch.mm(normalized_vectors, normalized_vectors.t())
        
        plt.figure(figsize=(8, 8))
        plt.imshow(dot_products.numpy(), cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Cosine Similarity')
        plt.title("Cosine Similarity Between Encoded Vectors")
        plt.savefig(save_path / "encoder_vector_correlations.png", dpi=300)
        plt.close()


################################################################################
# t-SNE visualization
################################################################################

def generate_tsne_plot(
    model: VAE, 
    test_loader: DataLoader, 
    save_path: Path,
    n_samples: int = 1000,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42
):
    """
    Generate t-SNE visualization of the latent space.
    
    Args:
        model: The VAE model
        test_loader: DataLoader for test data
        save_path: Path to save the plot
        n_samples: Number of test samples to use
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations for t-SNE
        random_state: Random seed for reproducibility
    """
    model.eval()
    
    latents, labels = encode_dataset(model, test_loader, _DEVICE)
    
    if n_samples < len(labels):
        indices = np.random.choice(len(labels), n_samples, replace=False)
        latents = latents[indices]
        labels = labels[indices]
    
    print(f"Computing t-SNE for {len(latents)} samples (dim={latents.shape[1]})...")
    tsne = TSNE(
        n_components=2, 
        perplexity=min(perplexity, len(latents) - 1), 
        n_iter=n_iter, 
        random_state=random_state,
        verbose=1
    )
    
    try:
        tsne_results = tsne.fit_transform(latents)
        
        plt.figure(figsize=(10, 8))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                tsne_results[mask, 0],
                tsne_results[mask, 1],
                c=[colors[i]],
                label=f"Class {label}",
                alpha=0.7,
                s=20
            )
        
        plt.title(f"t-SNE Visualization of Latent Space ({model.distribution} distribution)")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        return True
    except Exception as e:
        print(f"t-SNE generation failed: {str(e)}")
        return False


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


################################################################################
# Checkpoint utilities
################################################################################


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


################################################################################
# Training utilities
################################################################################


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
    """Train a single configuration and return metrics."""

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

    # Resume training if checkpoint exists
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

    # Final checkpoint / save
    save_checkpoint(model, optimizer, cfg["epochs"] - 1, ckpt_path)

    # Check unitary properties of encoder vectors
    model.eval()
    with torch.no_grad():
        batch_data = next(iter(test_loader))[0].to(_DEVICE)
        mu, _ = model.encoder(batch_data)
        unitary_metrics = check_unitary_property(mu)
    
    # Save unitary properties plots
    plot_unitary_properties(model, test_loader, exp_dir)
    
    # Generate t-SNE visualization
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

    # Add unitary metrics to the output
    knn_metrics["unitary_metrics"] = unitary_metrics

    # persist metrics
    with open(metrics_file, "w") as f:
        json.dump(knn_metrics, f, indent=4)

    # save confusion matrix plot
    plot_confusion_matrix(
        np.array(knn_metrics["confusion_matrix"]),
        exp_dir / "confusion_matrix.png",
    )

    return knn_metrics


################################################################################
# Plotting helpers
################################################################################


def compile_results(results: Dict[Tuple[str, float], Dict[str, List[Any]]]) -> Dict[Tuple[str, float], Dict[str, List[Any]]]:
    """Utility just returns same dict but ensures latent dims are sorted."""
    compiled_results = {}
    
    for k, result_data in results.items():
        # Create a new entry for this key with empty lists
        compiled_results[k] = {
            "latent_dims": [],
            "acc_mean": [],
            "acc_std": [],
            "f1_mean": [],
            "f1_std": [],
            "unitary_metrics": result_data.get("unitary_metrics", [])
        }
        
        # Skip if no data available
        if not result_data.get("latent_dims") or len(result_data["latent_dims"]) == 0:
            continue
            
        # Zip and sort only if we have data
        tmp = list(zip(
            result_data["latent_dims"], 
            result_data.get("acc_mean", [0] * len(result_data["latent_dims"])), 
            result_data.get("acc_std", [0] * len(result_data["latent_dims"])), 
            result_data.get("f1_mean", [0] * len(result_data["latent_dims"])), 
            result_data.get("f1_std", [0] * len(result_data["latent_dims"]))
        ))
        tmp.sort(key=lambda t: t[0])
        
        if tmp:  # If we have data after sorting
            (ld, am, asd, fm, fsd) = zip(*tmp)
            compiled_results[k]["latent_dims"] = list(ld)
            compiled_results[k]["acc_mean"] = list(am)
            compiled_results[k]["acc_std"] = list(asd)
            compiled_results[k]["f1_mean"] = list(fm)
            compiled_results[k]["f1_std"] = list(fsd)
    
    return compiled_results


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

    # Track all latent dimensions to set x-axis ticks properly
    all_latent_dims = set()

    for key, m in results.items():
        if not m.get("latent_dims") or len(m["latent_dims"]) == 0:
            continue  # Skip if no data
            
        color, style, label = styles.get(key, ("black", "-", str(key)))
        
        all_latent_dims.update(m["latent_dims"])
        
        ax1.errorbar(m["latent_dims"], m["acc_mean"], yerr=m["acc_std"], 
                    color=color, linestyle=style, marker="o", capsize=4, label=label)
        ax2.errorbar(m["latent_dims"], m["f1_mean"], yerr=m["f1_std"], 
                    color=color, linestyle=style, marker="o", capsize=4, label=label)

    # Only configure axes if we have data
    if all_latent_dims:
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
        # Create a simple empty plot with a message if no data
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, "No data available yet", 
                   horizontalalignment='center', verticalalignment='center')
            ax.set_xticks([])
            ax.set_yticks([])
        plt.savefig(save_path, dpi=300)
    
    plt.close()


def plot_unitary_metrics(results: Dict[Tuple[str, float], Dict[str, List[Any]]], save_path: Path):
    """
    Plot unitary metrics across different latent dimensions and distributions.
    
    Args:
        results: Dictionary with results containing unitary metrics
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    styles = {
        ("gaussian", 1.0): ("green", "-", "Gaussian (β=1)"),
        ("gaussian", 0.1): ("green", "--", "Gaussian (β=0.1)"),
        ("powerspherical", 1.0): ("blue", "-", "PowerSpherical (β=1)"),
        ("powerspherical", 0.1): ("blue", "--", "PowerSpherical (β=0.1)"),
        ("clifford", 1.0): ("red", "-", "Clifford (β=1)"),
        ("clifford", 0.1): ("red", "--", "Clifford (β=0.1)"),
    }
    
    # Track if we have any data to plot
    has_data = False
    
    for key, metrics in results.items():
        if not metrics.get("latent_dims") or len(metrics["latent_dims"]) == 0:
            continue
            
        if "unitary_metrics" not in metrics or not metrics["unitary_metrics"]:
            continue
            
        has_data = True
        color, style, label = styles.get(key, ("black", "-", str(key)))
        
        # Extract metrics if they exist
        norm_devs = []
        fft_uniforms = []
        for i, ld in enumerate(metrics["latent_dims"]):
            if i < len(metrics["unitary_metrics"]):
                um = metrics["unitary_metrics"][i]
                norm_devs.append(um.get("norm_deviation", 0))
                fft_uniforms.append(um.get("fft_uniformity", 0))
        
        if norm_devs and fft_uniforms:
            ax1.plot(metrics["latent_dims"][:len(norm_devs)], norm_devs, 
                    color=color, linestyle=style, marker="o", label=label)
            ax2.plot(metrics["latent_dims"][:len(fft_uniforms)], fft_uniforms, 
                    color=color, linestyle=style, marker="o", label=label)
    
    if has_data:
        ax1.set_xscale("log", base=2)
        ax2.set_xscale("log", base=2)
        
        ax1.set_title("Vector Norm Deviation vs Latent Dim")
        ax2.set_title("FFT Uniformity vs Latent Dim")
        
        ax1.set_xlabel("Latent Dimension")
        ax2.set_xlabel("Latent Dimension")
        
        ax1.set_ylabel("Mean Deviation from Unit Norm")
        ax2.set_ylabel("FFT Uniformity Error")
        
        ax1.grid(True, which="both", linestyle="--", alpha=0.5)
        ax2.grid(True, which="both", linestyle="--", alpha=0.5)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
    else:
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, "No unitary metrics data available yet", 
                   horizontalalignment='center', verticalalignment='center')
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, save_path: Path):
    """Simple heat-map confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


################################################################################
# Comparison and t-SNE plots
################################################################################

def plot_tsne_comparison(results: Dict[Tuple[str, float], Dict[str, Any]], base_output: Path, timestamp: str):
    """
    Create a comparison of t-SNE visualizations across different distributions.
    
    Args:
        results: Dictionary with experiment results
        base_output: Base output directory
        timestamp: Timestamp string for naming
    """
    # This is a placeholder - in practice, we'd need to either:
    # 1. Load existing t-SNE plots and arrange them, or
    # 2. Generate new t-SNE plots for selected models
    
    # Since t-SNE is generated per-experiment, this function could create
    # a summary page showing thumbnails of individual t-SNE plots
    pass


################################################################################
# Main entry point
################################################################################


def main():
    parser = argparse.ArgumentParser(description="Latent dimension vs KNN evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoints/metrics")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    base_output = Path(cfg.get("output_dir", "results"))
    base_output.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize results dictionary with proper structure
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
                "f1_std": [],
                "unitary_metrics": []
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

            # Add results to appropriate key
            results[key]["latent_dims"].append(ld)
            results[key]["acc_mean"].append(m["accuracy_mean"])
            results[key]["acc_std"].append(m["accuracy_std"])
            results[key]["f1_mean"].append(m["f1_mean"])
            results[key]["f1_std"].append(m["f1_std"])
            results[key]["unitary_metrics"].append(m.get("unitary_metrics", {}))

            try:
                compiled = compile_results(results)
                plot_results(compiled, base_output / f"plot_{timestamp}_intermediate.png")
                plot_unitary_metrics(compiled, base_output / f"unitary_metrics_{timestamp}_intermediate.png")
                
                # Save intermediate results to prevent loss on interrupt
                with open(base_output / f"results_{timestamp}_intermediate.json", "w") as f:
                    # convert tuple keys to strings for JSON serialization
                    serializable_results = {f"{k[0]}_{k[1]}": v for k, v in results.items()}
                    json.dump(serializable_results, f, indent=2)
            except Exception as e:
                print(f"Warning: Intermediate saving/plotting failed")
    except KeyboardInterrupt:
        plot_results(compile_results(results), base_output / f"plot_{timestamp}.png")
        plot_unitary_metrics(results, base_output / f"unitary_metrics_{timestamp}.png")
    
    # plot final results
    plot_tsne_comparison(results, base_output, timestamp)
    plot_results(compile_results(results), base_output / f"plot_{timestamp}.png")
    plot_unitary_metrics(results, base_output / f"unitary_metrics_{timestamp}.png")


if __name__ == "__main__":
    main()