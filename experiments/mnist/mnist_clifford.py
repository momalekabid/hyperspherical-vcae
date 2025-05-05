import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.adam import Adam
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))
from clifford import ModelVAE, compute_loss

# params reconstructed from https://arxiv.org/abs/1804.00891
H_DIM = 128
Z_DIM = 10
BATCH_SIZE = 64
EPOCHS = 50 
KNN_EVAL_SAMPLES = [100, 600, 1000]
N_RUNS = 3 
Z_DIMS = [20, 40]
PATIENCE = 50
DELTA = 1e-3


# ----------------------------------------------------------------------
# Dataset & device setup
# ----------------------------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available() # mps not supported
    else "cpu"
)

# Define dataset and loaders
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > torch.rand_like(x)).float()),  # dynamic binarization
    ]
)

dataset = datasets.MNIST("../datasets", train=True, download=True, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
test_dataset = datasets.MNIST("../datasets", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(f"Using device: {device}")


def encode_dataset(model, data_loader, device):
    """Get latent representations for entire dataset"""
    model.eval()
    all_z = []
    all_labels = []

    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            mu, _ = model.encode(data.view(-1, 784))
            all_z.append(mu.cpu())
            all_labels.append(labels)

    return torch.cat(all_z, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy()


def perform_knn_evaluation(model, train_loader, test_loader, device):
    X_train, y_train = encode_dataset(model, train_loader, device)
    results = {}

    for n_samples in KNN_EVAL_SAMPLES:
        test_subset = torch.utils.data.Subset(
            test_loader.dataset,
            indices=torch.randperm(len(test_loader.dataset))[:n_samples],
        )
        test_loader_subset = DataLoader(test_subset, batch_size=BATCH_SIZE)

        X_test, y_test = encode_dataset(model, test_loader_subset, device)

        knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        results[n_samples] = accuracy

    return results


def visualize_reconstructions(
    model, test_loader, save_path="./visualizations", z_dim=None, num_examples=10
):
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    dim_suffix = f"_zdim{z_dim}" if z_dim is not None else ""
    data, _ = next(iter(test_loader))
    data = data[:num_examples].to(device)

    with torch.no_grad():
        (_, _), (_, _), _, x_recon = model(data)
        x_recon = x_recon.view(-1, 28, 28).cpu()
        data = data.view(-1, 28, 28).cpu()

    fig, axes = plt.subplots(2, num_examples, figsize=(2 * num_examples, 4))

    for i in range(num_examples):
        axes[0, i].imshow(data[i], cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original")

    for i in range(num_examples):
        axes[1, i].imshow(x_recon[i], cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstructed")

    plt.suptitle(f"Original vs Reconstructed Images (z_dim={z_dim})")
    plt.tight_layout()
    plt.savefig(f"{save_path}/reconstructions{dim_suffix}.png")
    plt.close()


def train_and_evaluate(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    device,
    epochs=EPOCHS,
    z_dim=None,
):
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: min((epoch + 1) / 50, 1.0))
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            (_, _), (q_z, p_z), _, x_recon = model(data)
            loss = compute_loss(model, data)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                (_, _), (q_z, p_z), _, x_recon = model(data)
                loss = compute_loss(model, data)
                total_loss += loss.item()

        val_loss = total_loss / len(val_loader)
        scheduler.step()

        print(
            f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
        )

        if val_loss < best_val_loss - DELTA:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    # visualize_latent_space(model, test_loader, z_dim=z_dim)
    visualize_reconstructions(model, test_loader, z_dim=z_dim)
    return perform_knn_evaluation(model, train_loader, test_loader, device)


def run_experiment(z_dim, device, n_runs=N_RUNS):
    results = {samples: [] for samples in KNN_EVAL_SAMPLES}

    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs} for z_dim = {z_dim}")
        model = ModelVAE(H_DIM, z_dim, device=device).to(device)
        optimizer = Adam(model.parameters(), lr=1e-4)

        run_results = train_and_evaluate(
            model, train_loader, val_loader, test_loader, optimizer, device, z_dim=z_dim
        )

        for n_samples, accuracy in run_results.items():
            results[n_samples].append(accuracy)

    return results


def calculate_statistics(results):
    return {
        n_samples: (np.mean(accuracies) * 100, np.std(accuracies) * 100)
        for n_samples, accuracies in results.items()
    }


if __name__ == "__main__":
    results_table = []

    for z_dim in Z_DIMS:
        print(f"\nRunning experiments for z_dim = {z_dim}")
        results = run_experiment(z_dim, device)
        stats = calculate_statistics(results)

        for n_samples in KNN_EVAL_SAMPLES:
            mean, std = stats[n_samples]
            results_table.append(
                {
                    "d": z_dim,
                    "n_samples": n_samples,
                    "Clifford-VAE": f"{mean:.1f}±{std:.1f}",
                }
            )
            print(f"d={z_dim}, n_samples={n_samples}: {mean:.1f}±{std:.1f}")

    df = pd.DataFrame(results_table)
    df = df.pivot(index="d", columns="n_samples", values="Clifford-VAE")

    print("\nFinal Results:")
    print(df.to_string())
    df.to_csv("clifford_vae_results.csv")

