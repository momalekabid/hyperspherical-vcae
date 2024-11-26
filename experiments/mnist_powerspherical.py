import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.adam import Adam
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from models.vae_pws import ModelVAE, compute_loss

# constants
H_DIM = 128
Z_DIM = 10
BATCH_SIZE = 64
EPOCHS = 100
KNN_EVAL_SAMPLES = [100, 600, 1000]
N_RUNS = 20
Z_DIMS = [2, 5, 10, 20, 40]
PATIENCE = 10  # number of epochs to wait for improvement before stopping
DELTA = 0.001  # minimum change to qualify as an improvement
# device configuration
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# data loading
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: (x > torch.rand_like(x)).float()
        ),  # dynamic binarization
    ]
)
dataset = datasets.MNIST("../datasets", train=True, download=True, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
test_dataset = datasets.MNIST(
    "../datasets", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


def train_and_evaluate(
    model, train_loader, val_loader, test_loader, optimizer, device, epochs=EPOCHS
):
    def lr_lambda(epoch):
        warmup_epochs = 50
        if epoch < warmup_epochs:
            return epoch / warmup_epochs  # linear warm-up
        return 1.0  # maintain full learning rate after warm-up

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_mb, _ in train_loader:
            x_mb = x_mb.to(device)
            optimizer.zero_grad()
            loss = compute_loss(model, x_mb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_mb, _ in val_loader:
                x_mb = x_mb.to(device)
                val_loss += compute_loss(model, x_mb).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        scheduler.step()

        if val_loss < best_val_loss - DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_model is not None:
        model.load_state_dict(best_model)

    return perform_knn_evaluation(model, train_loader, test_loader)


def get_latent_representations(model, data_loader, n_samples=100):
    model.eval()
    latent_vectors, labels = [], []
    total_samples = 0
    with torch.no_grad():
        for data, target in data_loader:
            if total_samples >= n_samples:
                break

            data = data.to(device)
            z_mean, z_scale = model.encode(data.view(-1, 784))

            # calculate how many samples we still need
            samples_needed = min(n_samples - total_samples, len(data))
            latent_vectors.append(z_mean[:samples_needed])
            labels.append(target[:samples_needed])

            total_samples += samples_needed

    return (
        torch.cat(latent_vectors, dim=0).cpu().detach().numpy(),
        torch.cat(labels, dim=0).cpu().detach().numpy(),
    )


def perform_knn_evaluation(model, train_loader, test_loader):
    X_train, y_train = get_latent_representations(model, train_loader)
    X_test, y_test = get_latent_representations(model, test_loader)

    results = {}
    for n_samples in KNN_EVAL_SAMPLES:
        metric = "cosine" if model.distribution == "power_spherical" else "euclidean"
        knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
        knn.fit(X_train[:n_samples], y_train[:n_samples])
        y_pred = knn.predict(X_test)
        results[n_samples] = accuracy_score(y_test, y_pred)
        # print(f"Accuracy with {n_samples} samples: {results[n_samples]:.2f}")

    return results


def run_experiment(
    model_class,
    h_dim,
    z_dim,
    distribution,
    train_loader,
    val_loader,
    test_loader,
    device,
    n_runs=N_RUNS,
):
    results = {samples: [] for samples in KNN_EVAL_SAMPLES}

    for _ in range(n_runs):
        model = model_class(h_dim=h_dim, z_dim=z_dim, distribution=distribution).to(
            device
        )
        optimizer = Adam(model.parameters(), lr=1e-3)
        run_results = train_and_evaluate(
            model, train_loader, val_loader, test_loader, optimizer, device
        )

        for n_samples, accuracy in run_results.items():
            results[n_samples].append(accuracy)

    return results


def calculate_statistics(results):
    return {
        n_samples: (np.mean(accuracies) * 100, np.std(accuracies) * 100)
        for n_samples, accuracies in results.items()
    }


# run experiments and create results table
results_table = []
for z_dim in Z_DIMS:
    print(f"Running experiments for z_dim = {z_dim}")
    results_s = run_experiment(
        ModelVAE,
        H_DIM,
        z_dim + 1,
        "power_spherical",
        train_loader,
        val_loader,
        test_loader,
        device,
    )
    results_n = run_experiment(
        ModelVAE, H_DIM, z_dim, "normal", train_loader, val_loader, test_loader, device
    )
    stats_n = calculate_statistics(results_n)
    stats_s = calculate_statistics(results_s)

    for n_samples in KNN_EVAL_SAMPLES:
        mean_n, std_n = stats_n[n_samples]
        mean_s, std_s = stats_s[n_samples]

        results_table.append(
            {
                "d": z_dim,
                "n_samples": n_samples,
                "N-VAE": f"{mean_n:.1f}±{std_n:.1f}",
                "PWS-VAE": f"{mean_s:.1f}±{std_s:.1f}",
            }
        )
        # print the row added
        print(results_table[-1])

# create and display results table
df = pd.DataFrame(results_table)
df = df.pivot(index="d", columns="n_samples", values=["N-VAE", "PWS-VAE"])
df.columns.names = ["Method", "n_samples"]
df = df.reindex(
    columns=pd.MultiIndex.from_product([["N-VAE", "PWS-VAE"], KNN_EVAL_SAMPLES])
)

print(df.to_string())
# save as csv
df.to_csv("pws_vae_results.csv")


def highlight_best(s):
    is_best = s == s.max()
    return ["font-weight: bold" if v else "" for v in is_best]


styled_df = df.style.apply(highlight_best, axis=1)
print(styled_df.to_string())
