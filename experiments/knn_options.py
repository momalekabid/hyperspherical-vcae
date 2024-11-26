import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import random
from models.vcae import VAE

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='Train VAE model')
    parser.add_argument('--dataset', type=str, default='fashionmnist', choices=['mnist', 'fashionmnist', 'cifar10'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--distribution', type=str, default='powerspherical', choices=['gaussian', 'powerspherical'])

    parser.add_argument('--schedule', type=str, default='cosine', choices=['constant', 'linear', 'cosine']) 
    # Beta annealing parameters
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--use_beta_cycling', action='store_true', help='Enable beta cyclical annealing')
    parser.add_argument('--beta_cycles', type=int, default=4, help='Number of beta cycles during training')
    
    # Gamma annealing parameters
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--use_gamma_cycling', action='store_true', help='Enable gamma cyclical annealing')
    parser.add_argument('--gamma_cycles', type=int, default=4, help='Number of gamma cycles during training')
    
    # KNN evaluation parameters
    parser.add_argument('--knn_eval', action='store_true', help='Enable KNN evaluation')
    parser.add_argument('--knn_samples', type=int, default=1000, help='Number of test samples to use for KNN evaluation')
    parser.add_argument('--n_neighbors', type=int, default=5, help='Number of neighbors for KNN')
    parser.add_argument('--knn_runs', type=int, default=5, help='Number of KNN evaluation runs')
    
    parser.add_argument('--latent_dims', type=int, nargs='+', default=[32])
    parser.add_argument('--output_dir', type=str, default='results')
    return parser.parse_args()

def get_schedule_value(epoch, total_epochs, n_cycles=1, schedule_type="cosine"):
    t = (epoch % (total_epochs // n_cycles)) / (total_epochs // n_cycles)
    if schedule_type == "cosine":
        return 0.5 * (1 + np.cos(np.pi * (1 - t)))
    elif schedule_type == "linear":
        return t
    return 1.0

def encode_dataset(model, dataloader, device):
    """Encode entire dataset and return latent representations with labels."""
    model.eval()
    latent_vecs = []
    labels = []
    
    with torch.no_grad():
        for data, target in dataloader:
            x = data.to(device)
            mu, _ = model.encoder(x)
            latent_vecs.append(mu.cpu().numpy())
            labels.append(target.numpy())
    
    return np.concatenate(latent_vecs), np.concatenate(labels)

def evaluate_knn(model, train_loader, test_loader, n_samples, n_neighbors, n_runs, device, save_dir):
    """
    Evaluate classification performance using KNN in the latent space.
    Runs multiple times with different random test samples and returns mean/std metrics.
    """
    # Encode training set
    train_latent, train_labels = encode_dataset(model, train_loader, device)
    
    # Encode full test set
    test_latent, test_labels = encode_dataset(model, test_loader, device)
    
    # Initialize metrics storage
    accuracies = []
    f1_scores = []
    
    # run KNN evaluation multiple times with different random samples
    for run in range(n_runs):
        # randomly sample test indices
        if n_samples < len(test_labels):
            test_indices = random.sample(range(len(test_labels)), n_samples)
            sampled_test_latent = test_latent[test_indices]
            sampled_test_labels = test_labels[test_indices]
        else:
            sampled_test_latent = test_latent
            sampled_test_labels = test_labels
        
        # train and evaluate KNN
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(train_latent, train_labels)
        predictions = knn.predict(sampled_test_latent)
        
        # calculate metrics
        accuracy = accuracy_score(sampled_test_labels, predictions)
        f1 = f1_score(sampled_test_labels, predictions, average='weighted')
        
        accuracies.append(accuracy)
        f1_scores.append(f1)
    
    # calculate statistics
    knn_metrics = {
        'accuracy_mean': float(np.mean(accuracies)),
        'accuracy_std': float(np.std(accuracies)),
        'f1_mean': float(np.mean(f1_scores)),
        'f1_std': float(np.std(f1_scores)),
        'n_samples': n_samples,
        'n_neighbors': n_neighbors,
        'n_runs': n_runs
    }
    
    # save KNN metrics
    with open(save_dir / 'knn_metrics.json', 'w') as f:
        json.dump(knn_metrics, f, indent=4)
    
    return knn_metrics

    
# def get_cyclical_constant(epoch, total_epochs, n_cycles):
#     """
#     Implementation of the cyclical beta schedule as described in the paper.
#     Each cycle consists of:
#     1. Quick ramp up from 0 to 1
#     2. Maintain at beta=1 for remainder of cycle
    
#     Args:
#         epoch: Current epoch
#         total_epochs: Total number of epochs
#         n_cycles: Number of cycles to complete
    
#     Returns:
#         Current beta value
#     """
#     cycle_length = total_epochs // n_cycles
#     current_cycle = epoch // cycle_length
#     cycle_position = epoch % cycle_length
    
#     # Determine ramp-up period length (make it relatively quick, e.g. 10% of cycle)
#     ramp_up_length = cycle_length // 10
    
#     if cycle_position < ramp_up_length:
#         # In ramp-up phase: linear increase from 0 to 1
#         return cycle_position / ramp_up_length
#     else:
#         # Maintain beta=1 for remainder of cycle
#         return 1.0


def get_dataset(dataset_name, batch_size):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])
    
    dataset_map = {
        'fashionmnist': (datasets.FashionMNIST, 1),
        'cifar10': (datasets.CIFAR10, 3)
    }
    
    dataset_class, in_channels = dataset_map[dataset_name]
    
    train_dataset = dataset_class(root='./data', train=True, download=True, transform=transform)
    test_dataset = dataset_class(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, in_channels

def save_reconstructions(model, test_loader, epoch, save_path, dn):
    model.eval()
    with torch.no_grad():
        data = next(iter(test_loader))
        x = data[0][:8].to(_DEVICE) # your device
        recon_x, _, _, _, _ = model(x)
        
        # create comparison plot
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        for i in range(8):
            img = x[i].cpu().squeeze().numpy()
            if dn =='fashionmnist':
                axes[0, i].imshow(x[i].cpu().squeeze(), cmap='gray')
                axes[0, i].axis('off')
                axes[1, i].imshow(recon_x[i].cpu().squeeze(), cmap='gray')
                axes[1, i].axis('off')
            else: # cifar10/color 3 channels
             axes[0, i].imshow(x[i].cpu().permute(1, 2, 0)) # (C, H, W) -> (H, W, C)
             axes[0, i].axis('off')
             axes[1, i].imshow(recon_x[i].cpu().permute(1, 2, 0))
             axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/reconstructions_epoch_{epoch}.png")
        plt.close()

def visualize_latent_space(model, test_loader, save_path):
    model.eval()
    latent_vecs = []
    labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            x = data.to(_DEVICE) # your device
            mu, _ = model.encoder(x)
            latent_vecs.append(mu.cpu().numpy())
            labels.append(target.numpy())
    
    latent_vecs = np.concatenate(latent_vecs)
    labels = np.concatenate(labels)

    fft_magnitudes = []
    for vec in latent_vecs:
        fft_magnitudes.append(np.abs(np.fft.fftshift(np.fft.fft(vec))))
    fft_magnitudesnp = np.array(fft_magnitudes)
    print(f"Mean: {np.mean(fft_magnitudesnp)}")
    print(f"Variance: {np.var(fft_magnitudesnp)}")
    print(f"Min: {np.min(fft_magnitudesnp)}")
    print(f"Max: {np.max(fft_magnitudesnp)}")
    avg_magnitude = np.mean(fft_magnitudes, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(avg_magnitude)
    plt.title("Average Magnitude Spectrum of Encoded Vectors")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.savefig(f"{save_path}/average_magnitude_spectrum.png")
    plt.close()
    
    # compute t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_vecs)
    
    # plot
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title('t-SNE visualization of latent space')
    plt.savefig(f"{save_path}/latent_space_tsne.png")
    plt.close()

def train_model(model, train_loader, test_loader, optimizer, args, save_dir, dn):
    device = _DEVICE
    model = model.to(device)
    
    metrics = {
        'train_loss': [],
        'test_loss': [],
        'recon_loss': [],
        'kl_loss': [],
        'unitary_loss': [],
        'beta_values': [],
        'gamma_values': []
    }

    for epoch in range(args.epochs):
        # calculate current beta and gamma values
        current_beta = args.beta * get_schedule_value(
            epoch, args.epochs, 
            args.beta_cycles if args.use_beta_cycling else 1,
            args.schedule if args.use_beta_cycling else "constant"
        )
        
        current_gamma = args.gamma * get_schedule_value(
            epoch, args.epochs,
            args.gamma_cycles if args.use_gamma_cycling else 1,
            args.schedule if args.use_gamma_cycling else "constant"
        )
        
        metrics['beta_values'].append(current_beta)
        metrics['gamma_values'].append(current_gamma)
        
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar, q_z, p_z = model(data)
            loss, recon_loss, kl_loss, unitary_loss = model.compute_loss(
                data, recon_batch, mu, logvar, q_z, p_z, current_beta, current_gamma
            )
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                recon_batch, mu, logvar, q_z, p_z = model(data)
                loss, _, _, _ = model.compute_loss(
                    data, recon_batch, mu, logvar, q_z, p_z, current_beta, current_gamma
                )
                test_loss += loss.item()
        
        metrics['train_loss'].append(train_loss / len(train_loader.dataset))
        metrics['test_loss'].append(test_loss / len(test_loader.dataset))
        metrics['recon_loss'].append(recon_loss.item())
        metrics['kl_loss'].append(kl_loss.item())
        metrics['unitary_loss'].append(unitary_loss.item())
        
        # save reconstructions every ten epochs
        if epoch % 10 == 0:
            save_reconstructions(model, test_loader, epoch, save_dir, dn)
            
    # save final visualizations and reconstructions
    save_reconstructions(model, test_loader, args.epochs-1, save_dir, dn)
    visualize_latent_space(model, test_loader, save_dir)
    
    # Perform KNN evaluation if enabled
    if args.knn_eval:
        knn_metrics = evaluate_knn(
            model, train_loader, test_loader,
            args.knn_samples, args.n_neighbors, args.knn_runs,
            device, save_dir
        )
        metrics['knn_metrics'] = knn_metrics
    
    return metrics

def main():
    args = parse_args()
    base_output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    train_loader, test_loader, in_channels = get_dataset(args.dataset, args.batch_size)
    
    results = []
    
    for latent_dim in args.latent_dims:
        # create experiment directory with additional KNN info if enabled
        exp_name = f"{timestamp}_latent{latent_dim}"
        if args.use_beta_cycling:
            exp_name += f"_betacycle{args.beta_cycles}"
        if args.use_gamma_cycling:
            exp_name += f"_gammacycle{args.gamma_cycles}"
        if args.knn_eval:
            exp_name += f"_knn{args.knn_samples}"
        
        exp_dir = base_output_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # initialize model and optimizer
        model = VAE(latent_dim=latent_dim, in_channels=in_channels, distribution=args.distribution)
        optimizer = optim.Adam(model.parameters(), lr=3e-4)
        
        metrics = train_model(
            model, train_loader, test_loader, optimizer, 
            args, exp_dir, dn=args.dataset
        )
        
        result_dict = {
            'latent_dim': latent_dim,
            'beta': args.beta,
            'gamma': args.gamma,
            'beta_cycling': args.use_beta_cycling,
            'gamma_cycling': args.use_gamma_cycling,
            'final_train_loss': metrics['train_loss'][-1],
            'final_test_loss': metrics['test_loss'][-1],
            'final_recon_loss': metrics['recon_loss'][-1],
            'final_kl_loss': metrics['kl_loss'][-1],
            'final_unitary_loss': metrics['unitary_loss'][-1],
        }
        
        # Add KNN metrics to results if enabled
        if args.knn_eval:
            result_dict.update({
                'knn_accuracy_mean': metrics['knn_metrics']['accuracy_mean'],
                'knn_accuracy_std': metrics['knn_metrics']['accuracy_std'],
                'knn_f1_mean': metrics['knn_metrics']['f1_mean'],
                'knn_f1_std': metrics['knn_metrics']['f1_std']
            })
        
        results.append(result_dict)
        
        # save metrics
        with open(exp_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        torch.save(model.state_dict(), exp_dir / 'model.pth')

if __name__ == '__main__':
    main()