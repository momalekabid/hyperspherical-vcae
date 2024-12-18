import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch.functional as F
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
    parser.add_argument('--dataset', type=str, default='fashionmnist', 
                      choices=['mnist', 'fashionmnist', 'cifar10'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--distribution', type=str, default='powerspherical', 
                      choices=['gaussian', 'powerspherical'])
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--latent_dims', type=int, nargs='+', 
                      default=[64, 128, 256, 512, 1024, 2048, 3072, 4096])
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--knn_samples', type=int, default=1000, help='Number of test samples to use for KNN evaluation')
    parser.add_argument('--n_neighbors', type=int, default=5, help='Number of neighbors for KNN')
    parser.add_argument('--knn_runs', type=int, default=20, help='Number of KNN evaluation runs')
    parser.add_argument('--graph', action='store_true', 
                      help='Enable comparative plotting')
    return parser.parse_args()


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
    # encode training set
    train_latent, train_labels = encode_dataset(model, train_loader, device)
    
    # encode full test set
    test_latent, test_labels = encode_dataset(model, test_loader, device)
    
    # initialize metrics storage
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
        m = "cosine" if model.distribution == "powerspherical" else "euclidean"
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=m)
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

        metrics['beta_values'].append(args.beta)
        metrics['gamma_values'].append(args.gamma)
        # print("training") 
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar, q_z, p_z = model(data)
            loss, recon_loss, kl_loss, unitary_loss = model.compute_loss(
                data, recon_batch, mu, logvar, q_z, p_z, args.beta, args.gamma
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
                    data, recon_batch, mu, logvar, q_z, p_z, args.beta, args.gamma
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
    model_args = {
        'latent_dim': model.latent_dim,
        'distribution': model.distribution,
        'beta': args.beta,
        'gamma': args.gamma
    }
    knn_metrics = evaluate_knn(
        model, train_loader, test_loader,
        args.knn_samples, args.n_neighbors, args.knn_runs,
        device=device, save_dir=save_dir
    )
    metrics['knn_metrics'] = knn_metrics
    
    return metrics

def run_experiment(args, latent_dim=None):
    """
    Run a single experiment with given parameters
    """
    train_loader, test_loader, in_channels = get_dataset(args.dataset, args.batch_size)
    
    # use provided latent_dim if specified, otherwise use first from args.latent_dims
    latent_dim = latent_dim if latent_dim is not None else args.latent_dims[0]
    
    # create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{timestamp}_latent{latent_dim}_{args.distribution}_beta{args.beta}"
    exp_dir = Path(args.output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # initialize model and optimizer
    model = VAE(latent_dim=latent_dim, in_channels=in_channels, distribution=args.distribution)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    
    # train model and get metrics
    metrics = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        args=args,
        save_dir=exp_dir,
        dn=args.dataset
    )
    
    # save metrics and model
    with open(exp_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    torch.save(model.state_dict(), exp_dir / 'model.pth')
    
    return metrics

def main():
    args = parse_args()
    base_output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.graph:
        results_dict = {
            ('gaussian', 1.0): {'latent_dims': [], 'accuracy_means': [], 
                              'accuracy_stds': [], 'f1_means': [], 'f1_stds': []},
            ('gaussian', 0.1): {'latent_dims': [], 'accuracy_means': [], 
                              'accuracy_stds': [], 'f1_means': [], 'f1_stds': []},
            ('powerspherical', 1.0): {'latent_dims': [], 'accuracy_means': [], 
                                    'accuracy_stds': [], 'f1_means': [], 'f1_stds': []},
            ('powerspherical', 0.1): {'latent_dims': [], 'accuracy_means': [], 
                                    'accuracy_stds': [], 'f1_means': [], 'f1_stds': []}
        }
        
        for distribution in ['powerspherical','gaussian']: # TBD 
            for beta in [1.0, 0.1]:
                args.distribution = distribution
                args.beta = beta
                # if distribution == 'gaussian' and beta != 1.0:
                #     continue
                # elif distribution == 'powerspherical' and beta == 1.0:
                #     continue
                
                for latent_dim in args.latent_dims:
                    run_accuracies = []
                    run_f1_scores = []
                    
                    for run in range(args.knn_runs):
                        print(f"Running {distribution}, beta={beta}, dim={latent_dim}, run {run+1}/{args.knn_runs}")
                        metrics = run_experiment(args, latent_dim)
                        run_accuracies.append(metrics['knn_metrics']['accuracy_mean'])
                        run_f1_scores.append(metrics['knn_metrics']['f1_mean'])
                    
                    # calc statistics across runs
                    results_dict[(distribution, beta)]['latent_dims'].append(latent_dim)
                    results_dict[(distribution, beta)]['accuracy_means'].append(np.mean(run_accuracies))
                    results_dict[(distribution, beta)]['accuracy_stds'].append(np.std(run_accuracies))
                    results_dict[(distribution, beta)]['f1_means'].append(np.mean(run_f1_scores))
                    results_dict[(distribution, beta)]['f1_stds'].append(np.std(run_f1_scores))
        
        plot_path = base_output_dir / f"{timestamp}_comparative_results.png"
        plot_comparative_results(results_dict, plot_path)
    
    else:
        run_experiment(args)

def plot_comparative_results(results_dict, save_path):
    """
    Plot comparative results for different model configurations with improved x-axis labeling
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    styles = {
        ('gaussian', 1.0): ('orange', '--', 'Gaussian (β=1)'),
        ('gaussian', 0.1): ('orange', '-', 'Gaussian (β=0.1)'),
        ('powerspherical', 1.0): ('blue', '--', 'Power Spherical (β=1)'),
        ('powerspherical', 0.1): ('blue', '-', 'Power Spherical (β=0.1)')
    }
    
    for (dist, beta), metrics in results_dict.items():
        color, style, label = styles[(dist, beta)]
        
        # plot accuracy with error bars representing std across runs
        ax1.errorbar(
            metrics['latent_dims'],
            metrics['accuracy_means'],
            yerr=metrics['accuracy_stds'],
            label=f"{label}\n(std across runs)",
            color=color,
            linestyle=style,
            capsize=5,
            marker='o'
        )
        
        # plot F1 score with error bars representing std across runs
        ax2.errorbar(
            metrics['latent_dims'],
            metrics['f1_means'],
            yerr=metrics['f1_stds'],
            label=f"{label}\n(std across runs)",
            color=color,
            linestyle=style,
            capsize=5,
            marker='o'
        )
    
    # configure plots
    for ax in [ax1, ax2]:
        ax.set_xscale('log', base=2)  # use log base 2 for more intuitive spacing
        ax.grid(True, which="both", linestyle='--', alpha=0.7)
        ax.set_xlabel('Latent Dimension')
        
        # get the first set of latent dimensions (they're the same for all configurations)
        latent_dims = next(iter(results_dict.values()))['latent_dims']
        
        # set specific x-ticks at the actual latent dimension values
        ax.set_xticks(latent_dims)
        ax.set_xticklabels(latent_dims, rotation=45)
        
        # adjust y-axis limits to start from 0.3 and end at 0.9
        ax.set_ylim(0.3, 0.9)
    
    ax1.set_ylabel('Accuracy')
    ax2.set_ylabel('F1 Score')
    
    ax1.set_title('Accuracy vs Latent Dimension\n(with std across runs)')
    ax2.set_title('F1 Score vs Latent Dimension\n(with std across runs)')
    
    # adjust legend position and style
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()