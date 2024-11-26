import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from models.vcae import VAE
### TBD: add KNN evaluation, log f1 score, accuracy, std over n_runs
### TBD: multiple beta values?
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
def parse_args():
    parser = argparse.ArgumentParser(description='Train VAE model')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashionmnist', 'cifar10'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--distribution', type=str, default='powerspherical', choices=['gaussian', 'powerspherical'])
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--latent_dims', type=int, nargs='+', default=[32])# 64, 128, 256, 512, 1024])
    parser.add_argument('--gammas', type=float, nargs='+', default=[0]) #0.3, 0.3, 0.5, 0.5, 0.7])
    parser.add_argument('--output_dir', type=str, default='results')
    return parser.parse_args()

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

def save_reconstructions(model, test_loader, epoch, save_path, dataset='fashionmnist'):
    model.eval()
    with torch.no_grad():
        data = next(iter(test_loader))
        x = data[0][:8].to(_DEVICE) # your device
        recon_x, _, _, _, _ = model(x)
        
        # create comparison plot
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        for i in range(8):
            # img = x[i].cpu().squeeze().numpy()
            # if img.shape[0] == 1:
            # if dataset == 'fashionmnist':
            #     axes[0, i].imshow(x[i].cpu().squeeze(), cmap='gray')
            #     axes[0, i].axis('off')
            #     axes[1, i].imshow(recon_x[i].cpu().squeeze(), cmap='gray')
            #     axes[1, i].axis('off')
            # else:
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

def train_model(model, train_loader, test_loader, optimizer, epochs, beta, gamma, save_dir, dn='fashionmnist'):
    # device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device = _DEVICE
    model = model.to(device)
    
    metrics = {
        'train_loss': [],
        'test_loss': [],
        'recon_loss': [],
        'kl_loss': [],
        'unitary_loss': []
    }
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar, q_z, p_z = model(data)
            loss, recon_loss, kl_loss, unitary_loss = model.compute_loss(
                data, recon_batch, mu, logvar, q_z, p_z, beta, gamma
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
                    data, recon_batch, mu, logvar, q_z, p_z, beta, gamma
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
    save_reconstructions(model, test_loader, epoch, save_dir)
    visualize_latent_space(model, test_loader, save_dir)
    
    return metrics

def main():
    args = parse_args()
    base_output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    train_loader, test_loader, in_channels = get_dataset(args.dataset, args.batch_size)
    
    # ensure matching number of gammas and latent dims
    if len(args.gammas) != len(args.latent_dims):
        raise ValueError("Number of gamma values must match number of latent dimensions")
    
    results = []
    
    for latent_dim, gamma in zip(args.latent_dims, args.gammas):
        # create experiment directory
        exp_dir = base_output_dir / f"{timestamp}_latent{latent_dim}_gamma{gamma}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # initialize model and optimizer
        model = VAE(latent_dim=latent_dim, in_channels=in_channels, distribution=args.distribution)
        optimizer = optim.Adam(model.parameters(), lr=3e-4)
        
        metrics = train_model(
            model, train_loader, test_loader, optimizer, 
            args.epochs, args.beta, gamma, exp_dir, args.dataset
        )
        
        results.append({
            'latent_dim': latent_dim,
            'gamma': gamma,
            'beta': args.beta,
            'final_train_loss': metrics['train_loss'][-1],
            'final_test_loss': metrics['test_loss'][-1],
            'final_recon_loss': metrics['recon_loss'][-1],
            'final_kl_loss': metrics['kl_loss'][-1],
            'final_unitary_loss': metrics['unitary_loss'][-1]
        })
        
        # save metrics
        with open(exp_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f)
        
        torch.save(model.state_dict(), exp_dir / 'model.pth')
    

if __name__ == '__main__':
    main()