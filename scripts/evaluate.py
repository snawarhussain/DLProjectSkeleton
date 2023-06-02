import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from dataloader.pinwheel import PinwheelDataset
from torch.utils.data import DataLoader
from models.basemodel import BaseModel
import matplotlib.pyplot as plt
import numpy as np
from utils.project_manager import ProjectManager
from glob import glob
    

class EvaluateNetwork():
    """
    Evaluate the trained network
    
    """
    def __init__(self, config, data_loader):
        self.config = config
        self.model_path = Path(config['Project']['project_directory']) / 'model' / 'best_model/best_model.pth'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BaseModel(self.config).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        self.criterion = nn.MSELoss()
        self.data_loader = data_loader
    
    def compute_resonstruction_loss(self):
        """
        compute the MSE Loss between the real data and
        the reconstructed data

        Returns:
            torch.Tensor: Avg Loss over the dataset
        """
        total_loss = 0
        with torch.no_grad():
            for data, _ in self.data_loader:
                data = data.to(self.device)
                x_hat, mu, logvar = self.model(data)
                loss = self.criterion(x_hat, data)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.data_loader)
        print(f"Average loss: {avg_loss}")
        return avg_loss
    
    def generate_samples(self):
        """
        generate samples from the latent space
        using the decoder. sample z from a normal distribution

        Returns:
            torch.Tensor: Tensor of shape (num_samples, latent_dim)
        """
        z = torch.randn(1000, self.config['Model']['latent_dim']).to(self.device)
        sample = self.model.decoder['layers'][0](z)
        for layer in self.model.decoder['layers'][1:]:
            sample = layer(sample)
        sample = self.model.decoder['final'](sample)
        return sample
    
    def plot_samples(self):
        """
        plot the samples generated from the latent space
        """
        sample = self.generate_samples()
        sample = sample.cpu().detach().numpy()
        plt.scatter(sample[:,0], sample[:,1], s=20)
        plt.xticks([])
        plt.yticks([])
        plt.title('Samples Generated from latent space')
        plt.savefig(Path(self.config['Project']['project_directory']) / 'plots' / 'samples.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_embedding(self):
        """
        get the latent space embeddings for the dataset

        Returns:
            np.ndarray: latent space embeddings
            np.ndarray: labels
        """
        
        latent_space = []
        labels = []
        
        with torch.no_grad():
            for data, label in self.data_loader:
                data = data.to(self.device)
                _, mu, _ = self.model(data)
                latent_space.append(mu.cpu().detach().numpy())
                labels.append(label.numpy())
        latent_space = np.vstack(latent_space)
        labels = np.hstack(labels)
        np.save(Path(self.config['Project']['project_directory']) / 'results' / 'latent_space_embeddings.npy', latent_space)
        np.save(Path(self.config['Project']['project_directory']) / 'results' / 'latent_space_labels.npy', labels)
        return latent_space, labels
    
    def vizualize_latent_space(self):
        """
        vizualize the latent space using t-SNE
        
        """
        latent_space, labels = self.get_embedding()
        # Using t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=0)
        mu_2d = tsne.fit_transform(latent_space)
        plt.scatter(mu_2d[:,0], mu_2d[:,1], c=labels, s=20, cmap=plt.cm.Spectral)
        plt.xticks([])
        plt.yticks([])
        plt.title('t-SNE visualization of latent space')
        plt.savefig(Path(self.config['Project']['project_directory']) / 'plots' / 'latent_space.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def show_saved_plots(self):
        """
        read and show the saved plots from the project plot directory
        
        """
        #list files that have .png extention
        plots= glob(self.config['Project']['project_directory'] + '/plots/*.png')  
        plt.figure(figsize=(10,10))
        plt.subplots_adjust(hspace=1)
        for i, plot in enumerate(plots):
            plt.subplot(1,2,i+1)
            plt.imshow(plt.imread(Path(self.config['Project']['project_directory']) / 'results' / plot))
            plt.xticks([])
            plt.yticks([])
        plt.show()

if __name__=='__main__':
     # Initialize the parser
    parser = argparse.ArgumentParser(description='Evaluate model')

    # Add optional command-line arguments
    parser.add_argument('--config', type=str, default= "C:/Users/pc/vae_traing/VAE_new--Snawar--2023-06-02/config.yaml", 
                        help='Path to the configuration file')
    # Parse the command-line arguments
    args = parser.parse_args()
    pm = ProjectManager()
    config_path = args.config
    config = pm.load_project(config_path)
    dataset = PinwheelDataset(0.3, 0.05, 5, 300, 0.25)
    dataloader = DataLoader(dataset, batch_size=config['Model']['batch_size'], shuffle=True)
    _eval = EvaluateNetwork(config, dataloader)
    _eval.compute_resonstruction_loss()
    _eval.vizualize_latent_space()
    _eval.plot_samples()
    _eval.show_saved_plots()