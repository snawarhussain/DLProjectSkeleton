import torch 
from torch import nn
from torch.nn import functional as F
import logging
from utils.aux_func import ProjectConfig
# Initialize the logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class Block(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.ln2 = nn.LayerNorm(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.ln2(self.linear(x)))

class BaseModel(nn.Module):
    """
    BaseModel Class for implementing a VAE.

    Args:
        nn (_type_): _description_
    """
    def __init__(self, config:ProjectConfig):
        super().__init__()

        self.hidden_dim = config.model.hidden_size
        self.latent_dim = config.model.latent_dim
        self.input_dim = config.model.input_dim
        self.num_layers = config.model.num_layers
        self.encoder = nn.ModuleDict({
            'layers': nn.ModuleList([Block(self.input_dim if i == 0 else self.hidden_dim, 
                                              self.hidden_dim) for i in range(self.num_layers)]),
            'final': nn.Linear(self.hidden_dim, self.latent_dim * 2)  # For mu and logvar
        })

        self.decoder = nn.ModuleDict({
            'layers': nn.ModuleList([Block(self.latent_dim if i == 0 else self.hidden_dim, 
                                              self.hidden_dim) for i in range(self.num_layers)]),
            'final': nn.Linear(self.hidden_dim, self.input_dim)
        })

        logger.info(
            f"Initialized a base model with hidden size: {self.hidden_dim}, "
            f"latent dimension: {self.latent_dim}, "
            # f"and learning rate: {self.learning_rate}"
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        # Encoder
        for layer in self.encoder['layers']:
            x = layer(x)
        h = self.encoder['final'](x)
        mu, logvar = torch.split(h, h.size(-1) // 2, dim=-1)

        # Reparameterization trick
        z = self.reparameterize(mu, logvar)

        # Decoder
        for layer in self.decoder['layers']:
            z = layer(z)
        out = self.decoder['final'](z)

        return out, mu, logvar
    @staticmethod
    def loss_function(x, recon_x, mu, logvar):
        """calculatiing the BCE reconstruction loss and the KL divergence loss
         minimizing the reconstrution loss is similar to maximizing the likelyhood of p(x|z)
         ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))

        Args:
            x (_type_): _description_
            recon_x (_type_): _description_
            mu (_type_): _description_
            logvar (_type_): _description_

        Returns:
            _type_: _description_
        """
        MSE = F.mse_loss(recon_x, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = MSE + KLD
        return loss
