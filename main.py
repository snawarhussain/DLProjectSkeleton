"""
# --------------------------------------------------------
# @Project: Project based Model Train/Eval Pipeline
# @Author : Snawar 
# @E-mail : snawar.hussain18@gmail.com
# @Date   : 2023-05-30 16:26:26
# --------------------------------------------------------
    
"""

import argparse
from models.basemodel import BaseModel
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from dataloader.pinwheel import PinwheelDataset
from utils.aux_func import write_config
from utils.trainer import Trainer
import torch
from utils.project_manager import ProjectManager

#initialize Project
working_dir = 'C:/Users/pc/vae_traing'
pm = ProjectManager(p_name='VAE_new', working_dir=working_dir, experimenter='Snawar')


# Initialize the parser
parser = argparse.ArgumentParser(description='Train a model')

# Add optional command-line arguments
parser.add_argument('--config', type=str, default=None, 
                    help='Path to the configuration file')
parser.add_argument('--epochs', type=int, default=None,
                    help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.005,
                    help='Learning rate')
parser.add_argument('--batch_size', type=int, default=None,
                    help='Batch size')
parser.add_argument('--input_dim', type=int, default=None,
                    help='Input dimension')
parser.add_argument('--hidden_size', type=int, default=None,
                    help='Hidden size')
parser.add_argument('--num_layers', type=int, default=None,
                    help='Number of Hidden layers')
parser.add_argument('--latent_dim', type=int, default=None,
                    help='Latent dimension')
parser.add_argument('--decoder_type', type=str, default=None,
                    help='Decoder type')
parser.add_argument('--encoder_type', type=str, default=None,
                    help='Encoder type')
parser.add_argument('--optimizer', type=str, default=None,
                    help='Optimizer type')
parser.add_argument('--save_every', type=int, default=None,
                    help='Save model every n epochs')
parser.add_argument('--scheduler', type=str, default=None,
                    help='Scheduler type')

# Parse the command-line arguments
args = parser.parse_args()

if args.config is not None:
    config_path = args.config
    config = pm.load_project(config_path) 
else:
    # initialize project returns the path to the template config file
    config_path = pm._initialize_project()    
    # Load the config file
    config = pm.cfg(config_path)

# If command-line arguments are specified, they override the config file
params = ['epochs', 'learning_rate', 'batch_size', 'hidden_size', 'num_layers', 'latent_dim',
          'decoder_type', 'encoder_type', 'input_dim', 'optimizer', 'save_every', 'scheduler']

for param in params:
    arg_value = getattr(args, param)
    if arg_value is not None:
        config['Model'][param] = arg_value

  
# populate the config file with newer arguments
write_config(config_path,config)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize the dataset and dataloader

dataset = PinwheelDataset(0.3, 0.05, 5, 300, 0.25)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
trainloader = DataLoader(train_dataset, batch_size=config['Model']['batch_size'], shuffle=True)
valloader = DataLoader(val_dataset, batch_size=config['Model']['batch_size'], shuffle=False)

model = BaseModel(config)
# print(model)
if config['Model']['optimizer'] == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=config['Model']['learning_rate'])
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=config['Model']['learning_rate'])

trainer = Trainer(model, trainloader, valloader, optimizer, device, config)

trainer.train(config['Model']['epochs'])