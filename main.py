import argparse
from dataclasses import fields
import os
import torch
from torch.utils.data import DataLoader, random_split
from models.basemodel import BaseModel
from utils.trainer import Trainer
from utils.project_manager import ProjectManager, ProjectConfig
from dataloader.pinwheel import PinwheelDataset

torch.set_float32_matmul_precision('high')
# Initialize the parser
parser = argparse.ArgumentParser(description='Train a model')

# Dynamically add command-line arguments based on the fields in ProjectConfig
for field in fields(ProjectConfig):
    field_type = field.type
    if field_type == int:
        parser.add_argument(f'--{field.name}', type=int, default=None, help=f'{field.name}')
    elif field_type == float:
        parser.add_argument(f'--{field.name}', type=float, default=None, help=f'{field.name}')
    elif field_type == str:
        parser.add_argument(f'--{field.name}', type=str, default=None, help=f'{field.name}')

# Parse the command-line arguments
args = parser.parse_args()


# Initialize ProjectConfig dataclass
config = ProjectConfig()

# Initialize ProjectManager
pm = ProjectManager(config=config)

# Initialize the project (creates directories and sets up the project environment)
project_path = pm._initialize_project()


# Load the project configuration if the project directory already exists
if os.path.exists(project_path):
    config = pm.load_project(project_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(type(config))
# Initialize the dataset and dataloader
dataset = PinwheelDataset(0.3, 0.05, 5, 300, 0.25)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)

model = BaseModel(config)
# if it is not windows, compile the model
if os.name != 'nt':
    model = torch.compile(model, backend="inductor")
if config.training.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=config.training.learning_rate)

trainer = Trainer(model, train_loader, optimizer, 
                  device, config, val_loader)

trainer.train(config.training.epochs)
