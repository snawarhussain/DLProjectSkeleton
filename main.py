"""
# --------------------------------------------------------
# @Project: Project Model Training Main File
# @Author : Snawar 
# @E-mail : snawar.hussain18@gmail.com
# @Date   : 2023-05-30 16:26:26
# --------------------------------------------------------
"""
import argparse
from dataclasses import asdict, fields
import json
import os
import torch
from torch.utils.data import DataLoader, random_split
from models.basemodel import BaseModel
from utils.config import TrainingConfig
from utils.logger import get_logger, print_and_log_info
from utils.trainer import Trainer
from utils.project_manager import ProjectManager, ProjectConfig
from dataloader.pinwheel import PinwheelDataset

torch.set_float32_matmul_precision('high')
# Initialize the parser
parser = argparse.ArgumentParser(description='Train a model')
#default project path

project_path = os.getenv('PROJECT_DIR', None)


# Dynamically add command-line arguments based on the fields in ProjectConfig
for field in fields(ProjectConfig):
    field_type = field.type
    if field_type is int:
        parser.add_argument(f'--{field.name}', type=int, default=None, help=f'{field.name}')
    elif field_type is float:
        parser.add_argument(f'--{field.name}', type=float, default=None, help=f'{field.name}')
    elif field_type is str:
        parser.add_argument(f'--{field.name}', type=str, default=None, help=f'{field.name}')
    elif field_type is TrainingConfig:
        for field2 in fields(TrainingConfig):
            field_type2 = field2.type
            if field_type2 is int:
                parser.add_argument(f'--{field2.name}', type=int, default=None, help=f'{field2.name}')
            elif field_type2 is float:
                parser.add_argument(f'--{field2.name}', type=float, default=None, help=f'{field2.name}')
            elif field_type2 is str:
                parser.add_argument(f'--{field2.name}', type=str, default=None, help=f'{field2.name}')


# Parse the command-line arguments
args = parser.parse_args()


# Initialize ProjectConfig dataclass
config = ProjectConfig()

# Initialize ProjectManager
pm = ProjectManager(config=config)


# Load the project configuration if the project directory already exists
if   project_path is not None and os.path.exists(project_path):
    config = pm.load_project(project_path)
else:
    # Initialize the project (creates directories and sets up the project environment)
    config = pm._initialize_project()



logger = get_logger(config, __name__)
print_and_log_info(logger, json.dumps(asdict(config), indent=4, sort_keys=True))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize the dataset and dataloader
print_and_log_info(logger, "device for training: {}".format(device))

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
                  device, config, val_loader, logger)

trainer.train(config.training.epochs)
