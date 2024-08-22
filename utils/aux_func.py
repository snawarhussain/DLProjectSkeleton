import argparse
from dataclasses import dataclass, field, asdict
import os
@dataclass
class ModelConfig:
    block_size: int = 4096 * 2
    vocab_size: int = 512
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    input_dim: int = 2
    hidden_size: int = 256
    num_layers: int = 2
    latent_dim: int = 32
    decoder_type: str = "MLP"
    encoder_type: str = "MLP"

@dataclass
class TrainingConfig:
    epochs: int = 50
    learning_rate: float = 0.001
    batch_size: int = 16
    save_every: int = 10
    optimizer: str = "Adam"
    scheduler: str = "StepLR"

@dataclass
class ProjectConfig:
    project_name: str = "model_training"
    experimenter_name: str = "Bilbo Baggins"
    project_directory: str =  os.path.join(os.getcwd(), "output")
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def update_config_from_args(config: ProjectConfig, args: argparse.Namespace) -> ProjectConfig:
    args_dict = vars(args)
    
    for key, value in args_dict.items():
        if value is not None:
            setattr(config, key, value)
    
    return config