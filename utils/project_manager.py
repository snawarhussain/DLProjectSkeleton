import logging
import os
from pathlib import Path
from datetime import datetime as dt
from typing import Union
import yaml
from dataclasses import asdict, dataclass

from utils.aux_func import ModelConfig, ProjectConfig, TrainingConfig



class ProjectManager:
    def __init__(self, config: ProjectConfig = ProjectConfig()):
        super().__init__()

        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def _initialize_project(self):
        # Initialize the logger
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        date = dt.today().strftime('%Y-%m-%d')

        if self.config.project_directory == ".":
            self.config.project_directory = str(Path('.').resolve())
                
        work_dir = Path(self.config.project_directory).resolve()
        project_name = f"{self.config.project_name}--{self.config.experimenter_name}--{date}"
        
        # Create directories for the project
        project_path = Path(os.path.join(work_dir, project_name))
        if project_path.exists():
            print(f'project {project_path} already exists!')
            self.logger.info(f'project {project_path} already exists!')
            return project_path

        model_path = project_path / 'model'
        results_path = project_path / 'results'
        tb_logger = project_path / 'tensorboard'
        best_model_path = model_path / 'best_model'
        plot_path = project_path / 'plots'
        
        for p in [model_path, results_path, tb_logger, best_model_path, plot_path]:
            p.mkdir(parents=True)
            print(f'created {p} directory')
            self.logger.info(f"created {p} directory")
        
        self.config.project_directory = str(project_path)
        
        self.save_config()
        
        self.logger.info(f"Initialized a project with name: {project_name}")
        return " "

    def save_config(self):
        config_path = os.path.join(self.config.project_directory, 'config.yaml')
        with open(config_path, 'w') as file:
            yaml.dump(asdict(self.config), file)
        self.logger.info(f"Configuration saved to {config_path}")
    
    def load_project(self, project_path: Union[str, Path]) -> ProjectConfig:
        """
        Load a project configuration from the existing project directory.

        Returns:
            ProjectConfig: Config dataclass instance
        """
        config_path = os.path.join(project_path, 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config_dict = yaml.safe_load(file)
                self.config = ProjectConfig(
                    project_name=config_dict.get("project_name", "VAE_new"),
                    experimenter_name=config_dict.get("experimenter_name", "Snawar"),
                    project_directory=config_dict.get("project_directory", "./output"),
                    model=ModelConfig(**config_dict.get("model", {})),
                    training=TrainingConfig(**config_dict.get("training", {})),
                )
                self.logger.info(f"Loaded configuration from {config_path}")
        else:
            self.logger.warning(f"Configuration file not found at {config_path}, using default configuration")
        return self.config      
    
    def get_config(self) -> ProjectConfig:
        """
        Get the current configuration.

        Returns:
            ProjectConfig: Config dataclass instance
        """
        return self.config
