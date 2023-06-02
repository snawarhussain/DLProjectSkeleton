# create a project manager class to initiazlize the project and load it from the config file
import logging
import os
from pathlib import Path
import yaml
from datetime import datetime as dt

from utils.aux_func import create_config_template, read_config, write_config

class ProjectManager:
    def __init__(self, p_name='project', working_dir=None, experimenter='experimenter'):
        super().__init__()

        self.working_dir = working_dir
        self.experimenter = experimenter
        self.p_name = p_name
        self.logger = logging.getLogger(__name__)
    
    def _initialize_project(self):
        # Initialize the logger
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        date = dt.today().strftime('%Y-%m-%d')

        if self.working_dir == None:
            self.working_dir = '.'
                
        work_dir = Path(self.working_dir).resolve()
        project_name = "{pn}--{experimenter}--{date}".format(pn=self.p_name,  experimenter=self.experimenter, date=date)
        
        #create a config file for the project and create directories for the project like deeplabcut
        project_path = Path(os.path.join(work_dir, project_name))
        if project_path.exists():
            print(f'project {project_path} already exists !')
            self.logger.info(f'project {project_path} already exists !')
            projconfigfile=os.path.join(str(project_path),'config.yaml')
            return projconfigfile
        model_path = project_path / 'model'
        results_path = project_path / 'results'
        tb_logger = project_path / 'tensorboard'
        best_model_path = model_path / 'best_model'
        plot_path = project_path / 'plots'
        
        for p in [model_path, results_path, tb_logger, best_model_path, plot_path]:
            p.mkdir(parents=True)
            print (f'created{p} directory')
            self.logger.info(f"created{p} directory")
        
        cfg_file,ruamelFile = create_config_template()
        
        cfg_file['Project']['project_name'] = project_name
        cfg_file['Project']['experimenter_name'] = self.experimenter
        cfg_file['Project']['project_directory'] = str(project_path)
        
        
        
        self.logger.info(f"Initialized a project with config file: {cfg_file['Project']['config']}")
        projconfigfile=os.path.join(str(project_path),'config.yaml')
        write_config(projconfigfile,cfg_file)
        return projconfigfile     
    
    def load_project(self, configname):
        """
        Load a project from a config file.

        Args:
            configname (str): path to the config file

        Returns:
            yaml: config file
        """
        try:
            cfg = read_config(configname)
        except TypeError:
            cfg = {}
        return cfg
    
    @staticmethod
    def cfg(name='config.yaml'):
        """
        Get config file

        Returns:
            yaml: config file
        """
        try:
            cfg = read_config(name)
        except TypeError:
            cfg = {}
        return cfg