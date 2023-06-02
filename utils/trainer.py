from datetime import datetime
import os
from pathlib import Path
import torch
import logging
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
import natsort as ns
import coloredlogs

# setting up logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        if config['Model']['scheduler'] == 'StepLR':
            from torch.optim import lr_scheduler
            self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                            step_size=150, gamma=0.5) # Define your learning rate scheduler
        self.device = device
        self.config = config
        self.model_path = Path(config['Project']['project_directory']) / 'model' 
        # Get current date and time
        now = datetime.now()
        # Format datetime object to string
        time_string = now.strftime("%Y-%m-%d_%H-%M-%S")
        # Append time_string to log directory
        log_dir = Path(os.path.join(self.config['Project']['project_directory'], f'tensorboard/{time_string}'))
        self.writer = SummaryWriter(log_dir)
        self.writer.add_text('config', str(self.config))
        self.model_resume = False

    def train(self, epochs):
        latest_best = self.check_trained_model(self.model_path)
        best_val_loss = float('inf')
        # self.model.train()  # Set the model to training mode
        start_epoch = 0
        if self.model_resume:
            start_epoch = int(latest_best.stem.split('_')[-1])
            #epochs = epochs - start_epoch
            if epochs <= start_epoch:
                logger.info("No epochs left to train")
                logger.info("run evaluate.py to evaluate the model or increase the number of epochs in config.yaml")
                logger.info("Exiting...")
                exit()
            
            logger.info(f"Resuming training from epoch {start_epoch}")
            
        for epoch in range(start_epoch, epochs):
            z = None
            progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}", dynamic_ncols=True)
            for i, (data, _) in enumerate(progress_bar):
                data = data.to(self.device)
                self.optimizer.zero_grad()  # Reset the gradients
                x_hat, mu, logvar = self.model(data)  # Forward pass and loss computation
                z = self.model.reparameterize(mu, logvar)  # Get the latent vector
                loss = self.model.loss_function(data, x_hat, mu, logvar)
                loss.backward()  # Backward pass
                self.optimizer.step()  # Update the weights
                if self.config['Model']['scheduler'] == 'StepLR':
                    self.scheduler.step()
                    lr = torch.tensor(self.scheduler.get_last_lr())
                    self.writer.add_scalar(
                        'Learning_rate', 
                        lr,
                        epoch *  len(self.train_loader) + i
                                           )
                
                self.writer.add_scalar('Loss/train',
                                       loss.item(),
                                       epoch * len(self.train_loader) + i
                                     )
                
                # Log histograms of weights from the first layer of the model
                for name, param in self.model.named_parameters():
                    if 'weight' in name:
                        self.writer.add_histogram(name,
                                                  param.clone().cpu().data.numpy(),
                                                  epoch *  len(self.train_loader) + i)
                
                progress_bar.set_postfix({'loss': loss.item()})
                
            logger.info(f"Epoch {epoch + 1}: Loss = {loss.item()}")
                        
            # Validate after every epoch
            val_loss = self.validate()
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), f'{self.config["Project"]["project_directory"]}/model/best_model/best_model.pth')
                logger.info(f"Best model saved to {self.config['Project']['project_directory']}/model/best_model/best_model.pth") 
                
            # Save the model after defined number of epoch(s) 
            if (epoch + 1) % self.config['Model']['save_every'] == 0:
                save_path = Path(self.config['Project']['project_directory']) / 'model'
                torch.save(self.model.state_dict(), f'{save_path}/checkpoint_epoch_{epoch + 1}.pth')
                logger.info(f"Checkpoint saved to {save_path} / checkpoint_epoch_{epoch + 1}.pth")
            
            # Log the embeddings. For better visualization, you may want to do this less frequently than every iteration.
            self.writer.add_embedding(z, 
                                        global_step=epoch+1, 
                                        tag="latent_vector")
            logger.info('Latent vector embeddings logged', )
            
        # Log hyperparameters and final loss
        hparam_dict = { 'lr': self.config['Model']['learning_rate'], 
                        'batch_size': self.config['Model']['batch_size'],
                        'num_layers': self.config['Model']['num_layers'], 
                        'hidden_size': self.config['Model']['hidden_size'],
                        'latent_dim': self.config['Model']['latent_dim'],
                        'decoder_type': self.config['Model']['decoder_type'],
                        'encoder_type': self.config['Model']['encoder_type'],
                        'optimizer': self.config['Model']['optimizer'],
                        'scheduler': self.config['Model']['scheduler']
                        
                        }
        metric_dict = {'loss': loss.item()}
        self.writer.add_hparams(hparam_dict, metric_dict)
        logger.info('Hyperparameters logged')
        
        self.writer.close()
        

    def validate(self):
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0
        with torch.no_grad():  # Do not compute gradient for validation
            for data, _ in self.val_loader:
                data = data.to(self.device)
                x_hat, mu, logvar = self.model(data)
                loss = self.model.loss_function(data, x_hat, mu, logvar)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        logger.info(f"Validation Loss: {avg_loss}")

        return avg_loss
    
    def check_trained_model(self, model_path):
        if os.listdir(model_path / 'best_model') != []:
            latest_best_model = ns.natsorted(model_path.glob('*.pth'))[-1]
            logger.warning(f"TRAIND BEST MODEL FOUND IN {latest_best_model}")
            logger.info(f"Loading model from {latest_best_model}")
            self.model.load_state_dict(torch.load(latest_best_model))
            resume_training = input("Do you want to resume training? (y/n): ")
            if resume_training == 'y':
                self.model.train()
                self.model.to(self.device)
                self.model_resume = True
                return latest_best_model
            else:
                logger.info("Training aborted")
                logger.info("run evaluate.py to evaluate the model")
                logger.info("Exiting...")
                exit()
        else:
            logger.info("Training from scratch")
            self.model.train()
            self.model.to(self.device)
            return self.model
