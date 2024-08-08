import logging
import os
from pathlib import Path
from datetime import datetime
from accelerate import Accelerator
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import natsort as ns

from utils.aux_func import ProjectConfig

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, train_loader, optimizer,
                 device, config:ProjectConfig, val_loader=None):
        self.accelerator = Accelerator()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        if config.model.scheduler == 'StepLR':
            from torch.optim import lr_scheduler
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=150, gamma=0.5)  # Define your learning rate scheduler
        self.device = device
        self.config = config
        self.model_path = Path(config.project_directory) / 'model'

        # Prepare model, dataloaders, optimizer with accelerator
        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )

        # Setup TensorBoard
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = Path(config.project_directory) / f'tensorboard/{now}'
        self.writer = SummaryWriter(log_dir)
        self.writer.add_text('config', str(config))
        self.model_resume = False

    def train(self, epochs):
        latest_best = self.check_trained_model(self.model_path)
        best_val_loss = float('inf')
        start_epoch = 0
        if self.model_resume:
            start_epoch = int(latest_best.stem.split('_')[-1])
            if epochs <= start_epoch:
                logger.info("No epochs left to train. Increase the number of epochs in the configuration.")
                exit()
            logger.info(f"Resuming training from epoch {start_epoch}")

        for epoch in range(start_epoch, epochs):
            progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}", dynamic_ncols=True)
            for i, (data, _) in enumerate(progress_bar):
                data = data.to(self.device)
                self.optimizer.zero_grad()  # Reset the gradients
                x_hat, mu, logvar = self.model(data)  # Forward pass and loss computation
                z = self.model.reparameterize(mu, logvar)  # Get the latent vector
                loss = self.model.loss_function(data, x_hat, mu, logvar)
                self.accelerator.backward(loss)  # Use accelerator for backward pass
                self.optimizer.step()  # Update the weights

                if self.config.model.scheduler == 'StepLR':
                    self.scheduler.step()
                    lr = torch.tensor(self.scheduler.get_last_lr())
                    self.writer.add_scalar('Learning_rate', lr, epoch * len(self.train_loader) + i)

                self.writer.add_scalar('Loss/train', loss.item(), epoch * len(self.train_loader) + i)

                # Log histograms of weights from the first layer of the model
                for name, param in self.model.named_parameters():
                    if 'weight' in name:
                        self.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch * len(self.train_loader) + i)

                progress_bar.set_postfix({'loss': loss.item()})

            logger.info(f"Epoch {epoch + 1}: Loss = {loss.item()}")

            # Validate after every epoch
            if self.val_loader:
                val_loss = self.validate()
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.accelerator.wait_for_everyone()  # Ensure all processes have finished before saving
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    torch.save(unwrapped_model.state_dict(), f'{self.config.project_directory}/model/best_model/best_model.pth')
                    logger.info(f"Best model saved to {self.config.project_directory}/model/best_model/best_model.pth")

            # Save the model after defined number of epoch(s)
            if (epoch + 1) % self.config.training.save_every == 0:
                self.accelerator.wait_for_everyone()
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                save_path = self.model_path / f'checkpoint_epoch_{epoch + 1}.pth'
                torch.save(unwrapped_model.state_dict(), save_path)
                logger.info(f"Checkpoint saved to {save_path}")

            # Log the embeddings
            self.writer.add_embedding(z, global_step=epoch + 1, tag="latent_vector")
            logger.info('Latent vector embeddings logged')

        # Log hyperparameters and final loss
        hparam_dict = self.get_hyperparameters(self.config)
        metric_dict = {'Final_loss': loss.item()}
        self.writer.add_hparams(hparam_dict, metric_dict)
        logger.info('Hyperparameters logged')

        self.writer.close()

    def validate(self):
        """Validate the model on the validation dataset."""
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

    def get_hyperparameters(config:ProjectConfig):
        # Extract hyperparameters from training config
        training_hparams = {field.name: getattr(config.training, field.name) for field in fields(config.training)}

        # Extract hyperparameters from model config
        model_hparams = {field.name: getattr(config.model, field.name) for field in fields(config.model)}

        # Merge both dictionaries
        hparam_dict = {**training_hparams, **model_hparams}
        
        return hparam_dict
    
    def check_trained_model(self, model_path):
        """Check if a trained model exists and prompt to resume training if found."""


        if any(model_path.glob('*.pth')):
            latest_model = ns.natsorted(model_path.glob('*.pth'))[-1]
            logger.warning(f"Trained best model found: {latest_model}")
            logger.info(f"Loading model from {latest_model}")
            

            resume_training = input("Do you want to resume training? (y/n): ")
            if resume_training.lower() == 'y':
                self.model.load_state_dict(torch.load(latest_model))
                self.model.train()
                self.model.to(self.device)
                self.model_resume = True
                return latest_model
            else:
                logger.info("Training aborted. Run evaluate.py to evaluate the model.")
                exit()
        else:
            logger.info("Training from scratch.")
            self.model.train()
            self.model.to(self.device)
            return self.model
