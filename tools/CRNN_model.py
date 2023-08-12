from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import EarlyStopping

class CNN(pl.LightningModule):
    def __init__(self, num_classes, batch_size= 32, lr = 1e-3):
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv2d(1,16, kernel_size=(3,3), padding_mode='zeros')
        self.conv2 = nn.Conv2d(16,32, kernel_size=(3,3), padding_mode='zeros')
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3), padding_mode='zeros')
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3,3), padding_mode='zeros')
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3,3), padding_mode='zeros')
        self.fc = nn.Linear(256, num_classes)
        self.batch_size = batch_size
        self.learning_rate = lr
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = x[:,-1,:]
        x = self.fc(x)
        return x
    
    def training_step(self, batch, batch_idx):
        input, labels = batch
        outputs = self(input)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Validation step (called for each batch during validation)
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        # Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_Rate)
        return optimizer

class CRNN(pl.LightningModule):
    def __init__(self, num_classes, batch_size= 32, lr = 1e-3):
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv2d(1,16, kernel_size=(3,3), padding_mode='zeros')
        self.conv2 = nn.Conv2d(16,32, kernel_size=(3,3), padding_mode='zeros')
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3), padding_mode='zeros')
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3,3), padding_mode='zeros')
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3,3), padding_mode='zeros')
        self.lstm1 = nn.LSTM(input_size = 256, hidden_size = 512, num_layers = 1, batch_first = True )
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, num_classes)
        self.batch_size = batch_size
        self.learning_rate = lr
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = x.view(x.size(0), x.size(1, -1)).permute(0,2,1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:,-1,:]
        x = self.fc(x)
        return x
    
    def training_step(self, batch, batch_idx):
        input, labels = batch
        outputs = self(input)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Validation step (called for each batch during validation)
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        # Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_Rate)
        return optimizer
          
        
class RadarDataset(Dataset):
    def __init__(self, radar_data, labels, transform=None):
        self.radar_data = radar_data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.radar_data)

    def __getitem__(self, idx):
        sample = {
            'radar_sample': self.radar_data[idx],
            'label': self.labels[idx]
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
    
class ToTensor:
    def __call__(self, sample):
        radar_data, label = sample['radar_sample'], sample['label']
        return {
            'radar_data': torch.tensor(radar_data, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.int64)
        }

def create_dataloader(radar_data, labels, batch_size=16, shuffle=True):
    transform = ToTensor() 
    dataset = RadarDataset(radar_data, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


model = CRNN(4)
#1
# Choose the right batch size 
trainer = pl.Trainer(auto_scale_batch_size = True, auto_lr_find = True, truncated_bptt_steps = 5,
                     callbacks=[EarlyStopping('val_loss')])
trainer.tune(model)
#Find the learning rate
model.learning_rate
trainer.fit(model)



#2 For a single optimizer
lr_finder = trainer.tuner.lr_find(model)
fig = lr_finder.plot(suggest = True)
fig.show()
trainer.fit(model, train_loader, val_loader)

# ALL THESE ARE IN TRAINER 

# Vanishing Gradients : when the gradient turns to zero
# Exploding Gradients : when the gradient turns to infinit caused by the propagation, where you multiply things that are above zero
# track_grad_norm flag
# reload_dataloaders_every_epoch = True when you have a model in the production or data is changing
# weights_summary = 'full' or = 'top' displays the parameters for each layer 
# progress_bar_refresh_rate 
# profiler = True gives a high level descriptions of methods called and how long it took 
# min_epochs, max_epochs flags = int
# min_steps, max_steps flags = int
# check_val_every_n_epochs = int
# val_check_interval = 0.25 
# num_sanity_val_steps = 2 batches of validation 
# limit_train_batches, limit_val_batches, limit_test_batches
# ON A SINGLE MACHINE 
# GPU: put each tensor by default on the device (cuda)
# gpus = 4, auto_select_gpus = True, log_gpu_memory = "all" , "min_max", benchmark=True tp speed the training if data does not change
# deterministic = True to garantee the reproducible results to reduce the randomness in the training 
# TO RUN ON MULTIPLE MACHINES
# distributed_backend = 'ddp_spwan', gpus = 8, num_nodes = 8
# Debugging flags 
# fast_dev_run = True 
# overfit_batches = 1 pick  a single batch and overfit the batch, if it does not overfit, you have a bug
# accumulate_grad_batches = 4 : the number of forward steps when you have large data 
# mixed precision,  to reduce the memory and speedup the GPUs
# precision = 16
# 
