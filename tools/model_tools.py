import torch
import torch.nn as nn
from torch.utils.data import  DataLoader, random_split, TensorDataset
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import EarlyStopping, LearningRateFinder
import torchmetrics
import  matplotlib.pyplot as plt


class CNN(pl.LightningModule):
    """Creates a Convolutional Neural Network with 5 convolution layers, 2 Max Pooling layers and 2 Fully connected layers
    """
    def __init__(self, num_classes, batch_size= 8, lr = 1e-3):
        """
        Args:
            num_classes (int): number of classes for multiclass clasification task
            batch_size (int, optional): the batch size. Defaults to 8.
            lr (float, optional): the learning rate. Defaults to 1e-3.
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(32,64, kernel_size=(3,3))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3))
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3,3))
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(3,3))
        self.pool = nn.MaxPool2d(3,3)
        self.dense_1 = nn.Linear(9216 , 1024)
        self.dense_2 = nn.Linear(1024, 256)
        self.dense_3 = nn.Linear(256, num_classes)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = lr
        self.loss_fc = nn.CrossEntropyLoss()
        self.train_acc_tensor, self.val_acc_tensor, self.test_acc_tensor = [], [], []
        self.accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.pool(x))
        x = nn.ReLU()(self.conv3(x))
        x = nn.ReLU()(self.conv4(x))
        x = nn.ReLU()(self.pool(x))
        x = nn.ReLU()(self.conv5(x))
        x = torch.flatten(x, 1)
        x = nn.ReLU()(self.dense_1(x))
        x = nn.ReLU()(self.dense_2(x))
        x = self.dense_3(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = self.loss_fc(outputs, y)
        self.log(f"train_loss", loss, prog_bar=True)
        acc = self.accuracy_metric(outputs, y)
        self.log(f"train_acc", acc, on_epoch=True, prog_bar=True)
        self.train_acc_tensor.append(acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = self.loss_fc(outputs, y)
        self.log(f"val_loss", loss, prog_bar=True)
        self.val_acc = self.accuracy_metric(outputs, y)
        self.log(f"val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.val_acc_tensor.append(self.val_acc)
        return loss
    
    def test_step(self, batch,batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = self.loss_fc(outputs, y)
        self.log(f"test_loss", loss)
        self.test_acc = self.accuracy_metric(outputs, y)
        self.log(f"test_acc", self.test_acc, on_epoch=True, prog_bar=True)
        self.test_acc_tensor.append(self.test_acc)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        return optimizer

    def on_test_epoch_end(self):
        self.log("test_acc_epoch", self.test_acc, on_epoch=True, prog_bar=True)
    
    def on_validation_epoch_end(self):
        self.log("val_acc_epoch", self.val_acc, on_epoch=True, prog_bar=True)



class ProcessData:
    def __init__(self, data , labels, label_dict):
        self.label_dict = label_dict
        self.data = data 
        self.labels = labels

    def normalize_data(self, data):
        """Transforms data to torch tensors and normalize them between 0 and 1"""
        data = torch.tensor(data)
        return torch.nn.functional.normalize(data)
    
    def process_data(self):
        labels = [self.label_dict[l[0]] for l in self.labels]
        labels_ = torch.tensor(labels)
        data_norm = self.normalize_data(self.data)
        return data_norm, labels_

class CustomDataset(pl.LightningDataModule):
    def __init__(self, input_data, batch_size):
        super(CustomDataset, self).__init__()
        self.input_data = input_data
        self.batch_size = batch_size
        train_size = int(0.7 * len(self.input_data))
        val_size = int(0.5 * (len(self.input_data) - train_size))
        sizes = (train_size, len(self.input_data) - train_size)
        sizes_val = (len(self.input_data) - train_size - val_size, val_size)
        self.train_data, self.test_data = random_split(self.input_data, lengths = sizes)
        self.val_data, self.test_data = random_split(self.test_data, lengths = sizes_val)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=8)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

class CreateDataloader():
    def __init__(self, input_data, labels, batch_size):
        self.input_data = input_data
        self.labels = labels 
        self.batch_size = batch_size

    def create_dataloader(self):
        dataset = TensorDataset(self.input_data, self.labels)
        dataset = CustomDataset(dataset, batch_size= self.batch_size)
        return dataset

class TrainModel():
    def __init__(self, dataloader, model,):
        self.loss_vect = 0
        self.dataloader = dataloader
        self.model = model

    def train_model(self, min_epochs, max_epochs, debug = False, logger = None):
        self.trainer = pl.Trainer(devices="auto", accelerator="auto", 
                                max_epochs= max_epochs, min_epochs=min_epochs, 
                                fast_dev_run=debug, log_every_n_steps=10, logger = logger,
                                callbacks=[EarlyStopping(monitor="train_acc_epoch")], check_val_every_n_epoch=4)
        self.trainer.fit(self.model, self.dataloader.train_dataloader())
        self.trainer.validate(model=self.model, dataloaders=self.dataloader.val_dataloader(), verbose=True)
    
    def evaluate_model(self):
        self.trainer.test(self.model, self.dataloader.test_dataloader())
        print(pl.utilities.model_summary.summarize(self.model))
    
    def save_weights(self, version, model_name):
        self.trainer.save_checkpoint("./models/{}_v{}.ckpt".format(model_name, version))

    def compare_accuracies(self):
        train_accuracy = getattr(self.model, 'train_acc_tensor')
        train_accuracy = torch.mean(torch.stack(train_accuracy), dim=0)
        val_accuracy = getattr(self.model, 'val_acc_tensor' )
        val_accuracy = torch.mean(torch.stack(val_accuracy), dim=0)        
        test_acc = getattr(self.model, 'test_acc_tensor')
        test_acc = torch.mean(torch.stack(test_acc), dim=0)   
        print("Training accuracy = {}, validation accuracy = {}, test accurary = {}".format(train_accuracy, val_accuracy, test_acc))
        return train_accuracy, val_accuracy, test_acc








