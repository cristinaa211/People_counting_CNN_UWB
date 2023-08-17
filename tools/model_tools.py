import torch
import torch.nn as nn
from torch.utils.data import  DataLoader, random_split, TensorDataset
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import EarlyStopping
import torchmetrics
from tools.postgresql_operations import read_table_postgresql
import datetime
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

class FinalPipelineCNNTrainning:
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size

    def extract_data_db(self, columns, table_name, database_config , limit):
        """Extracts data from the database
        Args:
            columns (list[strings]): the names of the columns to be extracted from the DB 
            table_name (string): the table we want to extract data from
            database_config (dict): the database connection configuration like {
                                                                            'host': string,
                                                                            'port': int,
                                                                            'dbname': string,
                                                                            'user': string,
                                                                            'password': string
                                                                            }
            limit (int): the number of rows to be extracted from the table name
        Returns:
            labels : list of labels
            data   : list of data arrays
        """
        headers, dataset = read_table_postgresql(columns=columns, table_name=table_name, 
                                                database_config=database_config, limit=limit)
        labels, data = list(zip(*dataset))
        return labels, data 
    
    def transform_data(self, labels,  data, labels_dict = None):
        """Encodes labels and transform data to torch arrays
        Args:
            labels           : the lables list 
            label_dict (dict): the labels encoding 
                            such as {'1' : 0, '2' : 1, '3' : 2, '4' : 3 } where keys are original labels and values are the encoded labels
        Returns:
            labels_  : list of torch tensors representing the encoded lables    
            data     : list of torch arrays, each row representing a sample, each column a feature 
        """
        if labels_dict != None:
            labels_ = [labels_dict[l[0]] for l in labels]
        else: 
            try:  labels_ = [int(l[2]) if int(l[1]) == '0' else int(l[1:]) for l in labels]
            except Exception as e:
                print(e)
        self.num_classes = len(set(labels_))
        labels_ = torch.tensor(labels_)
        data = torch.tensor(data)
        self.features_shape = torch._shape_as_tensor(data)
        return labels_, data
    
    def create_dataloader(self, data, labels):
        """Creates a TensorDataset, normalizes the data, splits the data into training, validation and test data
        and creates training, validation and test dataloaders"""
        self.dataloader = CustomDataloader(input_data=data, labels=labels, batch_size=self.batch_size)
        return self.dataloader
    
    def create_cnn_model(self, lr = 1e-4):
        """Creates a Convolutional Neural Network having as attributes the batch size, the number of classes and the learning rate default to 0.0001"""
        self.learning_rate = lr
        self.model = CNN(batch_size=self.batch_size, num_classes=self.num_classes, lr = self.learning_rate)

    def train_model(self, max_epochs, min_epochs, debug, logger):
        """Trains the model on a minimum of min_epochs and a maximum of max_epochs

        Args:
            model (model object): The CNN model 
            max_epochs (int): maximum number of training epochs
            min_epochs (int): minimum number of training epochs
            debug (boolean): enables fast_dev_run attribute in the trainer class
            logger (TensorBoardLogger): the TensorBoard Logger
        """
        self.trainer = TrainModel(dataloader=self.dataloader,model= self.model)
        self.trainer.train_model(min_epochs = min_epochs, max_epochs = max_epochs, debug=debug, logger=logger)
    
    def evaluate_model(self):
        """Evaluates the model on the test dataset and compares training, validation and test accuracy"""
        self.trainer.evaluate_model()
        self.acc_train, self.acc_val, self.acc_test = self.trainer.compare_accuracies()
    
    def save_model(self, model_name, version):
        """Saves the model's parameters along with the model name, version, 
        datetime, mean of the training data, standard deviation of the training data"""
        json_file_path = f"./models/{model_name}_v{version}/info.json"
        self.trainer.save_weights(version, model_name)
        info= {
                "model_name" : model_name,
                "version" : version,
                "datetime" : datetime.datetime.now().strftime("%Y_%m_%d-%H_%M"),
                "mean" : getattr(self.dataloader, "mean").numpy().tolist(),
                "std"  : getattr(self.dataloader, "std").numpy().tolist(), 
                "batch_size" : self.batch_size,
                "learning_rate" : self.learning_rate,
                "num_classes" : self.num_classes,
                "feature_set_shape" : self.features_shape,
                "loss_function" : getattr(self.model, "loss_fc"),
                "training_accuracy" : self.acc_train.cpu().numpy().tolist(),
                "validation_accuracy": self.acc_val.cpu().numpy().tolist(),
                "test_accuracy" : self.acc_test.cpu().numpy().tolist()
                }
        with open(json_file_path, "w") as json_file:
            json.dump(info, json_file, indent = 4)

    def forward(self, columns, table_name, database_config, label_dict, model_name, version,
                    max_epochs, min_epochs,logger, debug = False ):
        """Runs the data extraction, data transformation, dataloader creation, model creation, 
        model training, evaluation and parameters saving"""
        labels, data = self.extract_data_db(columns, table_name, database_config, limit = None)
        labels_, data_ = self.transform_data(labels=labels,data= data, labels_dict=label_dict)
        self.create_dataloader(data=data_, labels=labels_)
        self.create_cnn_model()
        self.train_model(max_epochs=max_epochs, min_epochs=min_epochs, debug=debug, logger=logger )
        self.evaluate_model()
        self.save_model(model_name, version)

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
        self.log("train_acc_epoch",  acc, on_epoch=True, prog_bar=True)
        self.train_acc_tensor.append(acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = self.loss_fc(outputs, y)
        self.val_acc = self.accuracy_metric(outputs, y)
        self.val_acc_tensor.append(self.val_acc)
        return loss
    
    def test_step(self, batch,batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = self.loss_fc(outputs, y)
        self.test_acc = self.accuracy_metric(outputs, y)
        self.test_acc_tensor.append(self.test_acc)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        return optimizer
    
class ProcessData:
    def __init__(self, data, mean, std):
        self.data = data 
        
    # all the processing steps 
    # PCA feature extraction 
    # normalization 
    def normalize_data(self):
        """Transforms data to torch tensors and normalize them between 0 and 1"""
        normalized_data = (self.data - self.mean) / self.std
        return normalized_data

class CustomDataloader(pl.LightningDataModule):
    def __init__(self, input_data, labels, batch_size):
        super(CustomDataloader, self).__init__()
        self.batch_size = batch_size
        dataset = TensorDataset(input_data, labels)
        train_data, val_data, test_data = self.split_data(dataset=dataset)
        self.train_data = self.normalize_data(train_data, mode = 'training')
        self.val_data = self.normalize_data(val_data)
        self.test_data = self.normalize_data(test_data)

    def split_data(self, dataset):
        """Splits TensorDataset type of dataset into training data, validation data and test data"""
        train_size = int(0.7 * len(dataset))
        val_size = int(0.5 * (len(dataset) - train_size))
        sizes = (train_size, len(dataset) - train_size)
        sizes_val = (len(dataset) - train_size - val_size, val_size)
        train_data, test_data = random_split(dataset, lengths = sizes)
        val_data, test_data = random_split(test_data, lengths = sizes_val)
        return train_data, val_data, test_data

    def normalize_data(self, data, mode = None):
        """Normalizes data between 0 and 1 and saves the mean and standard deviation from the training data"""
        data_tensors = torch.stack([sample for sample, _ in data])
        labels = torch.stack([l for _, l in data])
        if mode ==  'training':
            self.mean = torch.mean(data_tensors)
            self.std = torch.std(data_tensors)
        norm_data = (data_tensors -  self.mean) / self.std 
        return TensorDataset(norm_data, labels)
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=8)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)
    
    def save_normalization_parameters(self):
        return self.mean, self.std


class TrainModel():
    def __init__(self, dataloader, model):
        self.loss_vect = 0
        self.dataloader = dataloader
        self.model = model

    def train_model(self, min_epochs, max_epochs, debug = False, logger = None):
        self.trainer = pl.Trainer(devices="auto", accelerator="auto", 
                                max_epochs= max_epochs, min_epochs=min_epochs, 
                                fast_dev_run=debug, log_every_n_steps=10, logger = logger,
                                callbacks=[EarlyStopping(monitor="train_loss")], check_val_every_n_epoch=3)
        self.trainer.fit(self.model, self.dataloader.train_dataloader())
        self.trainer.validate(model=self.model, dataloaders=self.dataloader.val_dataloader(), verbose=True)
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs

    def evaluate_model(self):
        self.trainer.test(self.model, self.dataloader.test_dataloader())
        print(pl.utilities.model_summary.summarize(self.model))
    
    def save_weights(self, version, model_name):
        model_parameters_path = f"./models/{model_name}_v{version}/{model_name}_v{version}.pkl"
        torch.onnx.export(self.model, (self.min_epochs, self.max_epochs)) 
        self.trainer.save_checkpoint(model_parameters_path)

    def compare_accuracies(self):
        train_accuracy = getattr(self.model, 'train_acc_tensor')
        train_accuracy = torch.mean(torch.stack(train_accuracy), dim=0)
        val_accuracy = getattr(self.model, 'val_acc_tensor' )
        val_accuracy = torch.mean(torch.stack(val_accuracy), dim=0)        
        test_acc = getattr(self.model, 'test_acc_tensor')
        test_acc = torch.mean(torch.stack(test_acc), dim=0)   
        print("Training accuracy = {}, validation accuracy = {}, test accurary = {}".format(train_accuracy, val_accuracy, test_acc))
        return train_accuracy, val_accuracy, test_acc




class CNNModelDeployment:
    def __init__(self, model_path, num_classes):
        self.model_path = model_path
        self.num_classes = num_classes

    def load_model(self):
        if self.model_path.endswith("pkl"):
            model = CNN(num_classes=self.num_classes)
            self.model = model.load_from_checkpoint(self.model_path, num_classes = self.num_classes).to(device)
        elif self.model_path.endswith("oonx"):
            self.model = torch.onnx.load(self.model_path)
            # Check that the model is well formed
            nn.onnx.checker.check_model(self.model)
            # Print a human readable representation of the graph
            print(nn.onnx.helper.printable_graph(self.modeldel.graph))
    
    def predict(self, data):
        self.load_model()
        self.model.eval()
        outputs = self.model(data)
        output = torch.argmax(outputs)
        return output
    
    
