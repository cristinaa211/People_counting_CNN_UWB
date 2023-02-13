import torch
from torch import nn
import pytorch_lightning as pl
import mat73
import os
import numpy as np
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch.nn.functional as F
from torchsummary import summary
from torchshape import tensorshape
import matplotlib.pyplot as plt
import torchmetrics 
import sklearn
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def get_data(filename, name = 'dataset', segm = 50,pca = False):
    file = mat73.loadmat(filename)
    data_r = file[name]
    data_r = np.float32(data_r)
    labels_cnn = data_r[:,-1]
    data_cnn = data_r[:,:-1]
    date = []
    labels = []
    if pca == True:
        pca = PCA(n_components=6)
        data_cnn = pca.fit_transform(data_cnn)       
    for i in range(0, int(data_r.shape[0]/segm)):
        date.append(data_cnn[segm*i:segm*(i+1),:])
        labels.append(np.mean(labels_cnn[segm*i:segm*(i+1)]))    
    labels = np.array(labels)
    labels = labels.reshape(-1,1)   
    date = np.array(date)
    date = torch.from_numpy(date)
    date = (date - torch.mean(date))/torch.max(date)
    labels = torch.from_numpy(labels)
    return date, labels

class Model(pl.LightningModule, nn.Module):
    def __init__(self, lr = 1e-3, no_classes = 21, pca = False):
        super(Model, self).__init__()
        self.lr = lr
        self.pca = pca
        self.no_classes = no_classes
        self.loss_vect = []
        self.acc = []
        self.precision_vect = []
        self.recall_vect = []
        self.f1_score_vect = []

        ## Netowrk for input shape of (1,50,6) -> PCA use
        if self.pca:
            self.network = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size = 3),
                nn.ReLU(),
                nn.Conv2d(16,16, kernel_size = 3 ),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Flatten(),
                nn.LazyLinear(self.no_classes),
                nn.Softmax(dim=1)
            ).cuda()
        else:
            ## Netowrk for input shape of (1,50,1280)
            self.network =  nn.Sequential(
                nn.Conv2d(1, 16, kernel_size = 3),
                nn.ReLU(),
                nn.Conv2d(16,32, kernel_size = 3 ),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Conv2d(32, 64, kernel_size = 3),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Conv2d(64, 128, kernel_size = 3),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Conv2d(128, 1,kernel_size = 3),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Flatten(),
                nn.LazyLinear(self.no_classes),
                nn.Softmax(dim=1)
            ).cuda()
        #summary(network,(1,50,6))
        self.loss_func = nn.MSELoss()
        # self.loss_func = nn.NLLLoss()

    def forward(self, input_data):
        if input_data.dim() > 3:
            input_data = torch.squeeze(input_data,dim = 1)
        elif input_data.dim() == 2:
            sh1 = input_data.shape[0]
            sh2 = input_data.shape[1]
            input_data = torch.reshape(input_data, (1,sh1,sh2))
        output = self.network(input_data).cuda()
        output = torch.flatten(output)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = torch.flatten(y) 
        out = self.forward(x) 
        y = y*F.one_hot(y.to(torch.int64), self.no_classes).float()
        y = torch.flatten(y)
        loss =self.loss_func(out,y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self,batch, batch_idx):
        x, y = batch
        y = torch.flatten(y) 
        y = y*F.one_hot(y.to(torch.int64),self.no_classes).float()
        out = self.forward(x)
        y = torch.flatten(y)
        loss =self.loss_func(out,y)
        self.log('valid_loss', loss, on_epoch=True)
        return  loss

    def validation_epoch_end(self,validation_step_outputs):
        try:
            avg_loss = torch.mean(torch.tensor(validation_step_outputs))
            self.loss_vect.append(avg_loss)
        except: RuntimeError

    def test_step(self, test_batch, batch_idx ):
        x, y = test_batch
        y = torch.flatten(y) 
        y = y*F.one_hot(y.to(torch.int64),self.no_classes).float()
        out = self.forward(x)
        y = torch.flatten(y)
        loss = self.loss_func(out,y)
        self.log('test_loss', loss)
        y_test = y.detach().cpu()
        y_predict = torch.flatten(out).detach().cpu()
        return y_test, y_predict
    
    def test_step_end(self, output_results):
        y_test, y_predict = output_results[0], output_results[1]
        y_test= y_test.reshape(-1,1)
        y_predict  = y_predict.reshape(-1,1)
        acc = sklearn.metrics.accuracy_score(y_test,y_predict.round(), normalize=True)
        prec = sklearn.metrics.precision_score(y_test, y_predict.round(), average = 'micro', zero_division=1 )
        rec = sklearn.metrics.recall_score(y_test, y_predict.round(), average = 'micro', zero_division=1)
        f1 = sklearn.metrics.f1_score(y_test, y_predict.round(), average = 'micro', zero_division=1)
        self.acc.append(np.mean(acc))
        self.precision_vect.append(np.mean(prec))
        self.recall_vect.append(np.mean(rec))
        self.f1_score_vect.append(np.mean(f1))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
class Data(pl.LightningDataModule):
    def __init__(self, input_data, batch_size = 16):
        super(Data, self).__init__()
        self.input_data = input_data
        self.batch_size = batch_size
        train_size = int(0.7 * len(self.input_data))
        val_size = int(0.5 * (len(self.input_data) - train_size))
        sizes = (train_size, len(self.input_data) - train_size)
        sizes_val = (len(self.input_data) - train_size - val_size, val_size)
        self.train_data, self.test_data = random_split(self.input_data, lengths=sizes)
        self.val_data, self.test_data = random_split(self.test_data, lengths=sizes_val)

    def train_dataloader(self):
        return DataLoader(self.train_data)

    def val_dataloader(self):
        return DataLoader(self.val_data)

    def test_dataloader(self):
        return DataLoader(self.test_data)
class LitDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=self.batch_size)

class Train_model():

    def __init__(self,input_data, targets, nr_epochs, batch_size,  lr = 1e-4, pca = False):

        self.input_data = input_data
        self.targets = targets
        self.nr_epochs = nr_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = Model(lr = self.lr, pca = pca)
        self.loss_vect = 0

    def dataset(self):
        self.input_data = torch.nn.functional.normalize(self.input_data)
        dataset = TensorDataset(self.input_data, self.targets)
        data = Data(dataset)
        return data

    def train(self,model_title):
        trainer = pl.Trainer(gpus=1, max_epochs=self.nr_epochs)
        trainer.tune(self.model)
        trainer.fit(self.model, self.dataset())
        trainer.validate(self.model, self.dataset().val_data)
        self.loss_vect = getattr(self.model, 'loss_vect')
        trainer.test(self.model,dataloaders= self.dataset())
        self.precision_vect = getattr(self.model, 'precision_vect')
        self.recall_vect = getattr(self.model, 'recall_vect')
        self.f1_score_vect = getattr(self.model, 'f1_score_vect')
        self.acc = getattr(self.model, 'acc')
        print(pl.utilities.model_summary.summarize(self.model, max_depth=1))
        parameters = list(self.model.parameters())
        trainer.save_checkpoint("{}.ckpt".format(model_title))
        return parameters

def choice(choice_dataset = 'raw', filename = r'{}\dataset_few_labels.mat'.format(os.getcwd()) ):
    if choice_dataset == 'pca':
        date, labels = get_data(filename, pca=True)
    elif choice_dataset == 'raw':
        date, labels = get_data(filename)
    return date, labels

def run_cnn(option = 'raw' , model_title = 'cnn_raw_data', name_csv = 'results_cnn_pytorch',
            nr_epochs = 50, batch_size = 32, lr = 1e-3):
    '''
    Args:
        option: str
         'raw' for the entire set of data
        'pca' for the PCA data compresion
        model_title: str
        name_csv: str
        nr_epochs: int
            number of epochs
        batch_size: int
            the batch size
        lr: float
            the learning rate

    Returns:
        set of data
        labels
        list of metrics = the performance of the algorithm
    '''
    date, labels = choice(option)
    if option == 'raw':
        train_model = Train_model(date, labels, pca = False, nr_epochs = nr_epochs, batch_size = batch_size, lr=lr)
    elif option == 'pca':
        train_model = Train_model(date, labels, pca = True,  nr_epochs = nr_epochs, batch_size = batch_size, lr=lr)
    train = train_model.train(model_title = model_title)
    # metrics
    precision = np.mean(getattr(train_model, 'precision_vect'))
    recall = np.mean(getattr(train_model, 'recall_vect'))
    accuracy = np.mean(getattr(train_model, 'acc'))
    f1_score = np.mean(getattr(train_model, 'f1_score_vect'))
    lista = [accuracy, recall, precision,f1_score]
    df = pd.DataFrame(data=np.reshape(lista, (1, 4)), columns=['accuracy', 'f1_score', 'recall', 'precision'])
    df.to_csv('{}.csv'.format(name_csv), index=False)
     # loss vs number of iterations
    loss_vect = getattr(train_model, 'loss_vect')
    plt.figure()
    plt.plot(range(0, len(loss_vect)), loss_vect)
    plt.xlabel('Nr iterations')
    plt.ylabel('Loss')
    plt.savefig('{}.png'.format(option))
    plt.show()
    plt.pause(10)
    plt.close()
    return date, labels,lista


if __name__ == '__main__':
    # date, labels, perf_raw = run_cnn(option= 'raw', nr_epochs=50, model_title='raw_final', name_csv='raw_final')
    date_pca, labels_pca, perf_pca = run_cnn(option= 'pca', nr_epochs=50, model_title='pca_final', name_csv='pca_final')