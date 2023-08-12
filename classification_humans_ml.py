import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import scipy.io as sio
import pickle
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report,precision_score , recall_score , f1_score, confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import neighbors, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import mat73
import os




def load_data(filename, norm):
    data = sio.loadmat(filename, mat_dtype = True)
    labels = data[:,-1]
    data = np.nan_to_num(data[:, :-1].astype(np.float32))
    if norm == True:
        data = sklearn.preprocessing.normalize(data[:, :-1].astype(np.float32))
    return data, labels

def split_data(data, labels, test_size = 0.3): 
    x_train, x_test, y_train, y_test = train_test_split( data, labels, test_size= test_size)
    return x_train, y_train, x_test, y_test
    

class AI_alg():
    
    def __init__(self, choice = 'mlp', option = 'raw', x_train = None, y_train= None,
                 x_test= None, y_test= None, print_results = None,
                 activation = 'relu', hidden_layer_size = (50,4)):
        self.choice = choice
        self.option = option
        self.x_train = x_train
        self.y_train = y_train.ravel()
        self.x_test = x_test
        self.y_test = y_test.ravel()
        self.print_results = print_results
        self.neighbours = 5
        self.activation = activation
        self.hidden_layer_size = hidden_layer_size


    def choice_model(self):
        if self.choice == 'knn':
            model =  neighbors.KNeighborsClassifier(self.neighbours)
        elif self.choice == 'random_forest':
            model =  RandomForestClassifier(n_estimators = 300, oob_score=False)
        elif self.choice == 'svm':
            model = svm.SVC(kernel='rbf')
        elif self.choice == 'lsvm':
            model = svm.SVC()
        elif self.choice == 'mlp':
            model = MLPClassifier(max_iter = 200, hidden_layer_sizes= self.hidden_layer_size, activation= self.activation,
                                   solver='adam', alpha= 1e-3,random_state=0,learning_rate='adaptive')
        return model

    def forward(self):
        clf = self.choice_model()
        clf.fit(self.x_train,self.y_train)
        self.y_pred = clf.predict(self.x_test)
        return clf

    def metrics(self):
        clf = self.forward()
        accuracy = clf.score(self.x_test, self.y_test.ravel())
        cross_val = np.mean(cross_val_score(clf, self.x_test,self.y_test,cv = 10, scoring='f1_macro'))
        f1 = np.mean(f1_score(self.y_test, self.y_pred))
        recall =  np.mean(recall_score(self.y_test, self.y_pred ))
        precision = np.mean(precision_score(self.y_test, self.y_pred ))
        print('Accuracy :', accuracy)
        print('Cross-val ', cross_val )
        print('F1 score == ', f1 )
        print('Recall == ', recall)
        print('Precision score ==', precision)
        if self.print_results == True:
        	fig, ax = plt.subplots()
             	fig.set_size_inches(15, 15, forward=True)
             	ConfusionMatrixDisplay.from_estimator(clf, self.x_test, self.y_test,normalize='true', cmap=plt.cm.Blues,ax=ax)
             	plt.title('Confusion matrix {}'.format(self.choice))
             	plt.show()
        return accuracy, f1, recall, precision,cross_val


    def save_model(self):
        with open('{}/{}_{}.pkl'.format(os.getcwd(),self.choice, self.option),'wb') as knn:
            pickle.dump(self.forward(),knn)


def get_data(data_r,labels, segm = 50,pca = False):
    date = []
    labels = []
    for i in range(0, int(data_r.shape[0]/segm)):
        date.append(data_r[segm*i:segm*(i+1),:])
        labels.append(np.mean(labels[segm*i:segm*(i+1)]))
    labels = np.array(labels)
    labels = labels.reshape(-1,1)   
    date = np.array(date)
    return date, labels

def run_alg_ml(data,labels, choice , name_csv = 'results_ml'):
    x_train, x_test, y_train, y_test = split_data(data, labels)
    metrics = []
    lista = ['knn', 'svm', 'mlp']
    for i in lista:
        print('I m training the {} model'.format(i))
        model = AI_alg(choice=i,option = choice, x_train = x_train, y_train = y_train, x_test= x_test, y_test= y_test)
        model.save_model()
        accuracy, f1, recall, precision, cross_val = model.metrics()
        metrics.append([i, accuracy, f1, recall, precision])
    np.reshape(metrics, (len(lista), 5))
    df = pd.DataFrame(data=metrics,
                          columns=['model', 'accuracy', 'precision', 'recall','f1_score'])
    df.to_csv('{}.csv'.format(name_csv), index=False)
    print(df)
    return data, labels, df

if __name__ == '__main__':
    # data,labels, df = run_alg_ml(choice = choice1, name_csv= 'res_ml_compl')
    data_pca,labels_pca, df_pca = run_alg_ml(choice = "mlp", name_csv= 'res_ml_pca')
