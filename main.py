from tools.postgresql_operations import read_table_postgresql
from tools.model_tools import ProcessData, CreateDataloader, CNN, TrainModel
import os 
import datetime
from pytorch_lightning.loggers import TensorBoardLogger
import  subprocess


if __name__ == "__main__":
    log_dir = "./models/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
    try:
        os.mkdir(log_dir)
    except:pass
    logger = TensorBoardLogger(log_dir, name = 'CNN')
    tb_process = subprocess.Popen(['tensorboard', '--logdir', log_dir, '--port', '6006'])    
    print("TensorBoard is open. Continuing with other tasks.")
    columns =  "label, radar_sample "
    table_name = "processed_data"
    database_config = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'UWB_Radar_Samples',
    'user': 'cristina',
    'password': 'cristina'
    }
    batch_size = 8
    min_epochs = 15
    max_epochs = 20
    model_name = "cnn"
    version = 3
    debug = False
    limit = None
    label_dict = {'1' : 0, '2' : 1, '3' : 2, '4' : 3 }
    headers, dataset = read_table_postgresql(columns=columns, table_name=table_name, 
                                             database_config=database_config, limit=limit)
    labels, data = list(zip(*dataset))
    num_classes = len(set(labels)) 
    process_pipeline = ProcessData(data, labels, label_dict)
    input_data, labels = process_pipeline.process_data()
    dataloader = CreateDataloader(input_data=input_data, labels=labels, batch_size=batch_size)
    dataload = dataloader.create_dataloader()
    dict_acc = {}
    model = CNN(batch_size=batch_size, num_classes=num_classes, lr = 1e-4)
    trainer = TrainModel(dataload, model)
    trainer.train_model(min_epochs = min_epochs, max_epochs = max_epochs, debug=debug, logger=logger)
    trainer.evaluate_model()
    acc_train, acc_val, acc_test = trainer.compare_accuracies()
    trainer.save_weights(version=version, model_name=model_name)
    tb_process.kill()

