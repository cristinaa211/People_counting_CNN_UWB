from tools.model_tools import FinalPipelineCNNTrainning
import os 
import datetime
from pytorch_lightning.loggers import TensorBoardLogger
import  subprocess


if __name__ == "__main__":
    model_name, version = "cnn_norm" ,  3
    log_dir = f"./models/{model_name}_v{version}"
    try: os.mkdir(log_dir)
    except:pass
    logger = TensorBoardLogger(log_dir, name = model_name)
    tb_process = subprocess.Popen(['tensorboard', '--logdir', log_dir, '--port', '6006'])    
    columns,table_name =  "label, radar_sample ","processed_data"
    database_config = {'host': 'localhost', 'port': 5432, 'dbname': 'UWB_Radar_Samples', 'user': 'cristina','password': 'cristina'}
    batch_size, min_epochs, max_epochs = 8, 15, 28
    debug = False
    limit = None
    label_dict = {'1' : 0, '2' : 1, '3' : 2, '4' : 3}
    pipeline = FinalPipelineCNNTrainning(batch_size)
    pipeline.forward(columns, table_name, database_config, label_dict, model_name, version,
                      max_epochs, min_epochs, debug=False,logger=logger)
    tb_process.kill()

