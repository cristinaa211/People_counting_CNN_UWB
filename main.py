from tools.model_tools import FinalPipelineCNNTrainning
import os 
import datetime
from pytorch_lightning.loggers import TensorBoardLogger
import  subprocess
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    model_name, version = "cnn_detailed_label" ,  1
    log_dir = f"./models/{model_name}_v{version}"
    try: os.mkdir(log_dir)
    except:pass
    logger = TensorBoardLogger(log_dir, name = model_name)
    tb_process = subprocess.Popen(['tensorboard', '--logdir', log_dir, '--port', '6006'])    
    columns,table_name =  "detailed_label, radar_sample ","processed_data"
    database_config = {'host': 'localhost', 'port': 5432, 'dbname': 'UWB_Radar_Samples', 'user': 'cristina','password': ''}
    batch_size, min_epochs, max_epochs = 4, 25, 40
    debug = False
    limit = None
    # label_dict = {'1' : 0, '2' : 1, '3' : 2, '4' : 3}
    pipeline = FinalPipelineCNNTrainning(batch_size)
    # labels_, data = pipeline.extract_data_db(columns, table_name, database_config, limit=None)
    # df_labels = pd.DataFrame(labels_).value_counts().plot(kind="bar")
    # plt.xlabel("Labels")
    # plt.ylabel("Amount of samples")
    # plt.show()
    pipeline.forward(columns=columns,table_name= table_name, database_config=database_config,model_name= model_name,version= version, 
                     label_dict=None, max_epochs=max_epochs, min_epochs= min_epochs, debug=False,logger=logger)
    tb_process.kill()

