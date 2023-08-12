from tools.visualize_data import visualize_radar_samples_by_scenario
from tools.postgresql_operations import read_table_postgresql
import numpy as np 

def extract_signal_db(table_name,number_persons, database_config):
    query = """SELECT radar_sample FROM PUBLIC.{} WHERE number_persons = '{}' LIMIT 1;""".format(table_name, number_persons)
    headers, data = read_table_postgresql(table_name=table_name,database_config= database_config, limit = 1, query = query)
    radar_sample = np.array(data)
    radar_sample_resh = np.reshape(radar_sample, (200,1280))
    radar_sample_resh_ = radar_sample_resh - np.mean(radar_sample_resh)
    return headers, radar_sample_resh_

if __name__ == "__main__":
    pass


