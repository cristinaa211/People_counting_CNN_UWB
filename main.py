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
    database_config = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'UWB_Radar_Samples',
    'user': 'cristina',
    'password': 'cristina'
    }
    tables ={
            "scenario_1": ["people_walking_5m_area", "people walking in 5m area"],
            "scenario_2" :["people_standing_queue_0_15", "people standing in a queue"],
            "scenario_3":  ["density_3_m2_11_20", "people walking in a room with 3 persons per m2"],
            "scenario_4": ["density_4_m2_11_20" , "people walking in a room with 4 persons per m2"]
            }
    table_name = tables["scenario_4"][0]
    scenario =  tables["scenario_4"][1]
    number_persons = 20
    # headers, data = extract_signal_db(table_name=table_name,number_persons = number_persons,database_config= database_config)
    visualize_radar_samples_by_scenario(scenario=scenario, table_name=table_name,number_persons=number_persons, database_config=database_config)



