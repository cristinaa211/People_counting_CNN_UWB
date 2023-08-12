from postgresql_operations import read_table_postgresql
import matplotlib.pyplot as plt
from process_data import process_data_pipeline
import numpy as np 

def plot_radar_signal(data, title):
    """Plots a single received signal from the radar sample"""
    plt.figure(figsize = (10, 5))
    plt.plot(data)
    plt.xlabel('# Sample (1280 samples = 5 meters range)')
    plt.ylabel("Amplitude [V]")
    plt.title(f"{title}")
    plt.savefig("./images_/{}".format(title))
    plt.show()

def visualize_radar_samples(data, title):
    """Plots a radar sample"""
    plt.figure(figsize = (10, 3))
    plt.imshow(data,interpolation="none")
    plt.xlabel('# Sample')
    plt.ylabel("")
    plt.title(f"{title}")
    plt.savefig("./images_/{}".format(title))
    plt.show()

def extract_data_per_no_persons(table_name, detailed_label, database_config):
    """Extracts a sigle radar sample from table_name, based on the number of persons number_persons"""
    query = """SELECT radar_sample FROM PUBLIC.{} WHERE detailed_label = '{}' LIMIT 1;""".format(table_name, detailed_label)
    headers, data = read_table_postgresql(table_name=table_name,database_config= database_config, limit = 1, query = query)
    return data 

def visualize_radar_samples_by_scenario(data, scenario, number_persons):
    """Visualize a radar sample and a single received signal from the radar sample by scenario and number of persons"""
    title_radar_sample = f"Radar Sample, {number_persons} persons, {scenario} scenario"
    title_plot_signal = f"Received Signal, {number_persons} persons in the radar range, {scenario} scenario"
    visualize_radar_samples(data, title_radar_sample)
    plot_radar_signal(data[100, :], title=title_plot_signal)


if __name__ == "__main__":
    database_config = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'UWB_Radar_Samples',
    'user': 'cristina',
    'password': ''
    }
    tables ={
            "scenario_1": ["people_walking_5m_area", "people walking in 5m area"],
            "scenario_2" :["people_standing_queue_0_15", "people standing in a queue"],
            "scenario_3":  ["density_3_m2_11_20", "people walking in a room with 3 persons per m2"],
            "scenario_4": ["density_4_m2_11_20" , "people walking in a room with 4 persons per m2",],
            "processed_data": ["processed_data", "pca features extracted from radar samples"]
            }
    table_name = tables["processed_data"][0]
    scenario =  tables["processed_data"][1]
    number_persons = 105
    shape = (200, 50)
    data = extract_data_per_no_persons(table_name=table_name, detailed_label=number_persons, database_config=database_config)
    data = np.array(data)
    data_res = np.reshape(data, shape)
    # data_prep = process_data_pipeline(data_res, fs=39*1e9)
    visualize_radar_samples_by_scenario(data=data_res, scenario=scenario, number_persons=number_persons)


