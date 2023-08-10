from tools.postgresql_operations import read_table_postgresql
import matplotlib.pyplot as plt
from tools.process_data import process_data_pipeline
import numpy as np 

def plot_radar_signal(data, title):
    fig = plt.figure(figsize = (10, 5))
    plt.plot(data)
    plt.xlabel('# Sample (1280 samples = 5 meters range)')
    plt.ylabel("Amplitude [V]")
    plt.title(f"{title}")
    plt.savefig("./{}".format(title))
    plt.show()

def visualize_radar_samples(data, title):
    fig = plt.figure(figsize = (10, 3))
    plt.imshow(data,interpolation="none")
    plt.xlabel('# Sample (1280 samples = 5 meters range)')
    plt.ylabel("Time (200 samples = 1.25 seconds)")
    plt.title(f"{title}")
    plt.savefig("./{}".format(title))
    plt.show()

def visualize_radar_samples_by_scenario(scenario, table_name, number_persons, database_config):
    query = """SELECT radar_sample FROM PUBLIC.{} WHERE number_persons = '{}' LIMIT 1;""".format(table_name, number_persons)
    headers, data = read_table_postgresql(table_name=table_name,database_config= database_config, limit = 1, query = query)
    title_radar_sample = f"Radar Sample, {number_persons} persons, {scenario} scenario"
    title_plot_signal = f"Received Signal, {number_persons} persons in the radar range, {scenario} scenario"
    data = np.array(data)
    data_res = np.reshape(data, (200,1280))
    data_prep = process_data_pipeline(data_res, fs=39*1e9)
    visualize_radar_samples(data_prep, title_radar_sample)
    plot_radar_signal(data_prep[100, :], title=title_plot_signal)