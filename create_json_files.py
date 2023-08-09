from tools.read_files import read_files_multiple_directories
import os 
import json
import numpy as np

def read_txt_file_data(filename):
    """Reads the content of a text file and returns the content as a string"""
    data = []
    with open(filename) as input_file:
        lines = input_file.readlines() 
        for line in lines:
            data.append(line)
    return data 

def load_float_values(file_path):
    """Returns the content of a text file as a list of float values"""
    try:
        float_values = np.loadtxt(file_path)
        return float_values
    except Exception as e:
        print(f"Error: Unable to load float values from '{file_path}'.")
        print(f"Details: {str(e)}")
        return []

def create_json_file(content_list, file_name):
    """Creates a json file having as content "content_list" provided, which can be a list of dictionaries"""
    with open(f"{file_name}.json", 'w') as json_file:
        json.dump(content_list, json_file, indent = 4)

def read_files_dump_to_json(directory, json_file_name, label):
    """Provides the content of multiple files in the "directory" as lists of float values, 
    creates a json file with the metainformation
    and creates a json file which is saved in the current directory. 
    The JSON file contains the following information about each radar sample:
    label           : the scenario (Example label = 100 points to the first scenario)
    detailed_label  : contains the scenario given by the first digit and the second and third 
                        digits represent the number of the persons in the radar sample
    context         : gives details of the scenario
    radar_sample    : contains a list of the values in the radar sample
    shape           : represents the shape of the radar sample
    type            : represents the type of the radar_sample """
    files_list = read_files_multiple_directories(directory)
    json_content = []
    for file in files_list:
        context = os.path.split(os.path.split((os.path.split(file)[0]))[0])[1]
        no_persons = os.path.split((os.path.split(file)[0]))[1]
        file_content = load_float_values(file)
        matrix_dimension = file_content.shape
        file_type = type(file_content)
        label_detailed = label + int(no_persons)
        json_content_file = {'label' : label , 'detailed_label':label_detailed ,
                            'context' : context , 'number_persons' : no_persons, 
                            'radar_sample' : file_content.tolist() ,
                            'shape' : list(matrix_dimension), 'type' : str(file_type)}
        json_content.append(json_content_file)
    create_json_file(json_content, json_file_name)

def read_json_file(json_file_path):
    """Returns the content of a JSON file"""
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    return data 


if __name__ == "__main__":
    read_files_dump_to_json(directory="./IR-UWB-Radar-Signal-Dataset-for-Dense-People-Counting-master/0-10 people walking in a 5 m area", json_file_name="people_0-10_walking_5m_area", label = 100)
    read_files_dump_to_json(directory="./IR-UWB-Radar-Signal-Dataset-for-Dense-People-Counting-master/0-15 people standing in a queue", json_file_name="0-15_people_standing_in_a_queue", label = 200)
    read_files_dump_to_json(directory="./IR-UWB-Radar-Signal-Dataset-for-Dense-People-Counting-master/11-20 people on a density of 3 persons per m2", json_file_name="11-20_people_on_a_density_of_3_persons_per_m2", label = 300)
    read_files_dump_to_json(directory="./IR-UWB-Radar-Signal-Dataset-for-Dense-People-Counting-master/11-20 people on a density of 4 persons per m2", json_file_name="11-20_people_on_a_density_of_4_persons_per_m2", label = 400)
