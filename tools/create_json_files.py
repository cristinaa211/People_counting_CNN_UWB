from tools.read_files import read_files_multiple_directories,load_float_values
import os 
import json


def create_json_file(content_list, file_name):
    """Creates a json file having as content "content_list" provided, which can be a list of dictionaries"""
    with open(f"{file_name}.json", 'w') as json_file:
        json.dump(content_list, json_file, indent = 4)

def read_files_dump_to_json(directory, json_file_name, label, create_two = False):
    """Provides the content of multiple files in the "directory" as lists of float values, 
    creates a json file with the metainformation
    and creates a json file (or two files, depending on the total space occupied) 
    which is saved in the current directory. 
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
    if create_two == False:
        create_json_file(json_content, json_file_name)
    else:
        half_indx = int(len(json_content)/2) if len(json_content) % 2 == 0 else int((len(json_content)+1)/2)
        create_json_file(json_content[0:half_indx], f"{json_file_name}_1")
        create_json_file(json_content[half_indx+1:], f"{json_file_name}_2")



def read_json_file(json_file_path):
    """Returns the content of a JSON file"""
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    return data 
