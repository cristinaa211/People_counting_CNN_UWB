
from tools.read_files import load_float_values
from tools.process_data import process_data_pipeline, apply_pca
from tools.model_tools import CNNModelDeployment
from tools.create_json_files import read_json_file
import numpy as np 
import torch
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

label_dict_people = {}
for i in range(21):
    label_dict_people.update({i: f"{i} persons in the radar range"})

device = "cuda" if torch.cuda.is_available() else "cpu"

def send_email(subject, body, 
               from_email = "", to_email = "", smtp_password = "", 
               smtp_server= "smtp.gmail.com", smtp_port=465):
    """ Create a MIMEText object to represent the email content
    Sends an email with the predictions
    Args:
        subject : the subject of the e-mail
        body    : the text content of the e-mail
        from_email  : the sender
        to_email    : the recipient
        smtp_password   : the """
    message = MIMEMultipart()
    message['From'] = from_email
    message['To'] = to_email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))
    with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
        server.login(from_email, smtp_password)
        server.sendmail(from_email, to_email, message.as_string())

def process_data(filename):
    float_data = load_float_values(filename)
    processed_data = process_data_pipeline(float_data)
    pca_features = apply_pca(processed_data)
    return pca_features


def predict_number_persons(pca_features, model_name):
    info_file = f"./models/{model_name}/info.json"
    model_file = f"./models/{model_name}/{model_name}.pkl"
    infos = read_json_file(info_file)
    mean, std = infos['mean'], infos['std']
    # num_classes = infos["num_classes"]
    norm_data =  (pca_features - mean) / std 
    norm_data = np.expand_dims(norm_data, 0)
    tensor_data = torch.tensor(norm_data, dtype = torch.float32).to(device)
    model_depl = CNNModelDeployment(model_file, num_classes = 21)
    prediction = model_depl.predict(tensor_data).cpu().item()
    prediction_ = label_dict_people[prediction]
    return prediction_


def predict_scenario(pca_features, model_name):
    info_file = f"./models/{model_name}/info.json"
    model_file = f"./models/{model_name}/{model_name}.pkl"
    label_dict = {
                0: '0-10 people walking',
                1 : '0-15 people standing in a queue with an average of 10 cm distance',
                2 : '11-20 people walking with a density of 3 pers/m2',
                3 : '11-20 people walking with a density of 4 pers/m2'
                }
    infos = read_json_file(info_file)
    mean, std = infos['mean'], infos['std']
    # num_classes = infos["num_classes"]
    norm_data =  (pca_features - mean) / std 
    norm_data = np.expand_dims(norm_data, 0)
    tensor_data = torch.tensor(norm_data, dtype = torch.float32).to(device)
    model_depl = CNNModelDeployment(model_file, num_classes = 4)
    prediction = model_depl.predict(tensor_data).cpu().item()
    prediction_ = label_dict[prediction]
    return prediction_

def predict_function(filename,recipient_email, model_name_scenario = "cnn_norm_v3", model_name_people = "cnn_detailed_label_v1"):
    """Predicts the sceario and the number of people in the radar range and sends an email notification to the responsible of the indoor location.
    Args:
        filename            : string which represents the file with the values of the radar sample sample
        recipient_email     : the recipient of the email notification
        model_name_scenario : the model for predicting the scenario
        model_name_people   : the model for preidictig the number of persons in the radar range
    """
    pca_features  = process_data(filename)
    prediction_scenario= predict_scenario(pca_features, model_name_scenario)
    prediction_persons = predict_number_persons(pca_features, model_name_people)
    email_text = "There are {} and actually {}.".format(prediction_scenario, prediction_persons)
    print(email_text)
    send_email(subject = "Status of the people in the radar range", body = email_text, to_email=recipient_email)


   
