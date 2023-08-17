from tools.prediction_function import predict_function
import schedule
import time



def main():
    model_name = "cnn_norm_v3"
    model_name_2 = "cnn_detailed_label_v1"
    filename = ""
    email_to = "something@gmail.com"
    predict_function(filename, email_to, model_name,  model_name_2)
    schedule.every(1).minutes.do(predict_function, filename, email_to)
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()