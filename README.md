# PEOPLE COUNTING IN A COVID-19 AND GDPR CONTEXT USING IR-UWB RADAR SIGNALS

**PURPOSE AND CONTEXT**

This repository presents a method for people counting using IR-UWB radar signals, in the COVID-19 and GDPR context. 
The purpose is the monitoring of the number of people  inside a room, where a one-meter minimum distance between the persons is required for limiting their number in a given area, which is important to limit the spread of the COVID-19 virus. The GDPR context refers to the protection of personal data, where an IR-UWB radar is used instead of a video camera to count the people indoors. The radar range is 5 meters.
The dataset is open-source and corresponds to the reference article [1] , where four scenarios are considered, involving 0 up to 20 persons randomly walking and standing in a queue. 

![image](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/df374e21-7e99-42e4-b628-80d5b97cb697)

**Ultra-wideband Technology**

Ultra-wideband (UWB) impulses have a duration of nanoseconds in the time domain, occupying a very large frequency bandwidth, from 500 MHz up to 7.5 GHz.
The effective bandwidth for the radar samples set is 5.65 GHz - 7.95 GHz.
UWB technology is used for surveillance, detection, positioning and other applications due to its fine temporal resolution and low emission power. 

**DATA ENGINERRING**

The raw data is formatted to JSON files and then stored in the PostgreSQL database.

The steps for data processing are:

- zero-mean centering

- clutter removal by applying the Running Average method 

- filtering in the 5.65 GHz - 7.95 GHz band to remove unwanted frequency components
  
![Radar Sample, 10 persons, people standing in a queue scenario](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/4ffa7e10-acc2-4842-bb7c-5137daa8c10b)

![Received Signal, 10 persons in the radar range, people standing in a queue scenario](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/e7b6dc47-52be-4199-a995-ebe3768f99ca)

![Radar Sample, 20 persons, people walking in a room with 4 persons per m2 scenario](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/85d731ec-c60f-424f-ab3e-a27bb0fd4604)

![Received Signal, 20 persons in the radar range, people walking in a room with 4 persons per m2 scenario](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/dad84559-0f5e-40b0-9a50-a8396d515ace)

**FEATURE ENGINEERING**

To reduce data dimennsionality and to keep the useful information in the same time, Principal Component Analysis method is applied on the pre-processed data.  The number of principal components that are kept is 50. 

![Radar Sample, 20 persons, in the last scenario](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/02b1a34c-dc60-4569-8fdc-4bae5d390b59)

![Received Signal, 20 persons in the radar range, in the last scenario](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/7de4865c-1f9b-4e29-8cb2-c15bba56fb4f)

**TRAINING AND EVALUATING THE MODEL**

**ONLY CONSIDERING THE SCENARIO**

The dataset is split in training data, validation data and test data. The data are scaled between [0,1], by using the mean and standard deviation of the training data. A CNN is used to classify the radar samples into 4 classes which represent the scenarios given in the table above. Multiple experiments are run for finding a set of good hyperparameters. The hyperparameters are: batch_size = 8, learning_rate = 0.0001, number_of_epochs = 15, using early stopping by monitoring the training loss. The network's layers and the number of parameters on each layer, plus the accuracy results are given in the following snapshot:

![image](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/aab3c140-482a-4297-8563-b68091abc9a4)
![Screenshot from 2023-08-15 17-25-56](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/5d406076-186f-4003-b9a9-12da6874d0eb)

**CONSIDERING THE NUMBER OF PERSONS**

For the same data but labelled by the number of persons in the radar range, and  the same network's architecture, the hyperparameters in this case are: batch_size = 4, number of epochs = 40, learning rate = 0.0001. In the figure you will find the accuracies and the network's layers:


![image](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/f56ffa21-3773-437c-9a42-2e16c4eb99f8)


**Shallow Deployment**

This model is designed for a local machine, in the indoor location of the radar. Although, it sends an e-mail notification every 10 minutes to predict the status of the room (it implies that a screening of the radar is done every 10 minutes, and the file containing the received signals' values sent to the same personal computer).

The final prediction template resembles to: 
![Screenshot from 2023-08-17 22-25-22](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/37d75e7e-0084-4382-8dcd-496ec79dfdbd)


where first the scenario is predicted by the first model, and then the number of persons is predicted by the second model.
The final prediction function being "predict_function" in the prediction_function.py file. 


**DRAWBACKS**

1. One drawback is the class imbalance, which may cause a problem when trying to make predictions for classes that have fewer samples. A solution is to augment the existing data, meaning generating new radar samples, either by adding noise, by using a Generative Adversial Network, or another method. For example, by using only the number of scenarios as the number of classes, the distribution of the number of samples for each class is the following:

![labels_distribution](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/9a2a2b10-1f4d-42ea-bdf0-ee040e421501)

also, for the number of people in each scenario:

![Screenshot from 2023-08-15 17-57-43](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/8ead450e-3008-4802-aecd-f121fff1186b)



[1] 	X. Yang, W. Yin, L. Li and L. Zhang, "Dense People Counting Using IR-UWB Radar with a Hybrid Feature Extraction Method," IEEE Geoscience and Remote Sensing Letters, vol. 16, no. 1, pp. 30-34, 2018. 





