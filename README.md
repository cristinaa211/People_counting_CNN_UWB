# PEOPLE COUNTING IN A COVID-19 AND GDPR CONTEXT USING IR-UWB RADAR SIGNALS

**PURPOSE AND CONTEXT**

This repository presents a method for people counting using IR-UWB radar signals, in the COVID-19 and GDPR context. 
The purpose is the monitoring of the number of people  inside a room, where a one-meter minimum distance between the persons is required for limiting their number in a given area, which is important to limit the spread of the COVID-19 virus. The GDPR context refers to the protection of personal data, where an IR-UWB radar is used instead of a video camera to count the people indoors. The radar range is 5 meters.
The dataset is open-source and corresponds to the reference article [1] , where four scenarios are considered, involving 0 up to 20 persons randomly walking and standing in a queue. 
![Screenshot from 2023-08-07 21-41-42](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/a8e5f92b-e47e-4fc0-a73c-df90b46682b7)

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
  ![Radar Sample, 15 persons, people walking in a room with 3 persons per m2 scenario](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/93a656f3-946c-4d25-82fe-0837206968f6)

![Received Signal, 15 persons in the radar range, people walking in a room with 3 persons per m2 scenario](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/d1459e8d-578e-4208-b046-da327a0fb70f)


**FEATURE ENGINEERING**

To reduce data dimennsionality and to keep the useful information in the same time, Principal Component Analysis method is applied on the pre-processed data.  The number of principal components that are kept is 50. 
![Radar Sample, 10 persons, in the first scenario](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/75cbdd2c-ec06-4d2f-a210-e503377555bf)

![Received Signal, 10 persons in the radar range, in the first scenario](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/c2ea85f4-4669-44a5-8227-4a76e540c595)


**TRAINING AND EVALUATING THE MODEL**

**ONLY CONSIDERING THE SCENARIO**

The dataset is split in training data, validation data and test data. The data are scaled between [0,1], by using the mean and standard deviation of the training data. A CNN is used to classify the radar samples into 4 classes which represent the scenarios given in the table above. Multiple experiments are run for finding a set of good hyperparameters. The hyperparameters are: batch_size = 8, learning_rate = 0.0001, number_of_epochs = 15, using early stopping by monitoring the training loss. The network's layers and the number of parameters on each layer, plus the accuracy results are given in the following snapshot:
![Screenshot from 2023-08-15 17-35-40](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/3e1dc1ec-64c3-4eb9-a4d8-939cf72719b2)
![Screenshot from 2023-08-15 17-25-56](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/8a5a7eb8-5a01-4589-b14b-93d8f09a3a5a)

**CONSIDERING THE NUMBER OF PERSONS**

For the same data but labelled by the number of persons in the radar range, and  the same network's architecture, the hyperparameters in this case are: batch_size = 4, number of epochs = 40, learning rate = 0.0001. In the figure you will find the accuracies and the network's layers:

![Screenshot from 2023-08-15 19-14-31](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/b3a44de4-f61b-4850-9f95-4fba97ef9674)


**Shallow Deployment**

This model is designed for a local machine, in the indoor location of the radar. Although, it sends an e-mail notification every 10 minutes to predict the status of the room (it implies that a screening of the radar is done every 10 minutes, and the file containing the received signals' values sent to the same personal computer).

The final prediction template resembles to: 
![Screenshot from 2023-08-17 22-25-22](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/6ea51747-22a4-4787-8760-707e70fe4f4b)


where first the scenario is predicted by the first model, and then the number of persons is predicted by the second model.
The final prediction function being "predict_function" in the prediction_function.py file. 


**DRAWBACKS**

1. One drawback is the class imbalance, which may cause a problem when trying to make predictions for classes that have fewer samples. A solution is to augment the existing data, meaning generating new radar samples, either by adding noise, by using a Generative Adversial Network, or another method. For example, by using only the number of scenarios as the number of classes, the distribution of the number of samples for each class is the following:
![labels_distribution](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/318e73c3-005a-449b-9b0b-a51bc998e9c1)


also, for the number of people in each scenario:

![Screenshot from 2023-08-15 17-57-43](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/98fbe6e8-7c82-423a-b939-160062a8b2b7)



[1] 	X. Yang, W. Yin, L. Li and L. Zhang, "Dense People Counting Using IR-UWB Radar with a Hybrid Feature Extraction Method," IEEE Geoscience and Remote Sensing Letters, vol. 16, no. 1, pp. 30-34, 2018. 





