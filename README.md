# PEOPLE COUNTING IN A COVID-19 AND GDPR CONTEXT, USING AN IR-UWB RADAR, BASED ON ARTIFICIAL INTELLIGENCE ALGORITHMS


This repository presents a method for people counting using an IR-UWB radar signals, based on Artificial Intelligence classification algorithms. 
Ultra-wideband (UWB) impulses have a duration of nanoseconds in the time domain, occupying a very large frequency bandwidth, from 500 MHz up to 7.5 GHz.
The effective bandwidth for the radar samples set is 5.65 GHz - 7.95 GHz.
UWB technology is used for surveillance, detection, positioning and other applications due to its fine temporal resolution and low emission power. 
Ultra-wideband echo radar signals are used for people counting, in a COVID-19 and GDPR context, where a one-meter minimum distance between the persons is required, thus limiting their number in a given area, which is important to limit the spread of the COVID-19 virus. 
The GDPR context refers to the protection of personal data, where an IR-UWB radar is used instead of a video camera to count the people indoors. The dataset is open-source and corresponds to the reference article [1] , where four scenarios are considered, involving 0 up to 20 persons randomly walking and standing in a queue, in the radar range. 

![image](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/df374e21-7e99-42e4-b628-80d5b97cb697)

The data are pre-processed, by extracting the direct current component, by applying the Running Average method for clutter removal for bandpass filtering in the 5.65 GHz - 7.95 GHz band.

![Radar Sample, 10 persons, people standing in a queue scenario](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/4ffa7e10-acc2-4842-bb7c-5137daa8c10b)

![Received Signal, 10 persons in the radar range, people standing in a queue scenario](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/e7b6dc47-52be-4199-a995-ebe3768f99ca)

![Radar Sample, 20 persons, people walking in a room with 4 persons per m2 scenario](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/85d731ec-c60f-424f-ab3e-a27bb0fd4604)

![Received Signal, 20 persons in the radar range, people walking in a room with 4 persons per m2 scenario](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/dad84559-0f5e-40b0-9a50-a8396d515ace)


Three methods for feature extraction are investigated: 
(i) the Curvelet Transform together with the Segmented-based method (the hybrid method),
(ii) a Convolutional Neural Network (CNN) on the raw dataset,
(iii) the Principal Component Analysis (PCA) algorithm, both for the raw dataset and the set of the extracted hybrid features.
The classification is done by using algorithms such as K-nearest neighbours (KNN), Support Vector Machine (SVM), Multilayer Perceptron Neural Network (MLP), and Convolutional Neural Network, where the targets are the number of persons in the radar range, based on an open-source database. 
The results show that the best performance is obtained by the hybrid features based on MLP (the frequency-based features provided by the Curvelet transform and the time-based features provided by the Segmented-based method). 
The frequency-based features provide us with salient detail and capture the edges in the radar sample, while the Segmented-based features provide us with resolution in distance. 
The drawback of this algorithm would be the high computational demand and its complexity. 
CNN applied on the set of features that resulted from applying PCA on the raw dataset is recommended for real-time applications. 


![image](https://github.com/cristinaa211/People_counting_CNN_UWB/assets/61435903/1a9fe4e6-3ca5-4ed5-835c-affa57b2930f)


[1] 	X. Yang, W. Yin, L. Li and L. Zhang, "Dense People Counting Using IR-UWB Radar with a Hybrid Feature Extraction Method," IEEE Geoscience and Remote Sensing Letters, vol. 16, no. 1, pp. 30-34, 2018. 





