# CNN-for-people-counting-

PEOPLE COUNTING IN A COVID-19 AND GDPR CONTEXT, USING AN IR-UWB RADAR, BASED ON ARTIFICIAL INTELLIGENCE ALGORITHMS

Cristina POPOVICI1, Emanuel RĂDOI2, Leontin TUȚĂ1
1 Military Technical Academy ”Ferdinand I”, Bucharest, Romania, cristina.popovici@mta.ro, leontin.tuta@mta.ro
2 University of Western Brittany, Brest, France, emanuel.radoi@univ-brest.fr



Abstract

People counting using an IR-UWB radar, based on Artificial Intelligence classification algorithms, is investigated in this paper. Ultra-wideband (UWB) impulses have a duration of nanoseconds in the time domain, occupying a very large frequency bandwidth, from 500 MHz up to 7.5 GHz [1]. UWB technology is used for surveillance, detection, positioning and other applications due to its fine temporal resolution and low emission power [2]. Ultra-wideband echo radar signals are used for people counting, in a COVID-19 and GDPR context, where a one-meter minimum distance between the persons is required, thus limiting their number in a given area, which is important to limit the spread of the COVID-19 virus. The GDPR context refers to the protection of personal data, where an IR-UWB radar is used instead of a video camera to count the people indoors. The dataset is open-source and corresponds to the reference article [3], where four scenarios are considered, involving 0 up to 20 persons randomly walking and standing in a queue, in the radar range. The data are pre-processed, by extracting the direct current component and by applying the Running Average method for clutter removal. Three methods for feature extraction are investigated: (i) the Curvelet Transform together with the Segmented-based method (the hybrid method) [3], (ii) a Convolutional Neural Network (CNN) on the raw dataset, and (iii) the Principal Component Analysis (PCA) algorithm, both for the raw dataset and the set of the extracted hybrid features. The classification is done by using Artificial Intelligence algorithms such as K-nearest neighbours (KNN), Support Vector Machine (SVM), Multilayer Perceptron Neural Network (MLP), and Convolutional Neural Network, where the targets are the number of persons in the radar range. K-NN uses 5 neighbours and is based on Euclidian distances, SVM uses the RBF kernel, MLP is composed of four layers of 50 neurons each, and the CNN is composed of Convolution layers, MaxPooling layers, LazyLinear layers, having as activation function the ReLu function and as logistic function, the SoftMax function. The input data for the classifiers are split into a training set and a test set. The algorithms’ performance is provided in terms of accuracy, precision, recall and f1-score functions, using the test dataset. The paper presents three methods for feature extraction used as input data for AI-supervised algorithms. The results show that the best performance is obtained for the hybrid features based on MLP (the frequency-based features provided by the Curvelet transform and the time-based features provided by the Segmented-based method), where the f1-score is 99.85%. The frequency-based features provide us with salient detail and capture the edges in the radar sample, while the Segmented-based features provide us with resolution in distance. The drawback of this algorithm would be the high computational demand and its complexity. CNN applied on the set of features that resulted from applying PCA on the raw dataset is recommended for real-time applications, which shows an f1-score of 95.41%. CNN has 800 parameters which are optimized during the training process, where the parameters are updated based on minimizing the error between the estimated value and the target value, using the Minimum Squared Error loss function with one-hot encoding targets and the Adam optimizer. The third method provides fast training and testing, but the data insufficiency can degrade the performance if the dataset is not large enough. The table below shows the algorithms’ performance in terms of accuracy, precision, recall and f1-score, which are the most used classification metrics. 

Table 1. People counting results
Dataset
(No_signals x No_features)	Feature exraction method	Classification Algorithm	Accuracy	Precision	Recall	F1-score
(6400 x 293)00	Hybrid	K-NN	94.08%.8%	94.08%.	94.08%.	94.08%.
(6400 x 293)	Hybrid	SVM	85.12%	85.12%	85.12%	82.76%
(6400 x 293)	Hybrid	MLP	99.85%	99.85%	99.85%	99.85%
(6400 x 3)	Hybrid + PCA	K-NN	73.66%	73.66%	73.66%	73.56%
(6400 x 3)	Hybrid + PCA	SVM	55.74%	55.74%	55.74%	53.43%
(6400 x 3)	Hybrid + PCA	MLP	64.31%	64.31%	64.31%	62.90%
(6400 x 1280)	CNN	CNN	95.44%	95.44%	95.44%	95.44%
(6400 x 6)	PCA	CNN	95.41%	95.41%	95.41%	95.41%





Bibliography

[1] 	T. C. a. G. P. M. I. Marco Cavallaro, "Gaussian Pulse Generator for Millimeter-Wave," IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS—I: REGULAR PAPERS, JUNE , Vols. NO. 6,VOL. 57, 2010. DOI: 10.1109/TCSI.2009.2031706
[2] 	Y. Rahayu, T. A. Rahman, R. Ngah and P. S. Hall, "Ultra wideband technology and its applications," 2008 5th IFIP International Conference on Wireless and Optical Communications Networks (WOCN '08), 2008, pp. 1-5, doi: 10.1109/WOCN.2008.4542537. 

[3] 	X. Yang, W. Yin, L. Li and L. Zhang, "Dense People Counting Using IR-UWB Radar with a Hybrid Feature Extraction Method," IEEE Geoscience and Remote Sensing Letters, vol. 16, no. 1, pp. 30-34, 2018. 



