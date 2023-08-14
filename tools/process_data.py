import pandas as pd
from pyspark import SparkContext, types
from pyspark.sql import SQLContext, Row, SparkSession
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from pyspark.sql import SparkSession, types

def process_data_pipeline(sig, fs):
    """Returns processed radar sample"""
    data_1 = extract_dc_component(sig)
    data_2 = moving_average_clutter_removal_2d(data_1, 5)
    data_3 = butter_bandpass_filter(data_2, 5.65*1e9, 7.95*1e9, fs)
    return data_3

def apply_pca(data, n_components):
    """Apllies Principal Component Analysis on the data and return n_components principal components"""
    pca_data = PCA(n_components = n_components).fit_transform(data)
    return pca_data

def moving_average_clutter_removal_1d(signal, window_size):
    """
    Perform clutter removal using moving average subtraction.
    Parameters:
    signal (numpy array): The input radar signal.
    window_size (int): Size of the moving average window.
    Returns:
    numpy array: The clutter-removed radar signal.
    """
    # Calculate the cumulative sum of the signal
    cumulative_sum = np.cumsum(signal)
    # Calculate the moving average by differencing the cumulative sum
    moving_average = np.zeros_like(signal)
    moving_average[window_size:] = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    # Subtract the moving average from the original signal
    clutter_removed_signal = signal - moving_average
    return clutter_removed_signal


def moving_average_clutter_removal_2d(signal, window_size):
    """
    Perform clutter removal using moving average subtraction along columns.
    Parameters:
    signal (numpy array): The input 2D radar signal with shape (num_rows, num_columns).
    window_size (int): Size of the moving average window along the columns.
    Returns:
    numpy array: The clutter-removed 2D radar signal.
    """
    num_rows, num_columns = signal.shape
    clutter_removed_signal = np.zeros_like(signal)
    for row in range(num_rows):
        # Calculate the cumulative sum along the columns
        cumulative_sum = np.cumsum(signal[row, :])
        # Calculate the moving average by differencing the cumulative sum
        moving_average = np.zeros(num_columns)
        moving_average[window_size:] = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
        # Subtract the moving average from the original signal
        clutter_removed_signal[row, :] = signal[row, :] - moving_average
    return clutter_removed_signal



def butter_bandpass(lowcut, highcut, fs, order=5):
    """Returns the coefficients of a butter bandpass filter"""
    return signal.butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, sampling_rate, order=5):
    """Applies a butter bandpass filter on the signal"""
    b, a = butter_bandpass(lowcut, highcut, sampling_rate, order=order)
    return signal.lfilter(b, a, data)

def compute_bandwidth(sig_, sampling_rate, frequency_range=None):
    """
    Compute the Power Spectral Density (PSD) of a signal in dBm.
    Parameters:
    signal (numpy array): The input signal.
    sampling_rate (float): The sampling rate in Hz.
    frequency_range (tuple, optional): Frequency range of interest in Hz (start_freq, end_freq).
                                      If not provided, the entire frequency range of the signal is used.
    Returns:
    bandwidth : the effective signal bandwidth
    """
    # Compute the PSD
    psd = np.abs(np.fft.rfft(sig_))**2 / len(sig_)
    # Frequencies corresponding to the PSD
    frequencies = np.fft.rfftfreq(len(sig_), 1 / sampling_rate)*1e-9
    # Convert the PSD to dBm/MHz
    psd_dbm = 10 * np.log10(psd/1000) 
    if frequency_range is not None:
        start_freq, end_freq = frequency_range
        freq_mask = (frequencies >= start_freq) & (frequencies <= end_freq)
        frequencies = frequencies[freq_mask]
        psd_dbm = psd_dbm[freq_mask]
    threshold = 0.38*np.min(psd_dbm)
    psd_dbm_m = [(x, f)  for x, f in zip(psd_dbm, frequencies) if x > threshold ]
    psd_v, freq  = list(zip(*psd_dbm_m))
    plt.figure(figsize=(8, 6))
    plt.plot(list(freq), list(psd_v))
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Power Spectral Density (dBm/MHz)')
    plt.title("Power Spectral Density")
    plt.grid(True)
    plt.savefig("./psd_signal.png")
    plt.show()
    bandwidth = [min(freq), max(freq)]
    return bandwidth

def extract_dc_component(sig):
    """Extracts the direct current component from the signal"""
    sig_prep = sig - np.mean(sig)
    return sig_prep

def spark_read_json(json_file_path):
    spark = SparkSession.builder.master("local").appName("Read_Json_File")\
            .getOrCreate()
    df = spark.read.json(json_file_path, multiLine=True)
    df.show(truncate = 0)

def test_spark():
    spark = SparkSession.builder \
        .master("local[1]") \
        .appName("SparkByExamples.com") \
        .getOrCreate() 
    data = [('James','','Smith','1991-04-01','M',3000),
    ('Michael','Rose','','2000-05-19','M',4000),
    ('Robert','','Williams','1978-09-05','M',4000),
    ('Maria','Anne','Jones','1967-12-01','F',4000),
    ('Jen','Mary','Brown','1980-02-17','F',-1)
    ]

    columns = ["firstname","middlename","lastname","dob","gender","salary"]
    df = spark.createDataFrame(data=data, schema = columns)
    df.show()

def read_text_file(filename):
    """Read json file as a text file with pyspark"""
    spark = SparkSession.builder.master("local").appName("Read_Json_File")\
            .getOrCreate()
    lines = spark.sparkContext.textFile(filename)
    processed_lines = lines.map(lambda line: line + "," if not line.endswith("}") else line)
    json_array = "[" + processed_lines.reduce(lambda a, b: a + b) + "]"
    json_rdd = spark.sparkContext.parallelize([json_array])
    df = spark.read.json(json_rdd)
    output_df = df.select("label", "radar_sample")
    output_df.show()


def to_pyspark_dataframe(data, columns):
    """Creates a spark dataframe from data, having as columns the columns argument"""
    spark = SparkSession.builder.appName("SaveToSparkDF").getOrCreate()
    schema = types.StructType([
                types.StructField(columns[0], types.StringType(), True),
                types.StructField(columns[1], types.StringType(), True),
                types.StructField(columns[2], types.ArrayType(types.ArrayType(types.DoubleType(), True), True), True)        ])
    df = spark.createDataFrame(data, schema = schema)
    df.printSchema()
    return df

