import pandas as pd
from create_json_files import read_json_file
from pyspark import SparkContext, types
from pyspark.sql import SQLContext, Row, SparkSession


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
    spark = SparkSession.builder.master("local").appName("Read_Json_File")\
            .getOrCreate()
    lines = spark.sparkContext.textFile(filename)
    processed_lines = lines.map(lambda line: line + "," if not line.endswith("}") else line)
    json_array = "[" + processed_lines.reduce(lambda a, b: a + b) + "]"
    json_rdd = spark.sparkContext.parallelize([json_array])
    df = spark.read.json(json_rdd)
    output_df = df.select("label", "radar_sample")
    output_df.show()



if __name__ == "__main__":
    spark = SparkSession.builder.master("local").appName("Read_Json_File")\
            .getOrCreate()
