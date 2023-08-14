from tools.create_json_files import read_json_file
import psycopg2
import json
import numpy as np

def read_table_postgresql(columns = None, table_name = None, database_config = None, 
                          limit = None, query = None):
    conn = psycopg2.connect(
        host=database_config['host'],
        port=database_config['port'],
        dbname=database_config['dbname'],
        user=database_config['user'],
        password=database_config['password']
    )    
    cursor = conn.cursor()
    if query == None:
        if limit : query = 'SELECT {} FROM PUBLIC."{}" LIMIT {}'.format(columns, table_name, int(limit))
        else: query = 'SELECT {} FROM PUBLIC."{}"'.format(columns, table_name)
    cursor.execute(query)
    data = cursor.fetchall()
    headers = [i[0] for i in cursor.description]
    print('Data fetched successfully')
    if conn:
        cursor.close()
        conn.close()
        print("PostgreSQL connection is closed")
    return headers, data


def  import_processed_data_to_postgresql(data, table_name, database_config):
    """Imports data into table_name in the database
    data            : a list of lists containing label, detailed label and the processed radar sample values
    table_name      : a string reffering to the table name in the database
    database_config : the database configuration """
    conn = psycopg2.connect(
        host=database_config['host'],
        port=database_config['port'],
        dbname=database_config['dbname'],
        user=database_config['user'],
        password=database_config['password']
    )
    query_table = f"""CREATE TABLE IF NOT EXISTS {table_name} (
    id SERIAL PRIMARY KEY,
    label VARCHAR,
    detailed_label VARCHAR,
    radar_sample JSONB
    )"""
    query = f"""INSERT INTO {table_name} 
            (label, detailed_label, radar_sample)
            VALUES (%s, %s, %s)"""
    try:
        with conn.cursor() as cur:
            cur.execute(query_table)
            for record in data:
                try:
                    cur.execute(query, (record['label'],record['detailed_label'], json.dumps(record['radar_sample'])))
                except:
                    cur.execute(query, (record[0],record[1], json.dumps(record[2])))
            conn.commit()
        print("Data inserted successfully!")
    except psycopg2.Error as e:
        print(f"Error: {e}")
    finally:
        cur.close()
        conn.close()


def import_data_to_postgresql(data, table_name, database_config):
    conn = psycopg2.connect(
        host=database_config['host'],
        port=database_config['port'],
        dbname=database_config['dbname'],
        user=database_config['user'],
        password=database_config['password']
    )
    query_table = f"""CREATE TABLE IF NOT EXISTS {table_name} (
    id SERIAL PRIMARY KEY,
    label VARCHAR,
    detailed_label VARCHAR,
    context VARCHAR ,
    number_persons VARCHAR,
    radar_sample JSONB,
    shape INTEGER[],
    type VARCHAR
    )"""
    query = f"""INSERT INTO {table_name} 
            (label, detailed_label, context, number_persons, radar_sample, shape, type)
            VALUES (%s, %s, %s, %s, %s, %s, %s)"""
    try:
        with conn.cursor() as cur:
            cur.execute(query_table)
            for record in data:
                cur.execute(query, (record['label'],record['detailed_label'],
                                    record['context'],record['number_persons'],
                                    json.dumps(record['radar_sample']), record['shape'], record['type']))
            conn.commit()
        print("Data inserted successfully!")
    except psycopg2.Error as e:
        print(f"Error: {e}")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    database_config = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'UWB_Radar_Samples',
    'user': 'cristina',
    'password': ''
    }
    json_file = ["0-15_people_standing_in_a_queue_1.json", "0-15_people_standing_in_a_queue_2.json", 
                 "11-20_people_on_a_density_of_3_persons_per_m2.json", 
                 "11-20_people_on_a_density_of_4_persons_per_m2.json"]
    table_name = ["people_standing_queue_0_15", "density_3_m2_11_20", "density_4_m2_11_20"]
    data = read_json_file(json_file[0])
    import_data_to_postgresql(data, table_name[0],database_config=database_config)
    import_data_to_postgresql(json_file[1], table_name[0],database_config=database_config)


def extract_signal_db(table_name,number_persons, database_config):
    query = """SELECT radar_sample FROM PUBLIC.{} WHERE number_persons = '{}' LIMIT 1;""".format(table_name, number_persons)
    headers, data = read_table_postgresql(table_name=table_name,database_config= database_config, limit = 1, query = query)
    radar_sample = np.array(data)
    radar_sample_resh = np.reshape(radar_sample, (200,1280))
    radar_sample_resh_ = radar_sample_resh - np.mean(radar_sample_resh)
    return headers, radar_sample_resh_