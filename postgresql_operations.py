import psycopg2
import json

def connect_postgresql(user, password, host, port, database):
    try:
        conn = psycopg2.connect(database = database, user = user, password = password, 
                                                 host = host, port = port)
        print("Database connected successfully")
    except Exception as e:
        print("Database not connected successfully")
        print(e)
    return conn

def read_table_postgresql(user, password, host, port, database, table_name):
    conn = connect_postgresql(user = user, password = password, host = host, port = port, database=database)
    cursor = conn.cursor()
    query = 'select * from Public."{}"'.format(table_name)
    cursor.execute(query)
    data = cursor.fetchall()
    headers = [i[0] for i in cursor.description]
    print('Data fetched successfully')
    if conn:
        cursor.close()
        conn.close()
        print("PostgreSQL connection is closed")
    return data, headers


def import_json_to_postgres(json_filename,table_name, database_config):
    # Extract database configuration parameters
    host = database_config['host']
    port = database_config['port']
    dbname = database_config['dbname']
    user = database_config['user']
    password = database_config['password']
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password
    )
    cursor = conn.cursor()
    # Read JSON file and insert data into the database
    with open(json_filename, 'r') as json_file:
        data = json.load(json_file)
        for item in data:
            # Assuming your JSON structure is a list of objects
            json_data = json.dumps(item)
            insert_query = f"INSERT INTO {table_name} (json_column) VALUES (%s)"
            cursor.execute(insert_query, (json_data,))
    # Commit changes and close connections
    conn.commit()
    cursor.close()
    conn.close()

