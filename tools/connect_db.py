import os
import psycopg2
import json

def read_json_data(file_path):
    """Returns the content of a JSON file"""
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def convert_to_json(data):
    """Convert nested array to JSON format"""
    return json.dumps(data)  

def fetch_data_db(query, db_url = os.environ["DATABASE_URL"]):
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            data = cur.execute(query).fetch_all()
            conn.commit()
    except psycopg2.Error as e:
        print(f"Error: {e}")
    finally:
        conn.close()
    return data

def execute_query_cockroach(query, db_url = os.environ["DATABASE_URL"]):
    """Executes a query in Cockroach lab database"""
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            conn.commit()
    except psycopg2.Error as e:
        print(f"Error: {e}")
    finally:
        conn.close()

def insert_data_in_db_table(data, table, db_url = os.environ["DATABASE_URL"]):
    """Inserts data from a JSON file into a table in Cockroach database"""
    conn = psycopg2.connect(db_url)
    try:
        query_table = f"""CREATE TABLE IF NOT EXISTS {table} (
            id SERIAL PRIMARY KEY,
            context VARCHAR,
            number_persons INTEGER,
            file_content JSONB,
            shape INTEGER[],
            "type" VARCHAR
            )"""
        with conn.cursor() as cur:
            cur.execute(query_table)
            for record in data:
                query = f"""INSERT INTO {table} 
                        (context, number_persons, file_content, shape, type)
                        VALUES (%s, %s, %s, %s, %s)"""
                cur.execute(query, (record['context'], record['number_persons'],
                                    convert_to_json(record['file_content']), record['shape'], record['type']))
            conn.commit()
    except psycopg2.Error as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    data_json_file = './people_0-10_walking_5m_area.json'
    # data = read_json_data(data_json_file)
    # insert_data_in_db_table(data, "people_0_10_walking_5m_area")
    query = "SHOW DATABASES;"
    data = fetch_data_db(query)
    print(data)

