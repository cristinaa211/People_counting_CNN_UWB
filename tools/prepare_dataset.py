from postgresql_operations import read_table_postgresql, import_processed_data_to_postgresql
from process_data import process_data_pipeline, apply_pca


if __name__ == "__main__":
    database_config = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'UWB_Radar_Samples',
    'user': 'cristina',
    'password': ''
    }
    tables_scenarios ={
            "scenario_1": ["people_walking_5m_area", "people walking in 5m area"],
            "scenario_2" :["people_standing_queue_0_15", "people standing in a queue"],
            "scenario_3":  ["density_3_m2_11_20", "people walking in a room with 3 persons per m2"],
            "scenario_4": ["density_4_m2_11_20" , "people walking in a room with 4 persons per m2"]
            }
    columns = {'columns' : 'label, detailed_label, radar_sample'}
    dataset = []
    for scenario in tables_scenarios.keys():
        headers, data_table = read_table_postgresql(columns=columns, table_name=tables_scenarios[scenario][0],
                                           database_config=database_config)
        for data in data_table:
            processed_signal = process_data_pipeline(data[2], fs = 39*1e9)
            pca_data = apply_pca(processed_signal, n_components=50)
            dataset.append([data[0], data[1], pca_data.tolist()])
        import_processed_data_to_postgresql(dataset, 'processed_data', database_config)

