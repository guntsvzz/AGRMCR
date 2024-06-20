import json
import sys
from tqdm.auto import tqdm

def main(input_file, output_file):
    # Read the text file and create a dictionary
    cid_dict = {}
    with open(input_file, 'r') as f:
        for line in f:
            cid, category = line.strip().split('\t')
            cid_dict[int(cid)] = category

    # Save the dictionary to a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(cid_dict, json_file, indent=4)

    print(f"JSON data saved to {output_file}")

def map_categories_to_indices(c2t_file, cid3_dict_file, output_file):
    # Load Office_Products_c2t.json
    with open(c2t_file, 'r') as f:
        c2t_data = json.load(f)
    
    # Load Office_Products_cid3_dict.json
    with open(cid3_dict_file, 'r') as f:
        cid3_dict = json.load(f)
    
    # Create a mapping dictionary from numeric keys to values
    cid3_mapping = {value: int(key) for key, value in cid3_dict.items()}
    # print(cid3_mapping)
    # Function to update the values in c2t_data using the numeric keys
    def update_values(mapping, data):
        updated_data = {}
        for key, values in tqdm(data.items()):
            updated_values = [mapping.get(v, v) for v in values]
            updated_data[key] = updated_values
        return updated_data

    # Update c2t_data using value_to_key_mapping
    updated_c2t_data = update_values(cid3_mapping, c2t_data)
    
    # Write the updated data back to Office_Products_c2t.json
    with open(output_file, 'w') as f:
        json.dump(updated_c2t_data, f, indent=4)
        
    print(f"JSON data saved to {output_file}")

# def create_index_mapping(c2t_index_file, c2t_file, output_file):
#     # Load the updated feature2id.json
#     with open(feature2id_path, 'r') as f:
#         feature2id_dict = json.load(f)
        
#     # Load the updated c2t_index.json
#     with open(c2t_index_file, 'r') as f:
#         c2t_index_data = json.load(f)
    
#     # Load the c2t.json
#     with open(c2t_file, 'r') as f:
#         c2t_data = json.load(f)
        
#     swapped_dict = {value: key for key, value in c2t_data.items()}
    
#     # Create a mapping from index to category
#     index_mapping = {}
#     for category, indices in c2t_index_data.items():
#         for index in indices:
#             index_mapping[index] = int(swapped_dict[category])

#     # Save the index mapping to a JSON file
#     with open(output_file, 'w') as json_file:
#         json.dump(index_mapping, json_file, indent=4)

#     print(f"JSON data saved to {output_file}")
    
if __name__ == '__main__':
    data_name = sys.argv[1]

    # Define file paths
    cid2 = "./tmp/{}_cid2_dict.txt".format(data_name)
    cid3 = "./tmp/{}_cid3_dict.txt".format(data_name)
    save_path_cid2 = "./tmp/{}_cid2_dict.json".format(data_name)
    save_path_cid3 = "./tmp/{}_cid3_dict.json".format(data_name)

    # Convert the files
    main(cid2, save_path_cid2)
    main(cid3, save_path_cid3)
    
    c2t_json = "./tmp/{}_c2t.json".format(data_name)
    c2t_json_index = "./tmp/{}_c2t_index.json".format(data_name)
    map_categories_to_indices(c2t_json, save_path_cid3, c2t_json_index)

    # # Create the index mapping
    # index_mapping_output = f"./tmp/small_to_large_{data_name}.json"
    # feature2id_path = f"./tmp/feature2id_{data_name}.json" 
    # create_index_mapping(c2t_json_index, save_path_cid2, index_mapping_output)