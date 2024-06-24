import pandas as pd
import json
import sys
from tqdm.auto import tqdm

def main():   
    #### Mapping feature
    # Read brand dictionary
    with open(brand_path, 'r') as f:
        brand_dict = json.load(f)
    # Read cid2 dictionary
    with open(cid2_path, 'r') as f:
        cid2_dict = json.load(f)
    # Read cid3 dictionary
    with open(cid3_path, 'r') as f:
        cid3_dict = json.load(f)
    # Create dictionary to store mappings
    feature2id_dict = {}
    current_id = 0
    # Populate dictionary with brand names and continuous IDs
    for brand_name in brand_dict:
        feature2id_dict[f'brand_{brand_name}'] = current_id
        current_id += 1
    # Populate dictionary with cid2 categories and continuous IDs
    for category in cid2_dict.values():
        feature2id_dict[f'category_{category}'] = current_id
        current_id += 1
    # Populate dictionary with cid3 categories and continuous IDs
    for category in cid3_dict.values():
        feature2id_dict[f'type_{category}'] = current_id
        current_id += 1
    
    # Write dictionary to feature2id.json
    with open(feature2id_save_path, 'w') as f_out:
        json.dump(feature2id_dict, f_out, indent=4)
        
    print(f"JSON data saved to {feature2id_save_path}")
    
    #### item_feature
    # Load the Office Products ID dictionary
    with open(item_path, 'r') as file:
        item_data = json.load(file) #dictionary
        
    item_feature = {}
    for key, value in item_data.items():
        item_feature[key] = [value['brand'], feature2id_dict[f"type_{value['belong_to']}"], feature2id_dict[f"category_{value['belong_to_large']}"]] #, value['price']
        # item_feature[key].extend(value['feature'])

    # Save the dictionary to a JSON file
    with open(item_feature_save_path, 'w') as json_file:
        json.dump(item_feature, json_file, indent=4)
        
    print(f"JSON data saved to {item_feature_save_path}")
      
def create_index_mapping():
    # Load the updated c2t.json
    with open(c2t_json, 'r') as f:
        c2t_data = json.load(f)
    # Load the feature2id_path
    with open(feature2id_path, 'r') as f:
        feature2id_data = json.load(f)
        
    # Create a mapping from index to category
    index_mapping = {}
    for category, indices in c2t_data.items():
        for index in indices:
            index_mapping[feature2id_data[f'type_{index}']] = feature2id_data[f'category_{category}']
        # break
        
    # Save the index mapping to a JSON file
    with open(index_mapping_output, 'w') as json_file:
        json.dump(index_mapping, json_file, indent=4)
        
    print(f"JSON data saved to {index_mapping_output}")
          
if __name__ == '__main__':    
    data_name = sys.argv[1]
    
    item_path = f"./tmp/item_dict_{data_name}.json"
    cid2_path = f"./tmp/{data_name}_cid2_dict.json"
    cid3_path = f"./tmp/{data_name}_cid3_dict.json"
    brand_path = f'./tmp/brand_dict_{data_name}.json'
        
    item_feature_save_path  = f"./tmp/item_feature_{data_name}.json"
    feature2id_save_path = f"./tmp/feature2id_{data_name}.json"
    # small_to_large_save_path = f"./tmp/small_to_large_{data_name}.json"
    
    main()

    feature2id_path = f"./tmp/feature2id_{data_name}.json" 
    c2t_json = "./tmp/{}_c2t.json".format(data_name)
    index_mapping_output = f"./tmp/small_to_large_{data_name}.json"
    
    create_index_mapping()