import pandas as pd
import json
import sys
from tqdm.auto import tqdm

def main():    
    # Load the Office Products ID dictionary
    item_id_dict = {}
    with open(item_id, 'r') as f:
        for line in f:
            asin, id_val = line.strip().split('\t')
            item_id_dict[asin] = int(id_val)
    
    # # Load all JSON lines into a list of dictionaries
    # with open(meta_file, 'r') as fp:
    #     # Load all JSON lines into a list of dictionaries
    #     data = [json.loads(line.strip()) for line in fp]
        
    # # Create the pandas DataFrame
    # df_meta = pd.DataFrame(data)
    
    parsed_data = []
    with open(meta_file, 'r') as file:
        for line in tqdm(file):
            try:
                # Evaluate the line (convert from str to dict) - considering each line is HTML wrapped in a JSON-like structure
                data = eval(line.strip())  
                # Append the parsed data to the list
                parsed_data.append(data)
            except Exception as e:
                print(f"Error parsing line: {line}, {e}")

    # Create DataFrame from the parsed data
    df_meta = pd.DataFrame(parsed_data)
    print("Number of Item :",df_meta.shape)
    
    # Filter the DataFrame to keep only rows where 'asin' is in the item_id_dict
    df_filtered = df_meta[df_meta['asin'].isin(item_id_dict.keys())]
    # Drop duplicate 'asin' entries, keeping the first occurrence
    df_filtered = df_filtered.drop_duplicates(subset='asin', keep='first')
    print("Number of Filter Item :",df_filtered.shape)
    assert df_filtered.shape[0] == len(item_id_dict), "Filter dataframe does not match the size of Item dict"
    
    ####BRAND####
    unique_brands = df_filtered['brand'].unique()
    # Replace &amp; with & in the brand names
    cleaned_brands = [brand.replace('&amp;', '&') if isinstance(brand, str) else brand for brand in unique_brands]
    # Create a dictionary mapping each brand to a unique ID
    brand_dict = {brand: idx for idx, brand in enumerate(cleaned_brands)}
    print("Number of Brand", len(brand_dict))
    
    # Save the dictionary to a JSON file
    with open(save_path_brand, 'w') as json_file:
        json.dump(brand_dict, json_file, indent=4)
    print(f"JSON data saved to {save_path_brand}")
        
    ####ITEM####
    # Create the desired dictionary
    result_dict = {}
    for index, row in df_filtered.iterrows():
        asin = row['asin']
        
        # Handle '&amp;' in the brand name
        brand = row['brand'].replace('&amp;', '&') if 'brand' in row else None
        cid2, cid3 = row['category'][2], row['category'][3]
        
        if asin in item_id_dict:
            mapped_index = item_id_dict[asin]
            result_dict[mapped_index] = {
                "asin": row['asin'],
                # "categories": row['category'],
                "also_buy": [item_id_dict.get(item, item) for item in row['also_buy']],
                "also_view": [item_id_dict.get(item, item) for item in row['also_view']],
                "feature": row['feature'],
                "price": row['price'],
                "brand": brand_dict[brand],
                "asin": asin,
                "belong_to": cid3,
                "belong_to_large": cid2,
                "interact": None
            }
        else:
            result_dict[asin] = {
                "asin": row['asin'],
                # "categories": row['category'],
                "also_buy": [item_id_dict.get(item, item) for item in row['also_buy']],
                "also_view": [item_id_dict.get(item, item) for item in row['also_view']],
                "feature": row['feature'],
                "price": row['price'],
                "brand": brand_dict[brand],
                "asin": asin,
                "belong_to": cid3,
                "belong_to_large": cid2,
                "interact": None
            }
    
    # Save the dictionary to a JSON file
    with open(save_path, 'w') as json_file:
        json.dump(result_dict, json_file, indent=4)
        
    print(f"JSON data saved to {save_path}")

if __name__ == '__main__':    
    data_name = sys.argv[1]
    
    item_id = f"./tmp/{data_name}_id_dict.txt"
    meta_file = f"./tmp/filtered_meta_{data_name}.json"
    save_path = f"./tmp/item_dict_{data_name}.json"
    save_path_brand = f"./tmp/brand_dict_{data_name}.json"
    
    main()
