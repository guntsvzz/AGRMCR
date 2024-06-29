from collections import OrderedDict
import os,sys
import gzip
from tqdm.auto import tqdm
import json

set_name = ["train", "test"] #, "validation"]
set_domain = ['Beauty','Cellphones', 'Cloth', 'CDs']

dataset_name= sys.argv[1] #'Beauty'

assert dataset_name in set_domain, f'{dataset_name} is not in {set_domain}'

for each in set_name:
    path_txt = f'data/{dataset_name.lower()}/Amazon_{dataset_name}_01_01/{each}.txt'
    print(path_txt)
    # Read data from the text file
    with open(path_txt, 'r') as file:
        data = file.readlines()

    # Initialize an empty dictionary
    result = {}

    # Process each line of the data
    for line in data:
        key, value = map(int, line.strip().split())
        if key not in result:
            result[key] = []
        result[key].append(value)

    # Sort the dictionary by keys
    review = OrderedDict(sorted(result.items()))
    if each == 'train':
        each = 'valid'
    if each == 'test':
        each = 'test'
    save_path_review = f'data/{dataset_name.lower()}/Amazon_{dataset_name}_01_01/review_dict_{each}_{dataset_name}.json'

    # Convert dictionary to JSON and save to a file
    with open(save_path_review, 'w') as json_file:
        json.dump(review, json_file, indent=4)
         
    print(f'Save path : {save_path_review}')