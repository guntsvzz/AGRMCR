import argparse
import random
import wandb
import random
import json
from tqdm.auto import tqdm
from easydict import EasyDict as edict
import pickle

def load_kg(path, set_name):
    kg_file = path + "/" + set_name + "_kg.pkl"
    print("Load KG:", kg_file)
    kg = pickle.load(open(kg_file, "rb"))
    return kg
    
def make_review_cold_start(config, args):
    # args.domain = 'Beauty'
    # Load JSON data from a file
    file_path = f'{config.processed_data_dir}/review_dict_test_{args.domain}.json'
    data = json.load(open(file_path, 'r'))
    print('original' , len(data))

    set_name="test"
    kg = load_kg(config.processed_data_dir, set_name)
    kg_args = config.KG_ARGS
    
    # only select the users that have interactions
    uids = [
        key
        for key in kg.G["user"]
        if kg.G["user"][key][kg_args.interaction]
    ]

    # Convert uids to strings, as JSON keys are strings
    uids_str = list(map(str, uids))
    # Remove keys from the data
    for uid in uids_str:
        if uid in data:
            del data[uid]        
    print('Remove cold-start user' , len(data))

    # Save the modified data back to the file (optional)
    file_path = f'{config.processed_data_dir}/review_dict_test_cold_start_{args.domain}.json'
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Config file.",
        default="config/beauty/graph_reasoning/UPGPR.json",
    )
    parser.add_argument(
        '--domain', 
        type=str, 
        default='Beauty', 
        choices=['Beauty','Cellphones', 'Clothing', 'CDs'],
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = edict(json.load(f))
        
    make_review_cold_start(config, args)