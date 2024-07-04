from __future__ import absolute_import, division, print_function

import os
import json
import argparse
import torch
import torch.optim as optim
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm
from utils import *
from easydict import EasyDict as edict

class UserPreferences(dict):
    def __init__(self):
        super().__init__()
        self.update({
            'interested_in': ['interested_in_u_c.txt', 'category'],
            'like': ['like_u_b_rate.txt', 'brand'],
            'disinterested_in': ['interested_in_u_c.txt', 'category'],
            'dislike': ['dislike_u_b_rate.txt', 'brand']
        })

    def items(self):
        return super().items()
    
class InitalUserEmbedding:
    def __init__(self, set_name, config):
        self.embeds  = load_embed(config.processed_data_dir, set_name)
        self.dataset = load_dataset(config.processed_data_dir, set_name)
        self.kg      = load_kg(config.processed_data_dir, set_name)
        self.offset  = len(self.kg.G['brand'])
        self.user_preferences = UserPreferences()
        
    def get_feature(self, idx):
        # Replace with your actual logic to determine if idx is a brand or category
        if idx >= self.offset:
            return 'category'
        else:
            return 'brand'
    
    def user_preference_config(
        self, user_acc_feature=None, user_rej_feature=None, user_rej_items=None, dataset=None):
        if user_acc_feature is None:
            user_acc_feature = list()
        if user_rej_feature is None:
            user_rej_feature = list()
        if user_rej_items is None:
            user_rej_items = list()
            
        # Create an empty user dictionary with the same keys and empty lists as values
        user_preferred = {key: [] for key in dataset.data_args.kg_relation.user.keys()}
        user_preferred['disinterested_in'] = []
        user_preferred['dislike'] = []
        user_preferred['non-purchase'] = []

        # Process user_acc_feature
        for idx in user_acc_feature:
            preference = self.get_feature(idx)
            if preference == 'brand':
                user_preferred['like'].append(idx)
            elif preference == 'category':
                user_preferred['interested_in'].append(idx- self.offset)

        # Process user_rej_feature
        for idx in user_rej_feature:
            preference = self.get_feature(idx)
            if preference == 'brand':
                user_preferred['dislike'].append(idx)
            elif preference == 'category':
                user_preferred['disinterested_in'].append(idx - self.offset)

        # Assign user_rej_items to non-purchase
        user_preferred['non-purchase'] = user_rej_items
        
        return user_preferred

def load_user_pref(path, domain):
    user_pref_path = os.path.join(path)
    # Load JSON data from a file
    user_pref = json.load(open(f'{user_pref_path}/user_preference_{domain}.json', 'r'))
    return user_pref

def make_cold_embeds(config, set_name, domain):
    # embeds  = load_embed(config.processed_data_dir, set_name)
    # dataset = load_dataset(config.processed_data_dir, set_name)
    # kg      = load_kg(config.processed_data_dir, set_name)
    
    init_embed = InitalUserEmbedding(
        set_name=set_name,
        config=config
    )
    embeds = init_embed.embeds
    
    transe_config = config.TRAIN_EMBEDS

    # load cold start users
    cold_users_path = os.path.join(config.processed_data_dir, "cold_start_users.json")
    cold_users = json.load(open(cold_users_path, "r"))

    # load cold start items
    cold_items_path = os.path.join(config.processed_data_dir, "cold_start_items.json")
    cold_items = json.load(open(cold_items_path, "r"))
    
    # set all cold start users embeddings to 0
    tmp_cold_users = cold_users["test"] + cold_users["validation"]
    embeds["user"][tmp_cold_users] = 0

    # # set all cold start items embeddings to 0
    # tmp_cold_items = cold_items["test"] + cold_items["validation"]
    # embeds["item"][tmp_cold_items] = 0
    
    tmp_cold_users = cold_users[set_name]
    tmp_cold_items = cold_items[set_name]
    # making a copy of the embeddings to avoid using the modified cold start embeddings in the next iteration
    tmp_embeds = deepcopy(embeds)
    
    nb_relations = 0
    user_preferences = UserPreferences()
    
    user_pref = load_user_pref(config.processed_data_dir, domain)
    print('user_pref', len(user_pref))

    all_user_pref = {}
    dataset = load_dataset(config.processed_data_dir, set_name)
    for idx in tqdm(range(len(user_pref))):
        user_id = user_pref[str(idx)]['idx_user']
        target_item = user_pref[str(idx)]['idx_item']
        user_acc_feature = user_pref[str(idx)]['user_acc_feature']
        user_rej_feature = user_pref[str(idx)]['user_rej_feature']
        user_rej_items = user_pref[str(idx)]['user_rej_items']
        
        user_preferred = init_embed.user_preference_config(
            user_acc_feature = user_acc_feature, 
            user_rej_feature = user_rej_feature, 
            user_rej_items = user_rej_items, 
            dataset= dataset)
        
        all_user_pref[user_pref[str(idx)]['idx_user']] = user_preferred
    
    print('all_user_pref', len(all_user_pref))
    
    # Accessing items in the dictionary:
    for idx, user in enumerate(all_user_pref):
        # for relation, entity in dataset.data_args.item_relation.items():
        for relation, entity in user_preferences.items():
            # print(f'RELATION : {relation.ljust(16)} | ENTITY : {entity}')
            if relation == 'disinterested_in':
                relation = 'interested_in'
                continue
            entities = user_preferred[relation]
            all_related_emb = (
                embeds[entity[1]][entities] - embeds[relation][0]
            )
            nb_relations += all_related_emb.shape[0]
            # sum all related entities embeddings
            if relation in ['interested_in', 'like', 'dislike']:
                tmp_embeds["user"][user] += all_related_emb.sum(axis=0)
            # elif relation in ['disinterested_in']:
            #     zero_embeds["user"] -= all_related_emb.sum(axis=0)
        # divide by the number of relations to get the average
        if nb_relations > 0:
            tmp_embeds["user"][user] /= nb_relations 
        
    # save the embeddings
    save_embed(
        config.processed_data_dir, f"{set_name}_cold_start_transe_embed.pkl", tmp_embeds
    )
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Config file.",
        default="config/coco_01_01/UPGPR_10.json",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        help="Random seed.", 
        default=0
    )
    parser.add_argument(
        '--domain', 
        type=str, 
        default='Beauty', 
        choices=['Beauty','Cellphones', 'Clothing', 'CDs'],
        help='One of {Beauty, Clothing, Cellphones, CDs}.'
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = edict(json.load(f))

    config.seed = args.seed
    config.TRAIN_EMBEDS.epochs = args.epochs
    config.TRAIN_EMBEDS.min_epochs = args.min_epochs
    
    transe_config = config.TRAIN_EMBEDS
    transe_config.use_user_relations = config.use_user_relations
    transe_config.use_entity_relations = config.use_entity_relations

    assert (
        transe_config.min_epochs <= transe_config.epochs
    ), "Minimum number of epochs should be lower than total number of epochs."

    if config.use_wandb:
        wandb.init(
            project=config.wandb_project_name, name=config.wandb_run_name, config=config
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = transe_config.gpu

    transe_config.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    
    set_name = "test"
    make_cold_embeds(config, set_name, args.domain)

if __name__ == "__main__":
    main()