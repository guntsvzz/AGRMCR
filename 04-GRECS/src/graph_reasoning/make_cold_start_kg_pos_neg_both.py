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
    def __init__(self, set_name, config, preference):
        self.embeds  = load_embed(config.processed_data_dir, set_name)
        self.dataset = load_dataset(config.processed_data_dir, set_name)
        self.kg      = load_kg(config.processed_data_dir, set_name)
        self.offset  = len(self.kg.G['brand'])
        self.user_preferences = UserPreferences()
        self.preference=preference
        
    ##Positive Preference
    def top_k_argmax(self, A, B, k):
        # Normalize A
        A_norm = A / np.linalg.norm(A)
        # Normalize B
        B_norm = B / np.linalg.norm(B, axis=1)[:, np.newaxis]
        # Compute cosine similarity
        similarity = np.dot(B_norm, A_norm)
        # Get top-k argmax indices
        top_k_max_indices = np.argpartition(similarity, -k)[-k:]
        top_k_max_indices = top_k_max_indices[np.argsort(similarity[top_k_max_indices])[::-1]]
        # Get top-k max similarities
        top_k_max_values = similarity[top_k_max_indices]
        return top_k_max_indices, top_k_max_values
    
    ##Negative Preference
    def top_k_argmin(self, A, B, k):
        # Normalize A
        A_norm = A / np.linalg.norm(A)
        # Normalize B
        B_norm = B / np.linalg.norm(B, axis=1)[:, np.newaxis]
        # Compute cosine similarity
        similarity = np.dot(B_norm, A_norm)
        # Get top-k argmin indices
        top_k_min_indices = np.argpartition(similarity, k)[:k]
        top_k_min_indices = top_k_min_indices[np.argsort(similarity[top_k_min_indices])]
        
        # Get top-k min similarities
        top_k_min_values = similarity[top_k_min_indices]
        return top_k_min_indices, top_k_min_values

    def overlap(self, top_k_max_indices, top_k_min_indices, top_k=5):
        overlap_user = np.intersect1d(top_k_max_indices, top_k_min_indices) #[:top_k]
        # Randomly select 10 indices
        # return np.random.choice(overlap_user, size=top_k, replace=False)
        # Find intersection of A and B
        intersection, indices_A, indices_B = np.intersect1d(
            top_k_max_indices, top_k_min_indices, return_indices=True)
        # Sort intersection based on indices in A to maintain A's order
        sorted_intersection = intersection[np.argsort(indices_A)]
        # Get the top-k elements from sorted_intersection
        return sorted_intersection[:top_k]
    
    def cos_sim(self, e_new, e_candidates, top_k=10):
        # Compute norms
        norm_e_new = np.linalg.norm(e_new)
        norm_e_candidates = np.linalg.norm(e_candidates, axis=1)
        # Handle zero norms by setting them to a small positive value
        norm_e_new = norm_e_new if norm_e_new != 0 else 1e-9
        norm_e_candidates[norm_e_candidates == 0] = 1e-9
        # Compute cosine similarity
        cosine = np.dot(e_candidates, e_new) / (norm_e_candidates * norm_e_new)
        # Find the indices of the top k maximum cosine similarities
        top_k_indices = np.argpartition(cosine, -top_k)[-top_k:]
        # Get the top k cosine similarities
        top_k_similarities = cosine[top_k_indices]
        # Sort the top k similarities and their indices
        sorted_indices = top_k_indices[np.argsort(-top_k_similarities)]
        sorted_similarities = top_k_similarities[np.argsort(-top_k_similarities)]
        return sorted_similarities, sorted_indices
    
    # def distance(self, user_pref_embed, top_k=10, preference='positive'):
    #     similarity, idx_user = self.cos_sim(
    #         e_new=user_pref_embed, e_candidates=self.embeds['user'], top_k=top_k)
    #     # print('Cosine Similarity', similarity)
    #     # print('Highest idx Cand User:', idx_user)
    #     # distance, idx_user = self.euc_dist(
    #     #     e_new=user_pref_embed, e_candidates=self.embeds['user'], top_k=top_k)
    #     # print("Euclidean distance:", distance)
    #     # print('Highest Cand User:', idx_user)
    #     cand_user_emb = self.embeds['user'][idx_user]
    #     return idx_user, cand_user_emb
        
    def distance(self, pos_pref_embed=None, neg_pref_embed=None, top_k=5):
        if self.preference == 'positive' and (pos_pref_embed is not None):
            top_k_max_indices, top_k_max_values = self.top_k_argmax(
                pos_pref_embed, self.embeds['user'], top_k)
            idx_user = top_k_max_indices
        elif self.preference == 'negative' and (neg_pref_embed is not None):
            top_k_min_indices, top_k_min_values = self.top_k_argmin(
                neg_pref_embed, self.embeds['user'], top_k)
            idx_user = top_k_min_indices
        elif (self.preference == 'both') and (pos_pref_embed is not None) and (neg_pref_embed is not None):
            top_k_max_indices, top_k_max_values = self.top_k_argmax(
                pos_pref_embed, self.embeds['user'], int(0.6*len(self.embeds['user'])))
            top_k_min_indices, top_k_min_values = self.top_k_argmin(
                neg_pref_embed, self.embeds['user'], int(0.6*len(self.embeds['user'])))
            idx_user = self.overlap(top_k_max_indices, top_k_min_indices, top_k)
            
        cand_user_emb = self.embeds['user'][idx_user]
        return idx_user, cand_user_emb
     
    # def translation(self, user_acc_feature=None, user_rej_feature=None, user_rej_items=None):
    #     # Construction from user's perference
    #     self.user_preferred = self.user_pref(user_acc_feature, user_rej_feature, user_rej_items)
    #     # Intialize zero user embedding
    #     zero_embeds = {'user': np.zeros(100,)} # zero_embeds['user']
    #     nb_relations = 0
    #     # Accessing items in the dictionary:
    #     for relation, entity in self.user_preferences.items():
    #         # print(f'RELATION : {relation.ljust(16)} | ENTITY : {entity}')
    #         if relation == 'disinterested_in':
    #             relation = 'interested_in'
    #             continue
    #         entities = self.user_preferred[relation]
    #         all_related_emb = (
    #             self.embeds[entity[1]][entities] - self.embeds[relation][0]
    #         )
    #         nb_relations += all_related_emb.shape[0]
    #         # sum all related entities embeddings
    #         if relation in ['interested_in', 'like', 'dislike']:
    #             zero_embeds["user"] += all_related_emb.sum(axis=0)
    #         # elif relation in ['disinterested_in']:
    #         #     zero_embeds["user"] -= all_related_emb.sum(axis=0)
    #     # divide by the number of relations to get the average
    #     if nb_relations > 0:
    #         zero_embeds["user"] /= nb_relations
            
    #     return zero_embeds["user"]
    
    def translation(self, user_acc_feature=None, user_rej_feature=None, user_rej_items=None):
        # Construction from user's perference
        self.user_preferred = self.user_preference_config(user_acc_feature, user_rej_feature, user_rej_items)
        
        # Intialize zero user embedding
        pos_zero_embeds = {'user': np.zeros(100,)} # zero_embeds['user']
        neg_zero_embeds = {'user': np.zeros(100,)} # zero_embeds['user']
        nb_relations = 0

        if (self.preference in ['positive', 'both']) and (user_acc_feature is not None):
            # Accessing items in the dictionary:
            for relation, entity in self.user_preferences.items():
                # print(f'RELATION : {relation.ljust(16)} | ENTITY : {entity}')
                if relation == 'disinterested_in':
                    relation = 'interested_in'
                    continue
                entities = self.user_preferred[relation]
                all_related_emb = (
                    self.embeds[entity[1]][entities] - self.embeds[relation][0]
                )
                nb_relations += all_related_emb.shape[0]
                # sum all related entities embeddings
                if relation in ['interested_in', 'like']:
                    pos_zero_embeds["user"] += all_related_emb.sum(axis=0)
  
            # divide by the number of relations to get the average
            if nb_relations > 0:
                pos_zero_embeds["user"] /= nb_relations
        else:
            pos_zero_embeds["user"] = None
        
        if (self.preference in ['negative', 'both']) and (user_rej_feature is not None):
            # Accessing items in the dictionary:
            for relation, entity in self.user_preferences.items():
                # print(f'RELATION : {relation.ljust(16)} | ENTITY : {entity}')
                if relation == 'disinterested_in':
                    relation = 'interested_in'
                    # continue
                entities = self.user_preferred[relation]
                all_related_emb = (
                    self.embeds[entity[1]][entities] - self.embeds[relation][0]
                )
                nb_relations += all_related_emb.shape[0]
                # sum all related entities embeddings
                if relation in ['interested_in', 'dislike']:
                    neg_zero_embeds["user"] += all_related_emb.sum(axis=0)
            
            # divide by the number of relations to get the average
            if nb_relations > 0:
                neg_zero_embeds["user"] /= nb_relations
        else:
            neg_zero_embeds["user"] = None
            
        return pos_zero_embeds["user"], neg_zero_embeds["user"]
    
    
    def get_feature(self, idx):
        # Replace with your actual logic to determine if idx is a brand or category
        if idx >= self.offset:
            return 'category'
        else:
            return 'brand'
    
    def user_preference_config(self, user_acc_feature=None, user_rej_feature=None, user_rej_items=None):
        if user_acc_feature is None:
            user_acc_feature = list()
        if user_rej_feature is None:
            user_rej_feature = list()
        if user_rej_items is None:
            user_rej_items = list()
            
        # Create an empty user dictionary with the same keys and empty lists as values
        user_preferred = {key: [] for key in self.dataset.data_args.kg_relation.user.keys()}
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

    cold_start_uids = {}
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
        )
        
        user_key = user_pref[str(idx)]['idx_user']
        if user_key in cold_start_uids:
            for key, value in user_preferred.items():
                if isinstance(value, list):
                    cold_start_uids[user_key][key].extend(value)
                    # Remove redundant values
                    cold_start_uids[user_key][key] = list(set(cold_start_uids[user_key][key]))
        else:
            cold_start_uids[user_key] = user_preferred
    
    print('cold_start_uids', len(cold_start_uids))
    
    # Accessing items in the dictionary:
    for idx, user in enumerate(cold_start_uids):
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
    parser.add_argument(
        '--preference', 
        type=str, 
        default='positive', 
        choices=['positive','negative', 'both'],
        help='One of {positive,negative,both}'
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = edict(json.load(f))

    config.seed = args.seed
    # config.TRAIN_EMBEDS.epochs = args.epochs
    # config.TRAIN_EMBEDS.min_epochs = args.min_epochs
    
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