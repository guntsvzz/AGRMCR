import argparse
from utils import *
import numpy as np
import json
import os
from functools import reduce
from easydict import EasyDict as edict
from train_transe_model import extract_embeddings 
from path_utils import predict_paths, evaluate_paths

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
    def __init__(self, path, set_name, config):
        self.embeds = load_embed(path, set_name)
        # self.embeds = extract_embeddings(config)
        self.dataset = load_dataset(path, set_name)
        self.kg = load_kg(path, set_name)
        self.offset = len(self.kg.G['brand'])
        self.user_preferences = UserPreferences()
          
    def get_feature(self, idx):
        # Replace with your actual logic to determine if idx is a brand or category
        if idx >= self.offset:
            return 'category'
        else:
            return 'brand'
    
    def user_pref(self, user_acc_feature=None, user_rej_feature=None, user_rej_items=None):
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
    
    def translation(self, user_acc_feature=None, user_rej_feature=None, user_rej_items=None):
        # Construction from user's perference
        self.user_preferred = self.user_pref(user_acc_feature, user_rej_feature, user_rej_items)
        # Intialize zero user embedding
        zero_embeds = {'user': np.zeros(100,)} # zero_embeds['user']
        nb_relations = 0
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
            if relation in ['interested_in', 'like', 'dislike']:
                zero_embeds["user"] += all_related_emb.sum(axis=0)
            # elif relation in ['disinterested_in']:
            #     zero_embeds["user"] -= all_related_emb.sum(axis=0)
        # divide by the number of relations to get the average
        if nb_relations > 0:
            zero_embeds["user"] /= nb_relations
            
        return zero_embeds["user"]
    
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
    
        # Find the index of the maximum cosine similarity
        # idx_cand_user = np.argmax(cosine)
        # return cosine, idx_cand_user

    def euc_dist(self, e_new, e_candidates, top_k=10):
        # Compute Euclidean distance
        distances = np.linalg.norm(e_candidates - e_new, axis=1)    
        # Find the indices of the top k minimum distances
        top_k_indices = np.argpartition(distances, top_k)[:top_k]
        # Get the top k distances
        top_k_distances = distances[top_k_indices]
        # Sort the top k distances and their indices
        sorted_indices = top_k_indices[np.argsort(top_k_distances)]
        sorted_distances = top_k_distances[np.argsort(top_k_distances)]
        return sorted_distances, sorted_indices
    
        # Find the index of the minimum distance
        # idx_cand_user = np.argmin(distances)
        # return distances, idx_cand_user

    def distance(self, user_pref_embed, top_k=10):
        # similarity, idx_user = self.cos_sim(
        #     e_new=user_pref_embed, e_candidates=self.embeds['user'], top_k=top_k)
        # print('Cosine Similarity', similarity)
        # print('Highest idx Cand User:', idx_user)
        distance, idx_user = self.euc_dist(
            e_new=user_pref_embed, e_candidates=self.embeds['user'], top_k=top_k)
        # print("Euclidean distance:", distance)
        # print('Highest Cand User:', idx_user)
        cand_user_emb = self.embeds['user'][idx_user]
        return idx_user, cand_user_emb
     
def load_user_pref(path, domain):
    user_pref_path = os.path.join(path)
    user_pref = json.load(open(f'{user_pref_path}/user_preference_{domain}.json', 'r'))
    return user_pref
  
def test_cand(config, idx_cand_user, trim=True):
    config_agent = config.AGENT
    kg_config = config.KG_ARGS

    policy_file = config_agent.log_dir + "/tmp_policy_model_epoch_{}.ckpt".format(
        config_agent.epochs
    )
    path_file = config_agent.log_dir + "/policy_paths_epoch_{}.pkl".format(
        config_agent.epochs
    )

    train_labels = load_labels(config.processed_data_dir, "train")
    test_labels = load_labels(config.processed_data_dir, "test")

    dataset_name = config.processed_data_dir.split("/")[-1]

    model_name = (
        "UPGPR_len_"
        + str(config_agent.max_path_len)
        + "_"
        + config.AGENT.reward
        + "_"
        + config.TRAIN_EMBEDS.cold_start_embeddings
        + "_mask_"
        + str(config.AGENT.mask_first_interaction)
        + "_max_cold_concept_"
        + str(kg_config.max_nb_cold_entities)
        + "_topk_"
        + "_".join(map(str, config_agent.topk))
        + "_draft_"
    )

    config_agent.result_file_dir = os.path.join(
        config_agent.result_file_dir, dataset_name, model_name, str(config.seed)
    )

    os.makedirs(
        config_agent.result_file_dir,
        exist_ok=True,
    )
    
    if config_agent.run_path:
        predict_paths(
            policy_file, 
            path_file, config, 
            config_agent, 
            kg_config
        )
    if config_agent.run_eval:
        evaluate_paths(
            config.processed_data_dir,
            path_file,
            train_labels,
            test_labels,
            kg_config,
            config.use_wandb,
            config_agent.result_file_dir,
            validation=False,
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Config file.",
        default="config/beauty/graph_reasoning/UPGPR.json",
    )
    parser.add_argument("--seed", type=int, help="Random seed.", default=0)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = edict(json.load(f))
    
    config.seed = args.seed
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

    transe_config.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    config.log_dir = "{}/{}".format(config.processed_data_dir, transe_config.name)
    if not os.path.isdir(config.log_dir):
        os.makedirs(config.log_dir)
    
    # path="./data/beauty/Amazon_Beauty_01_01"
    path = config.processed_data_dir
    config.seed = args.seed
    config_agent = config.AGENT
    
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project_name,
            name=config.wandb_run_name,
            config=config,
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = config_agent.gpu
    config_agent.device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

    if config_agent.early_stopping == True:
        with open("early_stopping.txt", "r") as f:
            config_agent.epochs = int(f.read())

    config_agent.log_dir = config.processed_data_dir + "/" + config_agent.name
    
    #### initialize new user preference
    init_emb = InitalUserEmbedding(
        path=path, 
        set_name="test",
        config=config
    )
    
    #### Given user preferences from conversation    
    user_id = 5164 
    target_item = 2085
    user_acc_feature = [2078]
    user_rej_feature = [2193, 2217, 2258, 2279]
    user_rej_items = [4096, 9728, 11779, 7687, 7692, 8725, 9751, 7709, 2591, 8740, 2602, 3132, 11330, 4683, 2129, 3666, 4199, 9320, 617, 6763, 7280, 11381, 10357, 1146, 5244, 6273, 2194, 7322, 10396, 5794, 9891, 7846, 6310, 3240, 5801, 2731, 1709, 5806, 7343, 6836, 3764, 9405, 5825, 1231, 9424, 1751, 11480, 218, 1757, 10986, 7403, 4341, 6911, 4864, 4354, 6403, 7940, 281, 10017, 1315, 2856, 10539, 812, 11565, 6958, 1332, 4924, 5954, 10056, 9546, 9548, 1358, 1360, 8035, 1380, 5991, 3435, 5487, 6001, 369, 898, 1931, 10123, 5527, 9113, 7587, 6565, 11692, 944, 6073, 3525, 10183, 10189, 4563, 3544, 473, 9185, 11750, 11243, 2046]

    #### Calculate initial user preference based on features
    user_pref_emb = init_emb.translation(
        user_acc_feature=user_acc_feature, 
        user_rej_feature=user_rej_feature, 
        # user_rej_items=user_rej_items
    )
    
    print('User Id (X) :', user_id)
    print('Target Item (Y) :', target_item)
    print('user_acc_feature =', user_acc_feature)
    print('user_rej_feature =', user_rej_feature)
    print('user_rej_items =', user_rej_items)
    
    user_emb = init_emb.embeds['user'][user_id]
    #### Calculate who is closer to new user with top-K
    idx_cand_user, cand_user_emb = init_emb.distance(user_pref_emb, top_k=10)
    
    ### Similarity comparing between X and X'
    # diff = np.dot(user_pref_emb, user_emb) / (np.linalg.norm(user_pref_emb) * np.linalg.norm(user_emb))
    # print('Similarity', diff)
    ### Distance comparing between X and X'
    # print(user_pref_emb.shape)
    # print(user_emb.shape)
    # dist = np.linalg.norm(user_pref_emb -  user_emb) 
    # print('Distance', dist)
    
    ##Take cand_user_emb to Path Reasoning and trim
    test_cand(config, idx_cand_user, trim = True)
        
    if config.use_wandb:
        wandb.finish()
