import argparse
from utils import *
import numpy as np
import json
from easydict import EasyDict as edict
from train_transe_model import extract_embeddings 
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
        # self.embeds = load_embed(path, set_name)
        self.embeds = extract_embeddings(config)
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
            print(f'RELATION : {relation.ljust(16)} | ENTITY : {entity}')
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
        similarity, idx_user = self.cos_sim(
            e_new=user_pref_embed, e_candidates=self.embeds['user'], top_k=top_k)
        print('Cosine Similarity', similarity)
        print('Highest idx Cand User:', idx_user)
        # distance, idx_user = self.euc_dist(
        #     e_new=user_pref_embed, e_candidates=self.embeds['user'], top_k=top_k)
        # print("Euclidean distance:", distance)
        # print('Highest Cand User:', idx_user)
        cand_user_emb = self.embeds['user'][idx_user]
        return idx_user, cand_user_emb
                
    def trim(
        self, 
        dir_path,
        path_file,
        train_labels,
        test_labels,
        kg_config,
        set_name="test",
        ):
        #taking pkl top-k remove idx
        results = pickle.load(open(path_file, "rb"))
        #modifued evaluate_paths form test_agent
        # predict_paths()
        ##### trim here
        pass
       
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
    set_name="train"
     
    init_emb = InitalUserEmbedding(
        path=path, 
        set_name=set_name,
        config=config
    )
    
    #### Given user preferences from conversation    
    user_id = 3246 
    target_item = 2838
    user_acc_feature = [2089]
    user_rej_feature = [308, 1789]
    user_rej_items = [11776, 11777, 7682, 9730, 11269, 2565, 7687, 1019, 2570, 4106, 1548, 7692, 9229, 12, 8721, 3090, 9749, 7739, 10778, 11803, 1052, 6686, 2079, 10272, 10271, 9248, 6176, 1062, 8234, 9774, 9264, 11829, 3129, 10809, 4155, 6203, 8253, 6206, 1594, 4160, 575, 4674, 5698, 10818, 6211, 7238, 10823, 8263, 9289, 10315, 11852, 10828, 6734, 8782, 7244, 7256, 5729, 1633, 9827, 4711, 1639, 7274, 5228, 10349, 10348, 1136, 3700, 8314, 9340, 10367, 7808, 6784, 4223, 10880, 8325, 10894, 10899, 7828, 2195, 10902, 8343, 7320, 9368, 11420, 3740, 668, 4767, 9888, 161, 2722, 3235, 5278, 670, 10918, 6139, 8872, 10401, 1704, 11435, 4779, 10412, 3751, 4784, 6835, 3254, 4279, 7353, 8890, 1210, 10940, 1724, 10431, 1217, 193, 6852, 3785, 11465, 8396, 724, 8916, 725, 5849, 7386, 3290, 1244, 2267, 4833, 8930, 2275, 10469, 1776, 7409, 2290, 5878, 247, 4856, 249, 12022, 8443, 10498, 4354, 7940, 3844, 3333, 6408, 9481, 8973, 9490, 8467, 11539, 9493, 10515, 9723, 3354, 12059, 10017, 7974, 12070, 2856, 7469, 303, 3887, 12081, 10546, 4916, 10550, 10551, 9018, 5946, 316, 2362, 5442, 5445, 5447, 1865, 3406, 846, 10576, 11089, 8021, 2205, 344, 4954, 10588, 7517, 4445, 10077, 865, 3939, 8040, 1387, 3950, 9584, 9072, 10610, 8051, 11124, 1394, 6523, 9087, 7554, 11655, 6026, 1419, 4490, 6541, 2958, 4494, 3472, 5514, 1938, 5002, 7583, 2975, 10145, 6051, 5028, 7589, 8615, 3498, 7084, 6061, 1453, 1455, 5553, 7603, 5043, 6069, 438, 5045, 4533, 1978, 11194, 2493, 3518, 4543, 11712, 5057, 11202, 7109, 8647, 5281, 5067, 8143, 463, 5585, 10194, 3539, 2004, 7637, 470, 4567, 7121, 473, 2010, 11737, 5596, 10712, 6620, 991, 10721, 9187, 3044, 4076, 2541, 10733, 10220, 2544, 9716, 10229, 10740, 503, 4091]
    #### Calculate initial user preference based on features
    user_pref_emb = init_emb.translation(
        user_acc_feature=user_acc_feature, 
        user_rej_feature=user_rej_feature, 
        user_rej_items=user_rej_items
    )
    
    print('User Id (X) :', user_id)
    print('Target Item (Y) :', target_item)
    
    user_emb = init_emb.embeds['user'][user_id]
    #### Calculate who is the closest user
    idx_cand_user, cand_user_emb = init_emb.distance(user_pref_emb, top_k=10)
    
    ### Similarity comparing between X and X'
    diff = np.dot(user_pref_emb, user_emb) / (np.linalg.norm(user_pref_emb) * np.linalg.norm(user_emb))
    print('Similarity', diff)
    ### Distance comparing between X and X'
    # print(user_pref_emb.shape)
    # print(user_emb.shape)
    dist = np.linalg.norm(user_pref_emb -  user_emb) 
    print('Distance', dist)
    
    ##Take cand_user_emb to Path Reasoning
    
    ##Take trim Path Reasoning
        