from utils import *
import numpy as np

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
    def __init__(self, path, set_name):
        self.embeds = load_embed(path, set_name)
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
    
    def cos_sim(e_new, e_candidates):
        # Compute norms
        norm_e_new = np.linalg.norm(e_new)
        norm_e_candidates = np.linalg.norm(e_candidates, axis=1)
        # Handle zero norms by setting them to a small positive value
        norm_e_new = norm_e_new if norm_e_new != 0 else 1e-9
        norm_e_candidates[norm_e_candidates == 0] = 1e-9
        # Compute cosine similarity
        cosine = np.dot(e_candidates, e_new) / (norm_e_candidates * norm_e_new)
        # Find the index of the maximum cosine similarity
        idx_cand_user = np.argmax(cosine)
        return cosine, idx_cand_user

    def euc_dist(e_new, e_candidates):
        # Compute Euclidean distance
        distances = np.linalg.norm(e_candidates - e_new, axis=1)
        # Find the index of the minimum distance
        idx_cand_user = np.argmin(distances)
        return distances, idx_cand_user


    def distance(self, user_pref_embed):
        similarity, idx_user = self.cos_sim(user_pref_embed, self.embeds['user'])
        print('Cosine Similarity', similarity)
        distance, idx_user = self.euc_dist(user_pref_embed, self.embeds['user'])
        print("Euclidean distance:", distance)
        cand_user_emb = self.embeds['user'][idx_user]
        return idx_user, cand_user_emb
        
        
    def trim(self):
        pass
       
if __name__ == '__main__':
    init_emb = InitalUserEmbedding()
    
    #### Given user preferences from conversation
    user_acc_feature=[]
    user_rej_feature=[] 
    user_rej_items=[]
    
    #### Calculate initial user preference based on features
    user_pref_emb = init_emb.translation(
        user_acc_feature=user_acc_feature, 
        user_rej_feature=user_rej_feature, 
        user_rej_items=user_rej_items
    )
    
    #### Calculate who is the closest user
    idx_cand_user, cand_user_emb = init_emb.distance(user_pref_emb)
    
    ##Take cand_user_emb to Path Reasoning
        