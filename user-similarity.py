####Pseudo Code

#MCR
## user's profile
user_acc_feature = test_env.user_acc_feature 
user_rej_feature = test_env.user_rej_feature 
cand_items = test_env.cand_items 

def initialize_embed(user_acc_feature, user_rej_feature, cand_items, mode='posneg'):
    if mode == 'null':
        pass
    elif mode == 'avg':
        pass
    elif mode == 'posneg':
        pass
    embed = ...
    return embed

## initialize_embed
new_user_embeds = initialize_embed(user_acc_feature, user_rej_feature, cand_items, mode='posneg')

#Use-Sim
import torch
import torch.nn.functional as F
## Define the similarity function
def sim_function(e_new,e_candidates):
    # Calculate the cosine similarity
    cos_sim = F.cosine_similarity(e_new.unsqueeze(0), e_candidates)
    return cos_sim

## Define the function to find the best matching candidate embedding
def find_best_match(e_new, e_candidates):
    # Calculate similarity scores
    similarity_scores = sim_function(e_new, e_candidates)
    # Find the index of the maximum similarity score
    best_match_index = torch.argmax(similarity_scores)
    # Retrieve the best matching candidate embedding
    best_match_embedding = e_candidates[best_match_index]
    return best_match_embedding, similarity_scores[best_match_index]

### Example embeddings
e_new = torch.tensor([1.0, 2.0])
e_candidates = torch.tensor([
    [3.0, 4.0],
    [1.0, 0.5],
])

### Find the best matching candidate embedding
best_match_embedding, best_match_score = find_best_match(e_new, e_candidates)

# Graph Reasoning (GR)
candidate_user = best_match_embedding
path_reasoning = RL_agent(candidate_user)

# Trim
def trim_embed(path_reasoning, user_acc_feature, user_rej_feature):
    new_path_reasoning = ...
    return new_path_reasoning
    
new_path_reasoning = trim_embed(path_reasoning, user_acc_feature, user_rej_feature)