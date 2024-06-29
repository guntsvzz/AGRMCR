from tqdm.auto import tqdm
from utils import *

path="../../data/cell_phones/Amazon_Cellphones_01_01"
set_name="train"
embeds = load_embed(path, set_name)
dataset = load_dataset(path, set_name)
kg = load_kg(path, set_name)
##### ================================================= ####
def get_feature(idx):
    # Replace with your actual logic to determine if idx is a brand or category
    # Check if idx is a brand
    brands = [979, 1001, 1042, 1032, 992, 973]  # Example brand indices
    categories = [1020, 1002, 1062, 1031, 991, 969, 990]  # Example category indices
    if idx in brands:
        return 'brand'
    elif idx in categories:
        return 'category'
    else:
        return None  # If idx is neither brand nor category

# Example lists
user_acc_feature = [979, 1001, 1020]  # Example indices
user_rej_feature = [1042, 1032, 992, 973, 1062, 1031, 991, 969, 990]  # Example indices
user_rej_items   = [1, 2,3]  # Example items

# Create an empty user dictionary with the same keys and empty lists as values
user_preferred = {key: [] for key in dataset.data_args.kg_relation.user.keys()}
user_preferred['non-purchase'] = []
user_preferred['dislike'] = []
user_preferred['disinterested_in'] = []

# Process user_acc_feature
for idx in user_acc_feature:
    preference = get_feature(idx)
    if preference == 'brand':
        user_preferred['like'].append(idx)
    elif preference == 'category':
        user_preferred['interested_in'].append(idx)

# Process user_rej_feature
for idx in user_rej_feature:
    preference = get_feature(idx)
    if preference == 'brand':
        user_preferred['dislike'].append(idx)
    elif preference == 'category':
        user_preferred['disinterested_in'].append(idx)

# Assign user_rej_items to non-purchase
user_preferred['non-purchase'] = user_rej_items
##### ================================================= ####

##### ================================================= ####
# Create a empty user dictionary with the same keys and empty lists as values
user_graph = {key: None for key in dataset.data_args.kg_relation.user.keys()}
user_embed = {value: [] for value in dataset.data_args.kg_relation.user.values()}
# user_graph
# {'purchase': None, 'mentioned': None, 'interested_in': None, 'like': None}
# user_embed
# {'item': [], 'word': [], 'category': [], 'brand': []})

# Update a user dictionary 
user_idx = 0
for relation, entity in dataset.data_args.kg_relation.user.items():
    print(f'Relation : {relation} | Entity : {entity} | Number {len(kg.G["user"][user_idx][relation])}' )
    user_graph[relation] = kg.G['user'][user_idx][relation]

user_graph.keys() #dict_keys(['purchase', 'mentioned', 'interested_in', 'like'])

for idx_list, (relation, entity) in zip(user_graph.values(), dataset.data_args.kg_relation.user.items()):
    for each_idx in idx_list:
        user_embed[entity].append(embeds[entity][each_idx])

user_embed['user'] = [embeds['user'][user_idx]]
user_embed.keys() #dict_keys(['item', 'word', 'category', 'brand'])
##### ================================================= ####

##### ================================================= ####
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Flatten the embeddings into a single list and keep track of labels
embeddings = []
labels = []

for key, vectors in user_embed.items():
    embeddings.extend(vectors)
    labels.extend([key] * len(vectors))

# Convert to numpy array for t-SNE
embeddings = np.array(embeddings)

# Perform t-SNE to reduce dimensions to 2
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Define colors for each key
colors = {
    'item': 'r',
    'word': 'g',
    'category': 'b',
    'brand': 'y',
    'user' : 'k'
}

# Plot the embeddings
plt.figure(figsize=(10, 8))
for key in user_embed.keys():
    idxs = [i for i, label in enumerate(labels) if label == key]
    plt.scatter(embeddings_2d[idxs, 0], embeddings_2d[idxs, 1], c=colors[key], label=key, alpha=0.6)

plt.legend()
plt.title("User Embeddings Visualization")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()
##### ================================================= ####