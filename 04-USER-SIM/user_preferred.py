from tqdm.auto import tqdm
from utils import *

path="../../data/cell_phones/Amazon_Cellphones_01_01"
set_name="train"
embeds = load_embed(path, set_name)
dataset = load_dataset(path, set_name)
kg = load_kg(path, set_name)

def get_feature(idx):
    # Replace with your actual logic to determine if idx is a brand or category
    offset = len(kg.G['brand'])
    if idx > offset:
        return 'category'
    else:
        return 'brand'

# Example lists
user_acc_feature = [2, 979, 1001, 1020]  # Example indices
user_rej_feature = [1042, 1032, 992, 973, 1062, 1031, 991, 969, 990]  # Example indices
user_rej_items   = [1, 2, 3]  # Example items

# Create an empty user dictionary with the same keys and empty lists as values
user_preferred = {key: [] for key in dataset.data_args.kg_relation.user.keys()}
user_preferred['non-purchase'] = []
user_preferred['dislike'] = []
user_preferred['disinterested_in'] = []

offset = len(kg.G['brand'])
# Process user_acc_feature
for idx in user_acc_feature:
    preference = get_feature(idx)
    if preference == 'brand':
        user_preferred['like'].append(idx)
    elif preference == 'category':
        user_preferred['interested_in'].append(idx-offset)

# Process user_rej_feature
for idx in user_rej_feature:
    preference = get_feature(idx)
    if preference == 'brand':
        user_preferred['dislike'].append(idx)
    elif preference == 'category':
        user_preferred['disinterested_in'].append(idx-offset)

# Assign user_rej_items to non-purchase
user_preferred['non-purchase'] = user_rej_items
# Output the updated dictionary
# user_preferred