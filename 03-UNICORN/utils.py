import pickle
import numpy as np
import random
import torch
import os
import sys
# from knowledge_graph import KnowledgeGraph
# from data_process import LastFmDataset
# from KG_data_generate.lastfm_small_data_process import LastFmSmallDataset
# from KG_data_generate.lastfm_knowledge_graph import KnowledgeGraph
# from Graph_generate.knowledge_graph import KnowledgeGraph
# from Graph_generate.knowledge_graph_m import KnowledgeGraph
from knowledge_graph_m import KnowledgeGraph

import logging
import logging.handlers
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
import wandb

#Dataset names
LAST_FM = 'LAST_FM'
LAST_FM_STAR = 'LAST_FM_STAR'
YELP = 'YELP'
YELP_STAR = 'YELP_STAR'
AMAZON = 'AMAZON'
AMAZON_STAR = 'AMAZON_STAR'
BEAUTY = 'BEAUTY'
CELLPHONES = 'CELLPHONES'
CLOTH = 'CLOTH'
CDS = 'CDS'
    
DATA_DIR = {
    LAST_FM     : './data/lastfm',
    YELP        : './data/yelp',
    LAST_FM_STAR: './data/lastfm_star',
    YELP_STAR   : './data/yelp',
    AMAZON      : './data/amazon',
    AMAZON_STAR : './data/amazon_star',
    BEAUTY      : './data/beauty',
    CELLPHONES  : './data/cellphones',
    CLOTH       : './data/cloth',
    CDS         : './data/cds',
}
TMP_DIR = {
    LAST_FM     : './tmp/last_fm',
    YELP        : './tmp/yelp',
    LAST_FM_STAR: './tmp/last_fm_star',
    YELP_STAR   : './tmp/yelp_star',
    AMAZON      : './tmp/amazon',
    AMAZON_STAR : './tmp/amazon_star',
    BEAUTY      : './tmp/beauty',
    CELLPHONES  : './tmp/cellphones',
    CLOTH       : './tmp/cloth',
    CDS         : './tmp/cds',
}
def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var

def save_dataset(dataset, dataset_obj):
    dataset_file = TMP_DIR[dataset] + '/dataset.pkl'
    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset_obj, f)

def load_dataset(dataset, mode='train'):
    dataset_file = TMP_DIR[dataset] + f'/{mode}_dataset.pkl'
    dataset_obj = pickle.load(open(dataset_file, 'rb'))
    return dataset_obj

def save_kg(dataset, kg):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    pickle.dump(kg, open(kg_file, 'wb'))

def load_kg(dataset, mode='train'):
    kg_file = TMP_DIR[dataset] + f'/{mode}_kg.pkl'
    kg = pickle.load(open(kg_file, 'rb'))
    return kg

def load_graph(dataset, mode='train'):
    graph_file = TMP_DIR[dataset] + f'/{mode}_graph.pkl'
    graph = pickle.load(open(graph_file, 'rb'))
    return graph

def save_graph(dataset, graph):
    graph_file = TMP_DIR[dataset] + '/graph.pkl'
    pickle.dump(graph, open(graph_file, 'wb'))


def load_embed(dataset, embed, epoch, mode='train'):
    if embed:
        # path = TMP_DIR[dataset] + '/embeds/' + '{}.pkl'.format(embed)
        path = TMP_DIR[dataset] + f'/{mode}_{embed}_embed.pkl'
    else:
        return None
    with open(path, 'rb') as f:
        embeds = pickle.load(f)
        print('{} Embedding load successfully!'.format(embed))
        return embeds

def load_rl_agent(dataset, filename, epoch_user, device):
    model_file = TMP_DIR[dataset] + '/RL-agent/' + filename + '-epoch-{}.pkl'.format(epoch_user)
    model_dict = torch.load(model_file, map_location=torch.device(device))
    print('RL policy model load at {}'.format(model_file))
    return model_dict

def save_rl_agent(dataset, model, filename, epoch_user):
    model_file = TMP_DIR[dataset] + '/RL-agent/' + filename + '-epoch-{}.pkl'.format(epoch_user)
    if not os.path.isdir(TMP_DIR[dataset] + '/RL-agent/'):
        os.makedirs(TMP_DIR[dataset] + '/RL-agent/')
    torch.save(model, model_file)
    print('RL policy model saved at {}'.format(model_file))


def save_rl_mtric(dataset, filename, epoch, SR, spend_time, mode='train'):
    PATH = TMP_DIR[dataset] + '/RL-log-merge/' + filename + '.txt'
    if not os.path.isdir(TMP_DIR[dataset] + '/RL-log-merge/'):
        os.makedirs(TMP_DIR[dataset] + '/RL-log-merge/')
    if mode == 'train':
        with open(PATH, 'a') as f:
            f.write('===========Train===============\n')
            f.write('Starting {} user epochs\n'.format(epoch))
            f.write('training SR@5: {}\n'.format(SR[0]))
            f.write('training SR@10: {}\n'.format(SR[1]))
            f.write('training SR@15: {}\n'.format(SR[2]))
            f.write('training Avg@T: {}\n'.format(SR[3]))
            f.write('training hDCG: {}\n'.format(SR[4]))
            f.write('Spending time: {}\n'.format(spend_time))
            f.write('================================\n')
            # f.write('1000 loss: {}\n'.format(loss_1000))
    elif mode == 'test':
        with open(PATH, 'a') as f:
            f.write('===========Test===============\n')
            f.write('Testing {} user tuples\n'.format(epoch))
            f.write('Testing SR@5: {}\n'.format(SR[0]))
            f.write('Testing SR@10: {}\n'.format(SR[1]))
            f.write('Testing SR@15: {}\n'.format(SR[2]))
            f.write('Testing Avg@T: {}\n'.format(SR[3]))
            f.write('Testing hDCG: {}\n'.format(SR[4]))
            f.write('Testing time: {}\n'.format(spend_time))
            f.write('================================\n')
            # f.write('1000 loss: {}\n'.format(loss_1000))

def save_rl_model_log(dataset, filename, epoch, epoch_loss, train_len):
    PATH = TMP_DIR[dataset] + '/RL-log-merge/' + filename + '.txt'
    if not os.path.isdir(TMP_DIR[dataset] + '/RL-log-merge/'):
        os.makedirs(TMP_DIR[dataset] + '/RL-log-merge/')
    with open(PATH, 'a') as f:
        f.write('Starting {} epoch\n'.format(epoch))
        f.write('training loss : {}\n'.format(epoch_loss / train_len))
        # f.write('1000 loss: {}\n'.format(loss_1000))

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def set_cuda(args):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    devices_id = [int(device_id) for device_id in args.gpu.split()]
    device = (
        torch.device("cuda:{}".format(str(devices_id[0])))
        if use_cuda
        else torch.device("cpu")
    )
    return device, devices_id



def get_entities(args):
    return list(args.kg_relation.keys())


def get_relations(args, entity_head):
    return list(args.kg_relation[entity_head].keys())


def get_entity_tail(args, entity_head, relation):
    return args.kg_relation[entity_head][relation]


def get_item_relations(args):
    return args.item_relation.keys()


def compute_tfidf_fast(vocab, docs):
    """Compute TFIDF scores for all vocabs.

    Args:
        docs: list of list of integers, e.g. [[0,0,1], [1,2,0,1]]

    Returns:
        sp.csr_matrix, [num_docs, num_vocab]
    """
    # (1) Compute term frequency in each doc.
    data, indices, indptr = [], [], [0]
    for d in docs:
        term_count = {}
        for term_idx in d:
            if term_idx not in term_count:
                term_count[term_idx] = 1
            else:
                term_count[term_idx] += 1
        indices.extend(term_count.keys())
        data.extend(term_count.values())
        indptr.append(len(indices))
    tf = sp.csr_matrix(
        (data, indices, indptr), dtype=int, shape=(len(docs), len(vocab))
    )

    # (2) Compute normalized tfidf for each term/doc.
    transformer = TfidfTransformer(smooth_idf=True)
    tfidf = transformer.fit_transform(tf)
    return tfidf


def get_logger(logname):
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s]  %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_labels(path, labels, mode="train", use_wandb=False):
    if mode not in ["train", "test", "validation"]:
        raise Exception("mode should be one of {train, test, validation}.")
    label_file = f"{path}/{mode}_label.pkl"
    with open(label_file, "wb") as f:
        pickle.dump(labels, f)
    if use_wandb:
        wandb.save(label_file)


def load_labels(path, mode="train"):
    if mode not in ["train", "test", "validation"]:
        raise Exception("mode should be one of {train, test, validation}.")
    label_file = f"{path}/{mode}_label.pkl"
    user_products = pickle.load(open(label_file, "rb"))
    return user_products

def create_data_file(data_dir, data, file_name):
    with open(data_dir + "/" + file_name, "w") as f:
        for d in data:
            f.write(d)


def get_ordered_item_relations(args):
    entities = list(filter(lambda x: x not in ["user", "item"], args.entities))
    item_relations = []

    relation_tail = list(map(lambda r: (r[0], r[1][1]), args.item_relation.items()))
    for e in entities:
        for rt in relation_tail:
            if e == rt[1]:
                item_relations.append(rt[0])
                break

    return item_relations


def get_ordered_item_relations_et(args):
    entities = list(filter(lambda x: x not in ["user", "item"], args.entities))
    tail_entities = []

    relation_tail = list(map(lambda r: (r[0], r[1][1]), args.item_relation.items()))
    for e in entities:
        for rt in relation_tail:
            if e == rt[1]:
                tail_entities.append(rt[1])
                break

    return tail_entities


def get_ordered_user_relations(args):
    entities = list(filter(lambda x: x not in ["user", "item"], args.entities))
    user_relations = []

    relation_tail = list(map(lambda r: (r[0], r[1][1]), args.user_relation.items()))
    for e in entities:
        for rt in relation_tail:
            if e == rt[1]:
                user_relations.append(rt[0])
                break

    return user_relations


def get_ordered_user_relations_et(args):
    entities = list(filter(lambda x: x not in ["user", "item"], args.entities))
    tail_entities = []

    relation_tail = list(map(lambda r: (r[0], r[1][1]), args.user_relation.items()))
    for e in entities:
        for rt in relation_tail:
            if e == rt[1]:
                tail_entities.append("ur_" + rt[1])
                break

    return tail_entities


def get_ordered_entity_relations(args):
    entities = list(filter(lambda x: x not in ["user", "item"], args.entities))
    entity_relations = []

    relation_tail = list(map(lambda r: (r[0], r[1][1]), args.entity_relation.items()))
    for e in entities:
        for rt in relation_tail:
            if e == rt[1]:
                entity_relations.append(rt[0])
                break

    return entity_relations


def get_ordered_entity_relations_et(args):
    entities = list(filter(lambda x: x not in ["user", "item"], args.entities))
    tail_entities = []

    relation_tail = list(map(lambda r: (r[0], r[1][2]), args.entity_relation.items()))
    for e in entities:
        for rt in relation_tail:
            if e == rt[1]:
                tail_entities.append("er_" + rt[1])
                break

    return tail_entities


def get_user_relations(args):
    if args.get("user_relation", None):
        user_relations = args.user_relation.keys()
        return user_relations


def get_entity_relations(args):
    if args.get("entity_relation", None):
        entity_relation = args.entity_relation.keys()
        return entity_relation


def get_batch_entities(kg_args):
    batch_entities = ["user", "item"]
    batch_entities.extend(get_ordered_item_relations_et(kg_args))
    batch_entities.extend(get_ordered_user_relations_et(kg_args))
    batch_entities.extend(get_ordered_entity_relations_et(kg_args))
    return batch_entities
