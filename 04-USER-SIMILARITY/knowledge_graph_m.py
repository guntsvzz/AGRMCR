from __future__ import absolute_import, division, print_function

import json
from tqdm.auto import tqdm
from utils import *


class KnowledgeGraph(object):

    def __init__(
        self, dataset, kg_args, set_name, use_user_relations, use_entity_relations
    ):
        self.G = dict()
        self.kg_args = kg_args
        self.set_name = set_name  # train, test, validation
        self._load_entities(dataset)
        self._load_interactions(dataset)
        self._load_knowledge(dataset)
        if use_user_relations == True:
            self._load_user_info(dataset)
        if use_entity_relations == True:
            self._load_entity_info(dataset)
        self._clean()
        self._categories_offset() #offset value of category
        self._item_categories_offset()   #offset category value of item
        self._combine_features()     #create feature node
        self._item_features()        #create feature key of item
        self.top_matches = None
    
    def _load_entities(self, dataset):
        print("Load entities...")
        num_nodes = 0
        for entity in get_entities(self.kg_args):
            self.G[entity] = {}
            vocab_size = getattr(dataset, entity).vocab_size
            for eid in range(vocab_size):
                self.G[entity][eid] = {
                    r: [] for r in get_relations(self.kg_args, entity)
                }
            num_nodes += vocab_size
        print("Total {:d} nodes.".format(num_nodes))

    def _load_interactions(self, dataset):
        print("Load interactions...")
        num_edges = 0
        for eid, data in enumerate(dataset.interactions.data):
            uid, cid = data
            # (2) Add edges.
            self._add_edge("user", uid, self.kg_args.interaction, "item", cid)
            num_edges += 2

        print("Total {:d} interaction edges.".format(num_edges))

    def _load_knowledge(self, dataset):
        for relation in get_item_relations(self.kg_args):
            print("Load item knowledge {}...".format(relation))
            data = getattr(dataset, relation).data
            num_edges = 0
            for cid, eids in enumerate(data):
                if len(eids) <= 0:
                    continue
                if not self.use_item_relations(relation, cid, dataset):
                    continue
                if not self.use_user_relations(relation, cid, dataset):
                    continue
                for eid in set(eids):
                    et_type = get_entity_tail(self.kg_args, "item", relation)
                    self._add_edge("item", cid, relation, et_type, eid)
                    num_edges += 2
            print("Total {:d} {:s} edges.".format(num_edges, relation))

    def use_user_relations(self, relation, cid, dataset):
        use = True
        if relation in self.kg_args.user_relation:
            if self.set_name == "train" and cid not in dataset.cold_users["train"]:
                use = False
            if self.set_name == "validation" and cid in dataset.cold_users["test"]:
                use = False
            if self.set_name == "test" and cid in dataset.cold_users["validation"]:
                use = False
        return use

    def use_item_relations(self, relation, cid, dataset):
        use = True
        if relation in self.kg_args.item_relation:
            if self.set_name == "train" and cid not in dataset.cold_items["train"]:
                use = False
            if self.set_name == "validation" and cid in dataset.cold_items["test"]:
                use = False
            if self.set_name == "test" and cid in dataset.cold_items["validation"]:
                use = False
        return use

    def _load_user_info(self, dataset):
        for relation in get_user_relations(self.kg_args):
            print("Load user knowledge {}...".format(relation))
            data = getattr(dataset, relation).data
            num_edges = 0
            for lid, skids in enumerate(data):
                if len(skids) <= 0:
                    continue
                for eid in set(skids):
                    et_type = get_entity_tail(self.kg_args, "user", relation)
                    self._add_edge("user", lid, relation, et_type, eid)
                    num_edges += 2
            print("Total {:d} {:s} edges.".format(num_edges, relation))

    def _load_entity_info(self, dataset):
        for relation in get_entity_relations(self.kg_args):
            print("Load entity knowledge {}...".format(relation))
            data = getattr(dataset, relation).data
            eh_type = getattr(dataset, relation).eh_type
            et_ty = getattr(dataset, relation).et_type
            num_edges = 0
            for eh_id, et_ids in enumerate(data):
                if len(et_ids) <= 0:
                    continue
                for etid in set(et_ids):
                    et_type = get_entity_tail(self.kg_args, eh_type, relation)
                    assert et_ty == et_type
                    self._add_edge(eh_type, eh_id, relation, et_type, etid)
                    num_edges += 2
            print("Total {:d} {:s} edges.".format(num_edges, relation))

    def _add_edge(self, etype1, eid1, relation, etype2, eid2):
        self.G[etype1][eid1][relation].append(eid2)
        self.G[etype2][eid2][relation].append(eid1)

    def _rename_category_keys(self):
        def rename_keys(d, old_new_key_pairs):
            """Rename keys in a dictionary."""
            for old_key, new_key in old_new_key_pairs.items():
                if old_key in d:
                    d[new_key] = d.pop(old_key)

        old_new_key_pairs = {
            'category_of': 'belong_to',
            'interested_in': 'like'
        }

        # Iterate over the 'category' dictionary and rename keys
        for key, sub_dict in self.G.get('category', {}).items():
            rename_keys(sub_dict, old_new_key_pairs)
           
    def _combine_features(self):
        print("Combining brand and category into feature...")
        # Initialize feature dictionary
        self.G['feature'] = {}

        # Add brand entities
        for eid, relations in tqdm(self.G.get('brand', {}).items()):
            self.G['feature'][eid] = relations

        # Add category entities with renaming
        # category_offset = len(self.G.get('brand', {}))
        category_offset = 0 
        for eid, relations in tqdm(self.G.get('category', {}).items()):            
            # Rename keys in category relations
            renamed_relations = {}
            for old_key, new_key in {'category_of': 'belong_to', 'interested_in': 'like'}.items():
                if old_key in relations:
                    renamed_relations[new_key] = relations[old_key]
                else:
                    renamed_relations[old_key] = relations[old_key]
            # if eid in [1183, 1184, 1185, 1186, 1187]:
            #     print('relations', relations)
            #     print(category_offset + eid)
            #     print(renamed_relations)
            self.G['feature'][category_offset + eid] = renamed_relations
        # print("Number of eid category", eid)
        assert len(self.G['feature']) == len(self.G['brand']) + len(self.G['category']), "feature size are not equal to brand & category"
     
    def _categories_offset(self):
        print("Offsetting category keys...")
        offset = len(self.G.get('brand', {}))
        # Create a new dictionary to hold the updated keys
        new_category_dict = {}
        original_category_keys = list(self.G['category'].keys())
        for key in original_category_keys:
            new_key = key + offset
            # self.G['category'][new_key] = self.G['category'].pop(key)
            new_category_dict[new_key] = self.G['category'][key]
        # Replace the old dictionary with the new one
        self.G['category'] = new_category_dict
 
    def _item_categories_offset(self):
        print("Offsetting category_of values of item...")
        offset = len(self.G.get('brand', {}))
        for item_id, relations in self.G.get('item', {}).items():
            if 'category_of' in relations:
                self.G['item'][item_id]['category_of'] = tuple(eid + offset for eid in relations['category_of'])
                
    def _item_features(self):
        print("Combining belong_to and category_of into feature for items...")
        for item_id in self.G.get('item', {}):
            belong_to = self.G['item'][item_id].get('belong_to', [])
            category_of = self.G['item'][item_id].get('category_of', [])
            self.G['item'][item_id]['feature'] = belong_to + category_of
    
    def _clean(self):
        print("Remove duplicates...")
        for etype in self.G:
            for eid in self.G[etype]:
                for r in self.G[etype][eid]:
                    data = self.G[etype][eid][r]
                    data = tuple(sorted(set(data)))
                    self.G[etype][eid][r] = data

    def compute_degrees(self):
        print("Compute node degrees...")
        self.degrees = {}
        self.max_degree = {}
        for etype in self.G:
            self.degrees[etype] = {}
            for eid in self.G[etype]:
                count = 0
                for r in self.G[etype][eid]:
                    count += len(self.G[etype][eid][r])
                self.degrees[etype][eid] = count

    def get(self, eh_type, eh_id=None, relation=None):
        data = self.G
        if eh_type is not None:
            data = data[eh_type]
        if eh_id is not None:
            data = data[eh_id]
        if relation is not None:
            data = data[relation]
        return data

    def __call__(self, eh_type, eh_id=None, relation=None):
        # print(eh_type, eh_id, relation)
        return self.get(eh_type, eh_id, relation)

    def get_tails(self, entity_type, entity_id, relation):
        return self.G[entity_type][entity_id][relation]

    def get_tails_given_user(self, entity_type, entity_id, relation, user_id):
        """Very important!
        :param entity_type:
        :param entity_id:
        :param relation:
        :param user_id:
        :return:
        """
        tail_type = self.kg_args.kg_relation[entity_type][relation]
        tail_ids = self.G[entity_type][entity_id][relation]
        if tail_type not in self.top_matches:
            return tail_ids
        top_match_set = set(self.top_matches[tail_type][user_id])
        top_k = len(top_match_set)
        if len(tail_ids) > top_k:
            tail_ids = top_match_set.intersection(tail_ids)
        return list(tail_ids)

    def trim_edges(self):
        degrees = {}
        for entity in self.G:
            degrees[entity] = {}
            for eid in self.G[entity]:
                for r in self.G[entity][eid]:
                    if r not in degrees[entity]:
                        degrees[entity][r] = []
                    degrees[entity][r].append(len(self.G[entity][eid][r]))

        for entity in degrees:
            for r in degrees[entity]:
                tmp = sorted(degrees[entity][r], reverse=True)
                print(entity, r, tmp[:10])

    def set_top_matches(self, u_u_match, u_p_match):
        self.top_matches = {
            "user": u_u_match,
            "item": u_p_match,
        }
