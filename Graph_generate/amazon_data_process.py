import json
from easydict import EasyDict

class AmazonDataset(object):
    def __init__(self, domain):
        self.domain = domain
        self.user    = f'./data_preprocess/tmp/user_dict_{self.domain}.json'
        self.item    = f'./data_preprocess/tmp/item_dict_{self.domain}.json'
        self.feature = f'./data_preprocess/tmp/feature2id_{self.domain}.json'
        with open(self.user) as f:
            user_data = json.load(f)
        with open(self.item) as f:
            item_data = json.load(f)
        with open(self.feature) as f:
            entity_data = json.load(f)
        
        # Assuming user_data is your original dictionary
        user_data = {int(key): user_data[key] for key in user_data.keys()}
        item_data = {int(key): item_data[key] for key in item_data.keys()}
        entity_data = {value: key for key, value in entity_data.items()}
        entity_id = {int(key): entity_data[key] for key in entity_data.keys()}

        entity_id=list(user_data)
        m=EasyDict(id=entity_id, value_len=max(entity_id)+1)
        setattr(self,'user',m)
        
        entity_id=list(item_data)
        m=EasyDict(id=entity_id, value_len=max(entity_id)+1)
        setattr(self,'item',m)
        
        entity_id=list(entity_data)
        m=EasyDict(id=entity_id, value_len=max(max(entity_id)+1,988))
        setattr(self,'feature',m)
        
if __name__ == '__main__': 
    domain =  'Appliances'
    dataset = AmazonDataset(domain)
    print('Dataset Name', dataset.domain)
    print('Number of User :', dataset.user.value_len) #4905
    print('Number of Item :', dataset.item.value_len) #42784 
    print('Number of Feature :', dataset.feature.value_len)