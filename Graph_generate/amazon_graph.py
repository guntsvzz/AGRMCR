from tqdm import tqdm
import json
import pickle
import os

class AmazonGraph(object):
    def __init__(self, domain):
        self.domain = domain  
        self.G = dict()  
        
        with open(f'./data_preprocess/tmp/item_feature_{self.domain}.json','r') as f:
            self.item_feature = json.load(f)
        with open(f'./data_preprocess/tmp/small_to_large_{self.domain}.json','r') as f:
            self.small_to_large = json.load(f)             
        with open(f'./data_preprocess/tmp/feature2id_{self.domain}.json','r') as f:
            self.feature2id = json.load(f)             
            
        self.__get_user__()
        self.__get_item__()
        self.__get_feature__()
            
    def __get_user__(self):
        with open(f'./data_preprocess/tmp/review_dict_valid_{self.domain}.json', 'r', encoding='utf-8') as f:
            ui_train=json.load(f)
            self.G['user']={}
            for user in tqdm(ui_train):
                self.G['user'][int(user)]={}
                # Filter out string elements from the interact tuple
                filtered_interact = tuple(x for x in ui_train[user] if not isinstance(x, str))
                self.G['user'][int(user)]['interact']=filtered_interact
                # self.G['user'][int(user)]['interact']=tuple(ui_train[user])
                self.G['user'][int(user)]['friends']=tuple(())
                self.G['user'][int(user)]['like']=tuple(())
                
    def __get_item__(self):
        self.G['item']={}
        feature_index={}
        i = 0
        for value in self.feature2id.values():
            if value in feature_index:
                continue
            else:
                feature_index[value]= i
                i+=1
                
        for item in tqdm(self.item_feature):
            self.G['item'][int(item)]={} 
            fea=[]
            for feature in self.item_feature[item]: 
                fea.append(feature_index[feature])
            
            self.G['item'][int(item)]['belong_to']=tuple(set(fea))
            self.G['item'][int(item)]['interact']=tuple(())
            self.G['item'][int(item)]['belong_to_large']=tuple(())
            
        for user in self.G['user']:
            for item in self.G['user'][user]['interact']:
                if type(item) == str:
                    continue
                self.G['item'][item]['interact']+=tuple([user])
                
    def __get_feature__(self):
        self.G['feature']={}
        feature_index={}
        i = 0
        for value in self.feature2id.values():
            if value in feature_index:
                continue
            else:
                feature_index[value]= i
                i+=1
                        
        self.feature2id = {value: key for key, value in self.feature2id.items()}

        for key in tqdm(self.feature2id.keys()):
            idx = feature_index[int(key)]
            self.G['feature'][idx]={}
            try:
                self.G['feature'][idx]['link_to_feature'] = tuple([self.small_to_large[str(key)]])
            except KeyError:
                self.G['feature'][idx]['link_to_feature'] = tuple()
            self.G['feature'][idx]['like']=tuple(())
            self.G['feature'][idx]['belong_to']=tuple(())
            
        for item in self.G['item']:
            for feature in self.G['item'][item]['belong_to']:
                self.G['feature'][feature]['belong_to']+=tuple([item])
                
if __name__ == '__main__':  
    dataset_name = 'Office_Products' 
    kg = AmazonGraph(dataset_name)    
    print('Node of User', len(kg.G['user']))
    print('Node of Item,',len(kg.G['item']))
    print('NOde of Feature', len(kg.G['feature']))
    
    # Create directory if it doesn't exist
    output_dir = '../tmp/amazon/'
    os.makedirs(output_dir, exist_ok=True)

    # Save the object
    with open(os.path.join(output_dir, 'kg.pkl'), 'wb') as f:
        pickle.dump(kg, f)