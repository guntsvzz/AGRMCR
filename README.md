# AGRMCR - Adapting Graph Reasoning for Explainable Cold Start Recommendation on Multi-Round Conversation Recommendation

## Data Preparation

<details>

<summary>Datasets</summary>

Four Amazon datasets (Amazon_Beauty, Amazon_Electronics, Amazon_Office_Products, Amazon_Home_and_Kitchen) are available in the "data_preprocess/raw_data/" directory and the split is consistent with [1].

All four datasets used in this paper can be downloaded below
-  [Metadata - Amazon dataset v2018](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)
- [Rating - Amazon dataset v2018](https://nijianmo.github.io/amazon/index.html)

### Summary statistics of datasets.

We 
Following ScomGNN, filter categories more than 4 categories & pricing is not empty
Here's the updated table with the new columns added:

| Category             | Reviews              | Metadata             | Filter Metadata      | Filter Usesr | Type | Instance | Brand |
|----------------------|----------------------|----------------------|----------------------|--------------|------|----------|-------|
| Appliances           | 602,777 reviews      | 30,459 products      | 804 products         | xx           | xx   | xx       | xx    |
| Electronics          | 20,994,353 reviews   | 786,868 products     | 46,611 products      | xx           | xx   | xx       | xx    |
| Grocery and Gourmet Food| 5,074,160 reviews | 287,209 products     | 38,548 products      | xx           | xx   | xx       | xx    |
| Home and Kitchen     | 21,928,568 reviews   | 1,301,225 products   | 75,514 products      | xx           | xx   | xx       | xx    |
| Office Products      | 5,581,313 reviews    | 315,644 products     | 42,785 products      | xx           | xx   | xx       | xx    |
| Sports and Outdoors  | 12,980,837 reviews   | 962,876 products     | 87,076 products      | xx           | xx   | xx       | xx    |

### Summary statistics of preprocessed datasets.

|                   | Appliances | Electronics | Grocery and Gourmet Food | Home and Kitchen | Office Products | Sports and Outdoors |
|-------------------|------------|-------------|--------------------------|------------------|-----------------|---------------------|
| **#Users**        | xx         | xx          | xx                       | xx               | xx              | xx                  |
| **#Items**        | xx         | xx          | xx                       | xx               | xx              | xx                  |
| **#Interactions** | xx         | xx          | xx                       | xx               | xx              | xx                  |
| **#Attributes**   | xx         | xx          | xx                       | xx               | xx              | xx                  |
| **#Entities**     | xx         | xx          | xx                       | xx               | xx              | xx                  |
| **#Relations**    | xx         | xx          | xx                       | xx               | xx              | xx                  |
| **#Triplets**     | xx         | xx          | xx                       | xx               | xx              | xx                  |


### Entities and Relations
```bash
Head    -> Relation         -> Tail
1. USER -> INTERACT         -> ITEM
2. ITEM -> ALSO_BUY         -> ITEM 
3. ITEM -> ALSO_VIEW        -> ITEM
4. ITEM -> BELONG_TO        -> FEATURE(TYPE)
5. ITEM -> PRODUCE_BY       -> FEATURE(BRAND)
6. ITEM -> DESCRIBED_BY     -> FEATURE(FUNCTION)
7. TYPE -> BELONG_TO_LARGE  -> CATEGORIES
```

</details>

<details>

<summary> Graph Formating </summary>

```bash
brand_dict.json
{
    "brandA" : 0,
    "brandB" : 1,
    ...
}
feature2id.json
{
    "brand1"    : 0,
    "brand2"    : 1,
    ...
    "brandX"    : x,
    ...
    "category1" :  x+1,
    "category2" :  x+2,
    ...
    "category"  :  x+y,
    ...
    "type1"     :  x+y+1,
    "type2"     :  x+y+2,
    ...
    "typez"     :  x+y+z,


}
```

```bash
user_dict.json
{
    "0" : {
        "friend" : [],
        "like" : [],
        "interact" [item(idx), item(idx),...]
    },
    "1" : {
        "friend" : [],
        "like" : [],
        "interact" []
    },
    ...
}
```

```bash
item_dict.json
{
    "0" : {
        "categories" : [int, int, int],
        "brand" : int,
        "feature_index" [int, int, int]
        "asin" : str
    },
    "1" : {
        "categories" : [int, int, int],
        "brand" : int,
        "feature_index" [int, int, int]
        "asin" : str
    },
    ...
}
```

```bash
feature_dict.json
{
    "0" : {
        "link_to_feature" : [],
        "like" : [],
        "belong_to" [item(idx), item(idx),...]
    },
    "1" : {
        "link_to_feature" : [],
        "like" : [],
        "belong_to" [item(idx), item(idx),...]
    },
    ...
}
```

```bash
first-layer_merged_tag_map.json
{
    "categories#1" : 0,
    "categories#2" : 0,
    ...
}
```

```bash
second-layer_oringinal_tag_map.json
{
    "type#1" : 0,
    "type#2" : 0,
    ...
}
```

```bash
2-layer taxonomy.json
{
    "Categories#1": [TypeA, TypeB, TypeC],
    "Categories#2": [TypeD, TypeE, TypeF],
}
```

</details>

## Requirement 
```bash
pip install -r requirements.txt
```

## How to run the code

### TransE Embedding
```bash
```

### Training
```bash
python3 RL_model.py --data_name AMAZON --domain Appliances --max_steps 10 --sample_times 1 
```
### Evaluation
```bash
python3 evaluate.py --data_name AMAZON --domain Appliances --load_rl_epoch 10
```
## Citation
```bash
```


## References
[1] Yongfeng Zhang, Qingyao Ai, Xu Chen, W. Bruce Croft. "Joint Representation Learning for Top-N Recommendation with Heterogeneous Information Sources". In Proceedings of CIKM. 2017.

[2] Yikun Xian, Zuohui Fu, S. Muthukrishnan, Gerard de Melo, Yongfeng Zhang. "Reinforcement Knowledge Graph Reasoning for Explainable Recommendation." In Proceedings of SIGIR. 2019.

[3] Yang Deng, Yaliang Li, Fei Sun, Bolin Ding, Wai Lam. "Unified Conversational Recommendation Policy Learning via Graph-based Reinforcement Learning." arXiv preprint arXiv:2105.09710, 2021.

[4] Jibril Frej, Marta Knezevic, Tanja Kaser. "Graph Reasoning for Explainable Cold Start Recommendation." arXiv preprint arXiv:2406.07420, 2024.