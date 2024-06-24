# AGRMCR - Adapting Graph Reasoning for Explainable Cold Start Recommendation on Multi-Round Conversation Recommendation

## Requirement 
```bash
pip install -r requirements.txt
```

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

## How to run the code

## JRL

<details>
<summary>Preprocessing</summary>

```bash
python3 index_and_filter_review_file.py
python3 match_cate_brand_related.py
```
</details>

## GRECS

<details>
<summary>Graph construction</summary>

### Preprocessing Dataset
```bash
python3 src/preprocess/beauty.py \
    --config config/beauty/graph_reasoning/preprocess.json
python3 src/preprocess/cds.py \
    --config config/cds/graph_reasoning/preprocess.json
python3 src/preprocess/cellphones.py \
    --config config/cellphones/graph_reasoning/preprocess.json
python3 src/preprocess/clothing.py \
    --config config/clothing/graph_reasoning/preprocess.json
```

### Make Dataset
```bash
python3 src/graph_reasoning/make_dataset.py \
    --config config/beauty/graph_reasoning/UPGPR.json
python3 src/graph_reasoning/make_dataset.py \
    --config config/cds/graph_reasoning/UPGPR.json
python3 src/graph_reasoning/make_dataset.py \
    --config config/cellphones/graph_reasoning/UPGPR.json
python3 src/graph_reasoning/make_dataset.py \
    --config config/clothing/graph_reasoning/UPGPR.json
```

</details>

<details>

<summary>TransE Embedding</summary>

### TransE Embedding
```bash
python3 src/graph_reasoning/train_transe_model.py \
    --config config/beauty/graph_reasoning/UPGPR.json
python3 src/graph_reasoning/train_transe_model.py \
    --config config/cds/graph_reasoning/UPGPR.json
python3 src/graph_reasoning/train_transe_model.py \
    --config config/cellphones/graph_reasoning/UPGPR.json
python3 src/graph_reasoning/train_transe_model.py \
    --config config/clothing/graph_reasoning/UPGPR.json
```
</details>

<details>
<summary>Train & Evaluation RL agent</summary>

### Train RL 
```bash
python3 src/graph_reasoning/train_agent.py \
    --config config/beauty/graph_reasoning/UPGPR.json
python3 src/graph_reasoning/train_agent.py \
    --config config/cds/graph_reasoning/UPGPR.json
python3 src/graph_reasoning/train_agent.py \
    --config config/cellphones/graph_reasoning/UPGPR.json
python3 src/graph_reasoning/train_agent.py \
    --config config/clothing/graph_reasoning/UPGPR.json
```

### Evaluation
```bash
python3 src/graph_reasoning/test_agent.py \
    --config config/beauty/graph_reasoning/UPGPR.json
python3 src/graph_reasoning/test_agent.py \
    --config config/cds/graph_reasoning/UPGPR.json
python3 src/graph_reasoning/test_agent.py \
    --config config/cellphones/graph_reasoning/UPGPR.json
python3 src/graph_reasoning/test_agent.py \
    --config config/clothing/graph_reasoning/UPGPR.json
python3 src/graph_reasoning/test_agent.py \
    --config config/coco/graph_reasoning/UPGPR.json
```
</details>

## UNICORN

<details>
<summary>Train & Evaluation RL agent</summary>

### Training
```bash
python3 RL_model.py --data_name AMAZON --domain beauty --max_steps 10 --sample_times 1 
python3 RL_model.py --data_name AMAZON --domain cds --max_steps 10 --sample_times 1 
python3 RL_model.py --data_name AMAZON --domain cellphones --max_steps 10 --sample_times 1 
python3 RL_model.py --data_name AMAZON --domain clothing --max_steps 10 --sample_times 1 
```

### Evaluation
```bash
python3 evaluate.py --data_name AMAZON --domain beauty --load_rl_epoch 10
python3 evaluate.py --data_name AMAZON --domain Appliances --load_rl_epoch 10
python3 evaluate.py --data_name AMAZON --domain cellphones --load_rl_epoch 10
python3 evaluate.py --data_name AMAZON --domain clothing --load_rl_epoch 10
```

</details>

## Citation
Todsavad Tangtortan. 2024.Adapting Graph Reasoning for Explainable Cold Start Recommendation on Multi-Round Conversation Recommendation (AGRMCR). AIT, Thailand.

## References
[1] Yongfeng Zhang, Qingyao Ai, Xu Chen, and W. Bruce Croft. 2017. Joint Representation Learning for Top-N Recommendation with Heterogeneous Information Sources. In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management (CIKM '17). Association for Computing Machinery, New York, NY, USA, 1449–1458. https://doi.org/10.1145/3132847.3132892

[2] Yukuo Cen, Jianwei Zhang, Xu Zou, Chang Zhou, Hongxia Yang, and Jie Tang. 2020. Controllable Multi-Interest Framework for Recommendation. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20). Association for Computing Machinery, New York, NY, USA, 2942–2951. https://doi.org/10.1145/3394486.3403344

[3] Yang Deng, Yaliang Li, Fei Sun, Bolin Ding, and Wai Lam. 2021. Unified Conversational Recommendation Policy Learning via Graph-based Reinforcement Learning. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '21). Association for Computing Machinery, New York, NY, USA, 1431–1441. https://doi.org/10.1145/3404835.3462913

[4] Yikun Xian, Zuohui Fu, S. Muthukrishnan, Gerard de Melo, and Yongfeng Zhang. 2019. Reinforcement Knowledge Graph Reasoning for Explainable Recommendation. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR'19). Association for Computing Machinery, New York, NY, USA, 285–294. https://doi.org/10.1145/3331184.3331203

[5] Jibril Frej, Neel Shah, Marta Knezevic, Tanya Nazaretsky, and Tanja Käser. 2024. Finding Paths for Explainable MOOC Recommendation: A Learner Perspective. In Proceedings of the 14th Learning Analytics and Knowledge Conference (LAK '24). Association for Computing Machinery, New York, NY, USA, 426–437. https://doi.org/10.1145/3636555.3636898

[6] Jibril Frej, Marta Knezevic, Tanja Kaser. "Graph Reasoning for Explainable Cold Start Recommendation." arXiv preprint arXiv:2406.07420, 2024.