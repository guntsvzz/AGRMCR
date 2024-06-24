# AGRMCR - Adapting Graph Reasoning for Explainable Cold Start Recommendation on Multi-Round Conversation Recommendation

## Requirement 
```bash
pip install -r requirements.txt
```

## Data Preparation

<details>

<summary>Datasets</summary>

Four Amazon datasets (Amazon_Beauty, Amazon_CDs, Amazon_Cellphones, Amazon_Clothing) are available in the "JRL/raw_data/" directory and the split is consistent with [1] and [2].

All four datasets used in this paper can be downloaded below
- [Metadata & Review-5-core v2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)
- [Metadata - Amazon dataset v2018](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)
- [Review-5-core - Amazon dataset v2018](https://nijianmo.github.io/amazon/index.html)

### Summary statistics of datasets.

### Entity Statistics for E-commerce Datasets

|                | **CDs** | **Cloth.** | **Cell.** | **Beauty** |
|----------------|---------|------------|-----------|------------|
| **#Entities**  |         |            |           |            |
| User           | 75k     | 39k        | 27k       | 22k        |
| Product        | 64k     | 23k        | 10k       | 12k        |
| Word           | 202k    | 21k        | 22k       | 22k        |
| Brand          | 1.4k    | 1.1k       | 955       | 2k         |
| Category       | 770     | 1.1k       | 206       | 248        |

### Relation Statistics for E-commerce Datasets

|                                      | **CDs** | **Cloth.** | **Cell.** | **Beauty** |
|--------------------------------------|---------|------------|-----------|------------|
| **#Relations**                       |         |            |           |            |
| User $\xrightarrow{\text{purchase}}$ Product               | 1.1M    | 278k       | 194k      | 198k       |
| User $\xrightarrow{\text{mention}}$ Word                   | 191M    | 17M        | 18M       | 18M        |
| User $\xrightarrow{\text{like}}$ Brand | 192k    | 60k        | 90k       | 132k       |
| User $\xrightarrow{\text{interested\_in}}$ Category | 2.0M    | 949k       | 288k      | 354k       |
| Product $\xrightarrow{\text{described\_by}}$ Word          | 191M    | 17M        | 18M       | 18M        |
| Product $\xrightarrow{\text{belong\_to}}$ Category | 466k    | 154k       | 36k       | 49k        |
| Product $\xrightarrow{\text{produced\_by}}$ Brand | 64k     | 23k        | 10k       | 12k        |
| Product $\xrightarrow{\text{also\_bought}}$ Product        | 3.6M    | 1.4M       | 590k      | 891k       |
| Product $\xrightarrow{\text{also\_viewed}}$ Product        | 78k     | 147k       | 22k       | 155k       |
| Product $\xrightarrow{\text{bought\_together}}$ Product    | 78k     | 28k        | 12k       | 14k        |

### Entities and Relations 
| Head | Relation           | Tail                 |
|------|--------------------|----------------------|
| USER | INTERACT           | ITEM                 |
| USER | MENTION            | WORD                 |
| USER | LIKE**             | BRAND                |
| USER | INTERESTED_IN**    | CATEGORY             |
| ITEM | DESCRIBED_BY       | WORD                 |
| ITEM | BELONG_TO**        | CATEGORY (FEATURE)   |
| ITEM | PRODUCED_BY**      | BRAND (FEATURE)      |
| ITEM | ALSO_BUY           | ITEM                 |
| ITEM | ALSO_VIEW          | ITEM                 |
| ITEM | BOUGHT_TOGETHER    | ITEM                 |

** denoted it used to integrate cold users or cold items into the KG.

</details>

<details>

<summary> Graph Formating </summary>

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
python3 RL_model.py \
    --data_name AMAZON --data_name BEAUTY --domain beauty --max_steps 100 --sample_times 100 --embed transe
python3 RL_model.py \
    --data_name AMAZON --data_name CDS --domain cds --max_steps 100 --sample_times 100 --embed transe
python3 RL_model.py \
    --data_name AMAZON --data_name CELLPHONES --domain cellphones --max_steps 100 --sample_times 100 --embed transe
python3 RL_model.py \
    --data_name AMAZON --data_name CLOTHING --domain clothing --max_steps 100 --sample_times 100 --embed transe
```

### Evaluation
```bash
python3 evaluate.py \
    --data_name AMAZON --data_name BEAUTY --domain beauty --load_rl_epoch 10 --embed transe
python3 evaluate.py \
    --data_name AMAZON --data_name CDS --domain cds --load_rl_epoch 10 --embed transe
python3 evaluate.py \
    --data_name AMAZON --data_name CELLPHONES --domain cellphones --load_rl_epoch 10 --embed transe
python3 evaluate.py     \
    --data_name AMAZON --data_name CLOTHING --domain clothing --load_rl_epoch 10 --embed transe
```

</details>


## Ablation Study
<details>

1. Does past history of other user preferences in the form of graph improve the success rate of recommendation ?

### User-similarity

2. How can we best initialize the embedding of new user by utilizing other similar users?
### Cold Embeddings for User
While the agent can navigate the Knowledge Graph (KG) from a cold user (or to a cold item) via their integration in the KG, it needs meaningful embeddings in its state representation to take an action that will lead to a relevant recommendation. To this end, we propose to calculate the embedding for a new entity by using the average translations from its related entities:

$$
\boldsymbol{e} = \sum_{(r', e'_t) \in \mathcal{G}_{e}} \left(\boldsymbol{e'_t} - \boldsymbol{r'}\right)/|\mathcal{G}_{e}|
$$

where $\mathcal{G}_{e}$ is the subset of all triplets in $\mathcal{G}$ whose head entity is $e$. This choice is motivated by the KG embeddings being trained using a translation method as described below:

$$
f(e_h, e_t | r) = <\boldsymbol{e_h} + \boldsymbol{r}, \boldsymbol{e_t}> + b_{e_t}
$$

where $\boldsymbol{e_h}, \boldsymbol{r}, \boldsymbol{e_t}$ are the embeddings of $e_h, r$ and $e_t$ respectively and $b_{e_t}$ is the bias of $e_t$.

To evaluate our cold embeddings assignment strategy, we will also compare it to using null embeddings (zero values everywhere) that correspond to no prior knowledge about users or items. In the following sections, we denote models using the average translation embeddings as `PGPR_a`/`UPGPR_a`, null embeddings as `PGPR_0`/`UPGPR_0`, and both methods regardless of the embeddings as `PGPR`/`UPGPR`.



3. Overall, how does our technique compare to SOTA techniques?
### Run the baselines

To run a baseline on Beauty, choose a yaml config file in config/beauty/baselines and run the following:

```bash
python src/baselines/baseline.py --config config/baselines/Pop.yaml
```

This example runs the Pop baseline on the Beauty dataset.

You can ignore the warning "command line args [--config config/baselines/Pop.yaml] will not be used in RecBole". The argument is used properly.

</details>

## Citation
Todsavad Tangtortan. 2024. Adapting Graph Reasoning for Explainable Cold Start Recommendation on Multi-Round Conversation Recommendation (AGRMCR). AIT, Thailand.

## References
[1] Yongfeng Zhang, Qingyao Ai, Xu Chen, and W. Bruce Croft. 2017. Joint Representation Learning for Top-N Recommendation with Heterogeneous Information Sources. In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management (CIKM '17). Association for Computing Machinery, New York, NY, USA, 1449–1458. https://doi.org/10.1145/3132847.3132892

[2] Yukuo Cen, Jianwei Zhang, Xu Zou, Chang Zhou, Hongxia Yang, and Jie Tang. 2020. Controllable Multi-Interest Framework for Recommendation. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20). Association for Computing Machinery, New York, NY, USA, 2942–2951. https://doi.org/10.1145/3394486.3403344

[3] Yang Deng, Yaliang Li, Fei Sun, Bolin Ding, and Wai Lam. 2021. Unified Conversational Recommendation Policy Learning via Graph-based Reinforcement Learning. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '21). Association for Computing Machinery, New York, NY, USA, 1431–1441. https://doi.org/10.1145/3404835.3462913

[4] Yikun Xian, Zuohui Fu, S. Muthukrishnan, Gerard de Melo, and Yongfeng Zhang. 2019. Reinforcement Knowledge Graph Reasoning for Explainable Recommendation. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR'19). Association for Computing Machinery, New York, NY, USA, 285–294. https://doi.org/10.1145/3331184.3331203

[5] Jibril Frej, Neel Shah, Marta Knezevic, Tanya Nazaretsky, and Tanja Käser. 2024. Finding Paths for Explainable MOOC Recommendation: A Learner Perspective. In Proceedings of the 14th Learning Analytics and Knowledge Conference (LAK '24). Association for Computing Machinery, New York, NY, USA, 426–437. https://doi.org/10.1145/3636555.3636898

[6] Jibril Frej, Marta Knezevic, Tanja Kaser. "Graph Reasoning for Explainable Cold Start Recommendation." arXiv preprint arXiv:2406.07420, 2024.