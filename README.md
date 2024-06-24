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

## JRL - Preprocessing dataset

<details>
<summary>Preprocessing dataset </summary>

```bash
python3 index_and_filter_review_file.py
python3 match_cate_brand_related.py
```
### 1. `index_and_filter_review_file.py `

This script processes the review data to generate various entity files.
#### Generated Files:
- `vocab.txt`: Contains a list of unique words from the reviews.
- `user.txt`: Contains a list of unique user IDs.
- `product.txt`: Contains a list of unique product IDs.
- `review_text.txt`: Contains the text of the reviews.
- `review_u_p.txt`: Maps reviews to users and products.
- `review_id.txt`: Contains unique review IDs.

### 2. `match_cate_brand_related.py`

This script processes the data to generate relation files, which describe various relationships between entities such as products, brands, and categories.
#### Generated Files:
- `also_bought_p_p.txt`: Contains pairs of products that are often bought together.
- `also_view_p_p.txt`: Contains pairs of products that are often viewed together.
- `bought_together_p_p.txt`: Contains pairs of products that are frequently bought together.
- `brand_p_b.txt`: Maps products to their respective brands.
- `category_p_c.txt`: Maps products to their respective categories.
- `brand.txt`: Contains a list of unique brands.
- `category.txt`: Contains a list of unique categories.
- `related_product` : Contains a list of unique related_product product IDs.

</details>

## GRECS - Path Reasoning

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

### 1. `preprocess/domain.py`

This script processes the review data to generate various entity files.
#### Generated Files:
- `mentioned_by_u_w.txt`    :
- `described_as_p_w.txt`    : 
- `purchases.txt`           :
- `interested_in_u_c.txt`   :

### 2. `make_dataset.py`

This script processes the purchase.txt to generate pair(user,item) of train/test/validation.txt
#### Generated Files:
- `train.txt`               : 
- `test.txt`                :
- `validation.txt`          :
- `train_dataset.pkl`       :
- `test_dataset.pkl`        :
- `valiation_dataset.pkl`   :
- `kg.pkl`                  :

</details>

<details>

<summary>Train TransE</summary>

### Transitaitonal Embedding (TransE) [3]
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

## UNICORN - Multi-round Conversation

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
<summary>Initialize Embedding</summary>

**How can we best initialize the embedding of new user by utilizing other similar users?**

### Cold Embeddings for Users/Items

#### Average Translations
While the agent can navigate the Knowledge Graph (KG) from a cold user (or to a cold item) via their integration in the KG, it needs meaningful embeddings in its state representation to take an action that will lead to a relevant recommendation. To this end, we propose to calculate the embedding for a new entity by using the `average translations` from its related entities:

$$
\boldsymbol{e} = \sum_{(r', e'_t) \in \mathcal{G}_{e}} \left(\boldsymbol{e'_t} - \boldsymbol{r'}\right)/|\mathcal{G}_{e}|
$$

where $\mathcal{G}_{e}$ is the subset of all triplets in $\mathcal{G}$ whose head entity is $e$. This choice is motivated by the KG embeddings being trained using a translation method as described below:

$$
f(e_h, e_t | r) = <\boldsymbol{e_h} + \boldsymbol{r}, \boldsymbol{e_t}> + b_{e_t}
$$

where $\boldsymbol{e_h}, \boldsymbol{r}, \boldsymbol{e_t}$ are the embeddings of $e_h, r$ and $e_t$ respectively and $b_{e_t}$ is the bias of $e_t$.

#### Pos-Neg-Translations
Given pairs $(r', e'_t)$ where $r$ could be actions like "purchase", "mention", "interested", "like", or negative actions like "don't like", "don't interested", and $e_t$ could be associated items, categories, or brands, it compute a weighted average of these pairs.

Let's denote the weight of each pair $(r', e'_t)$ as $w_{r', e'_t}$. If $w_{r', e'_t} = 1$ for `positive pairs` and $-1$ for `negative pairs`, the modified equation could be:

$$ \boldsymbol{e} = \frac{\sum_{(r', e'_t) \in \mathcal{G}_{e}} w_{r', e'_t} \cdot (\boldsymbol{e'_t} - \boldsymbol{r'})}{|\mathcal{G}_{e}|} $$

Where

- $ \mathcal{G}_{e}$ is still the set of pairs \((r, e_t)\).
- $ \boldsymbol{e_t} $ represents the vector associated with $e_t$.
- $ \boldsymbol{r} $ represents the vector associated with \(r\).
- $ w_{r, e_t} $ is the weight assigned to each pair, where $ w_{r, e_t} = 1 $ for positive pairs like (purchase, item), (mention, item), etc.
- $ w_{r, e_t} = -1 $ for negative pairs like (disike, brand), (disinterested, category).

This modification allows you to adjust the contribution of each pair based on whether it is positive or negative, while still computing an average vector $\boldsymbol{e}$ that reflects the relationships captured by your pairs $(r', e'_t)$.

#### Null embeddings
To evaluate our cold embeddings assignment strategy, we will also compare it to using `null embeddings` (zero values everywhere) that correspond to no prior knowledge about users or items. In the following sections, we denote models using the average translation embeddings as `PGPR_a`/`UPGPR_a`, null embeddings as `PGPR_0`/`UPGPR_0`, negative embeddings as `PGPR_n`/`UPGPR_n`, and these methods regardless of the embeddings as `PGPR`/`UPGPR`.

</details>

<details>
<summary>Past history in the form of graph</summary>

**Does past history of other user preferences in the form of graph improve the success rate of recommendation ?**

### User-similarity

- `Graph from MCR` : calculate new user embedding $e_{new}$ from last state which consist of $s_t = [\mathcal{H}_u^{(t)},\mathcal{G}_u^{(t)}]$ 
  - $\mathcal{H}_u^{(t)} = [\mathcal{P}_u^{(t)}, \mathcal{P}_{\mathrm{rej}}^{(t)}, \mathcal{V}_{\mathrm{rej}}^{(t)}]$ denotes the conversation history until timestep $t$ 
  - $\mathcal{G}_u^{(t)}$ denotes the dynamic subgraph of $\mathcal{G}$ for the user $u$ at timestep $t$
  - $\mathcal{P}_u$ denotes the user-preferred attribute. 
  - $\mathcal{P}_{\mathrm{rej}}$ is the attributes rejected by the user 
  - $\mathcal{V}_{\mathrm{rej}}$ are the attributes rejected by the user

- `Graph Past history of existing user` : calculate all users $ \textbf{e}_\textbf{U} $

- Similarity function : $ argmax(f(e_{new}, \textbf{e}_\textbf{U}))$ where $f(e_{\text{new}}, \textbf{e}_\textbf{U}) \in [0, 1] $

- `Generating Graph Reasoning (GR)`: 

- `Trim` : After obtaining GR of $e_u$, we eliminate the nodes which are $\mathcal{P}_{\mathrm{rej}}$ and $\mathcal{V}_{\mathrm{rej}}$ 
</details>

<details>
<summary>Comparing SOTA techniques</summary>

**Overall, how does our technique compare to SOTA techniques?**

### Run the baselines

To run a baseline on Beauty, choose a yaml config file in config/beauty/baselines and run the following:

```bash
python src/baselines/baseline.py --config config/baselines/Pop.yaml
```

This example runs the Pop baseline on the Beauty dataset.

You can ignore the warning "command line args [--config config/baselines/Pop.yaml] will not be used in RecBole". The argument is used properly.

</details>

## Citation
Todsavad Tangtortan, Pranisaa Charnparttaravanit, Akraradet Sinsamersuk, Chaklam Silpasuwanchai. 2024. Adapting Graph Reasoning for Explainable Cold Start Recommendation on Multi-Round Conversation Recommendation (AGRMCR). 

## References
[1] Yongfeng Zhang, Qingyao Ai, Xu Chen, and W. Bruce Croft. 2017. Joint Representation Learning for Top-N Recommendation with Heterogeneous Information Sources. In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management (CIKM '17). Association for Computing Machinery, New York, NY, USA, 1449–1458. https://doi.org/10.1145/3132847.3132892

[2] Yukuo Cen, Jianwei Zhang, Xu Zou, Chang Zhou, Hongxia Yang, and Jie Tang. 2020. Controllable Multi-Interest Framework for Recommendation. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20). Association for Computing Machinery, New York, NY, USA, 2942–2951. https://doi.org/10.1145/3394486.3403344

[3] Antoine Bordes, Nicolas Usunier, Alberto Garcia-Durán, Jason Weston, and Oksana Yakhnenko. 2013. Translating embeddings for modeling multi-relational data. In Proceedings of the 26th International Conference on Neural Information Processing Systems - Volume 2 (NIPS'13). Curran Associates Inc., Red Hook, NY, USA, 2787–2795.

[4] Yang Deng, Yaliang Li, Fei Sun, Bolin Ding, and Wai Lam. 2021. Unified Conversational Recommendation Policy Learning via Graph-based Reinforcement Learning. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '21). Association for Computing Machinery, New York, NY, USA, 1431–1441. https://doi.org/10.1145/3404835.3462913

[5] Yikun Xian, Zuohui Fu, S. Muthukrishnan, Gerard de Melo, and Yongfeng Zhang. 2019. Reinforcement Knowledge Graph Reasoning for Explainable Recommendation. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR'19). Association for Computing Machinery, New York, NY, USA, 285–294. https://doi.org/10.1145/3331184.3331203

[6] Jibril Frej, Neel Shah, Marta Knezevic, Tanya Nazaretsky, and Tanja Käser. 2024. Finding Paths for Explainable MOOC Recommendation: A Learner Perspective. In Proceedings of the 14th Learning Analytics and Knowledge Conference (LAK '24). Association for Computing Machinery, New York, NY, USA, 426–437. https://doi.org/10.1145/3636555.3636898

[7] Jibril Frej, Marta Knezevic, Tanja Kaser. "Graph Reasoning for Explainable Cold Start Recommendation." arXiv preprint arXiv:2406.07420, 2024.