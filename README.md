# AGRMCR - Adapting Graph Reasoning for Explainable Cold Start Recommendation on Multi-Round Conversation Recommendation

## Environment Setup 
<details>
<summary> 1. Requirements </summary>

```bash
pip install -r requirements.txt
```

</details>

<details>
<summary> 2. Docker Compose </summary>

For those who prefer containerization, Docker offers an isolated and consistent environment. Ensure Docker is installed on your system by following the [official Docker installation guide](https://docs.docker.com/get-docker/).

1. **Start the Application with Docker Compose:**
    ```bash
    docker compose up -d 
    ```
    If you've made changes and want them to reflect, append `--build` to the command above.
2. **Stopping the Application:**
   To stop and remove all running containers, execute:
   ```bash
   docker-compose down
   ```
</details>

## Data Preparation
Four Amazon datasets (Amazon_Beauty, Amazon_CDs, Amazon_Cellphones, Amazon_Clothing) are available in the "JRL/raw_data/" directory and the split is consistent with [1] and [2]. All four datasets used in this paper can be downloaded [here](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) which consist of metadata and 5-core review.

<details>

<summary> Statistics of dataset</summary>

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
| User $\xrightarrow{\text{dislike}}$ Brand | 192k    | 60k        | 90k       | 132k       |
| User $\xrightarrow{\text{interested in}}$ Category | 2.0M    | 949k       | 288k      | 354k       |
| Product $\xrightarrow{\text{described by}}$ Word          | 191M    | 17M        | 18M       | 18M        |
| Product $\xrightarrow{\text{belong to}}$ Category | 466k    | 154k       | 36k       | 49k        |
| Product $\xrightarrow{\text{produced by}}$ Brand | 64k     | 23k        | 10k       | 12k        |
| Product $\xrightarrow{\text{also bought}}$ Product        | 3.6M    | 1.4M       | 590k      | 891k       |
| Product $\xrightarrow{\text{also viewed}}$ Product        | 78k     | 147k       | 22k       | 155k       |
| Product $\xrightarrow{\text{bought together}}$ Product    | 78k     | 28k        | 12k       | 14k        |

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

## How to run the code
## JRL - Preprocessing dataset

1. Index datasets 
2. Split datasets for training and test 
3. Extract gzip to txt
4. Matching Relations brands, categories, related products
5. Matching Feature

```bash
source 01-JRL/preprocessing_data.sh
source 01-JRL/clone_to_pr.sh
```

<details>
<summary> Details code </summary>

```bash
```

</details>


<details>
<summary> Description </summary>

### STEP 1 : Index datasets (Entity) 
`index_and_filter_review_file.py `

This script processes the review data to generate various entity files.
#### Generated Files:
- `vocab.txt`       : Contains a list of unique words from the reviews.
- `user.txt`        : Contains a list of unique user IDs.
- `product.txt`     : Contains a list of unique product IDs.
- `review_text.txt` : Contains the text of the reviews.
- `review_u_p.txt`  : Maps reviews to users and products.
- `review_id.txt`   : Contains unique review IDs.
- `train.txt`       : 
- `test.txt`        :
- `validation.txt`  :

### STEP 2 : Split datasets for training and test 
`split_train_test.py`

### STEP 3 : Extract gzip to txt 
`gzip -d *.txt.gz`

### STEP 4 : Matching Relations
`match_cate_brand_related.py`

This script processes the data to generate relation files, which describe various relationships between entities such as products, brands, and categories.
#### Generated Files:
- `also_bought_p_p.txt`: Contains pairs of products that are often bought together.
- `also_view_p_p.txt`: Contains pairs of products that are often viewed together.
- `bought_together_p_p.txt`: Contains pairs of products that are frequently bought together.
- `brand_p_b.txt`: Maps products to their respective brands.
- `category_p_c.txt`: Maps products to their respective categories.
- `brand.txt`: Contains a list of unique brands.
- `category.txt`: Contains a list of unique categories.
- `related_product.txt` : Contains a list of unique related_product product IDs.

### STEP 5 : Clone preprocessed dataset to Path Reasoning 

</details>

## Transitional Embedding (TranSE)

1. Process original files
2. Dataset Split, Cold users/items, and Knowledge Graph Creation
3. Train the Knowledge Graph Embeddings

<details>
<summary> Details code </summary>

```bash
```

</details>

<details>
<summary> Description </summary>

</details>

## UNICORN - Multi-round Conversation Recommendation (MCR)
1. Training RL Agent
2. Evaluation RL Agent
3. Inference User Preference

```bash
source run_unicorn.sh
source inference_cold_start.sh
source clone_to_grec.sh
```

<details>

<summary>Details code</summary>

```bash
```

</details>

<details>
<summary> Description </summary>

### STEP 1 : Training RL Agent `RL_model.py`

This script will train RL policy network. Given $p_0$, the agent will decide which items to recommend.

### STEP 2 : Evaluation RL Agent`evaluate.py`

This script will evaluate RL policy network. Given $p_0$, the agent will decide which items to recommend

### STEP 3 : Inferencce User perference `evaluate.py`

This script will inference cold_start user to construct user perference

</details>

## GRECS - Graph Reasoning (GR)

1. Train the RL agent
2. Evaluation

```bash
source 02-GRECS/run_grec.sh
source 02-GRECS/clone_to_mcr.sh
```
<details> 
<summary>Details code</summary>
```bash
```
</details>

<details>
<summary>Description</summary>

### STEP 1 : Preprocessing `preprocess/domain.py`

This script processes the review data to generate various entity files.
#### Generated Files:
- `like_u_b.txt`            :
- `like_u_b_rating.txt`     :
- `dislike_u_b_rating.txt`  :
- `mentioned_by_u_w.txt`    :
- `described_as_p_w.txt`    : 
- `purchases.txt`           :
- `interested_in_u_c.txt`   :

### STEP 2 : Make dataset `make_dataset.py`

This script processes the purchase.txt to generate pair(user,item) of train/test/validation.txt
#### Generated Files:
- `cold_start_users.json`   :
- `cold_start_items.json`   :
- `train_dataset.pkl`       :
- `test_dataset.pkl`        :
- `valiation_dataset.pkl`   :
- `train_kg.pkl`            :
- `test_kg.pkl`             :
- `validation_kg.pkl`       :
- `train_label.pkl`         :
- `test_label.pkl`          :
- `validation_label.pkl`    :

### STEP 3 : Transitional Embedding (TransE) [3] `train_transe_model.py`
#### Generated Files:
- `train_transe_model/transe_model_sd_epoch_{}.ckpt` : original embedded
- `train_transe_model.pkl` : null/avg translation train embedded
- `test_transe_model.pkl` : null/avg translation test embedded
- `validation_transe_embed.pkl` : null/avg translation valid embedded

### STEP 4 : Clone transE embedding to Multi-round conversation

### STEP 5 : Train RL agent `train_agent.py`
#### Generated Files:

### STEP 6 : Evaluation RL agent `test_agent.py`
#### Generated Files:

</details>

<!-- ## Methodology

1. Construct New user preferred (NUP) in the form of graph.
2. Initializing NUP embeddings for Users/Items by translation
3. Extracting Existing User embeddings
4. Similarity
5. Generate Path Reasoning
6. Trim


<details>
<summary>Translation</summary>

**How can we best initialize the embedding of new user by utilizing other similar users?**

#### Average Translations
While the agent can navigate the Knowledge Graph (KG) from a cold user (or to a cold item) via their integration in the KG, it needs meaningful embeddings in its state representation to take an action that will lead to a relevant recommendation. To this end, [7] propose to calculate the embedding for a new entity by using the `average translations` from its related entities:

$$
\boldsymbol{e} = \sum_{(r', e'_t) \in \mathcal{G}_{e}} \left(\boldsymbol{e'_t} - \boldsymbol{r'}\right)/|\mathcal{G}_{e}|
$$

where $\mathcal{G}_{e}$ is the subset of all triplets in $\mathcal{G}$ whose head entity is $e$. This choice is motivated by the KG embeddings being trained using a translation method as described below:

$$
f(e_h, e_t | r) = <\boldsymbol{e_h} + \boldsymbol{r}, \boldsymbol{e_t}> + b_{e_t}
$$

where $\boldsymbol{e_h}, \boldsymbol{r}, \boldsymbol{e_t}$ are the embeddings of $e_h, r$ and $e_t$ respectively and $b_{e_t}$ is the bias of $e_t$.

#### Positive/Negative Translations
Given pairs $(r', e'_t)$ where $r$ could be actions like "purchase", "mention", "interested", "like", or negative actions like "don't like", "don't interested", and $e_t$ could be associated items, categories, or brands, it compute a weighted average of these pairs.

Let's denote the weight of each pair $(r', e'_t)$ as $w_{r', e'_t}$. If $w_{r', e'_t} = 1$ for `positive pairs` and $-1$ for `negative pairs`, the modified equation could be:

$$ \boldsymbol{e} = \frac{\sum_{(r', e'_t) \in \mathcal{G}_{e}} w_{r', e'_t} \cdot (\boldsymbol{e'_t} - \boldsymbol{r'})}{|\mathcal{G}_{e}|} $$
Where
- $ \mathcal{G}_{e}$ is still the set of pairs $(r, e_t)$.
- $ \boldsymbol{e_t} $ represents the vector associated with $e_t$.
- $ \boldsymbol{r} $ represents the vector associated with $r$.
- $ w_{r, e_t} $ is the weight assigned to each pair, where $ w_{r, e_t} = 1 $ for positive pairs like (purchase, item), (mention, item), etc.
- $ w_{r, e_t} = -1 $ for negative pairs like (disike, brand), (disinterested, category).

This modification allows you to adjust the contribution of each pair based on whether it is positive or negative, while still computing an average vector $\boldsymbol{e}$ that reflects the relationships captured by your pairs $(r', e'_t)$.

#### Null embeddings
To evaluate our cold embeddings assignment strategy, we will also compare it to using `null embeddings` (zero values everywhere) that correspond to no prior knowledge about users or items. In the following sections, we denote models using the average translation embeddings as `PGPR_a`/`UPGPR_a`, null embeddings as `PGPR_0`/`UPGPR_0`, negative embeddings as `PGPR_n`/`UPGPR_n`, and these methods regardless of the embeddings as `PGPR`/`UPGPR`.

</details>

<details>
<summary>Past history in the form of graph</summary>

**Does past history of other user preferences in the form of graph improve the success rate of recommendation ?**

### User Embedding

- `User Profile : new users embedding from MCR` : 
We construct a pair consisting of an entity and a relation based on the last state $s_t$ which consist of $[\mathcal{H}_u^{(t)},\mathcal{G}_u^{(t)}]$ where
  - $\mathcal{H}_u^{(t)} = [\mathcal{P}_u^{(t)}, \mathcal{P}_{\mathrm{rej}}^{(t)}, \mathcal{V}_{\mathrm{rej}}^{(t)}]$ denotes the conversation history until timestep $t$ 
  - $\mathcal{G}_u^{(t)}$ denotes the dynamic subgraph of $\mathcal{G}$ for the user $u$ at timestep $t$
  - $\mathcal{P}_u$ denotes the user-preferred attribute. 
  - $\mathcal{P}_{\mathrm{rej}}$ denotes the attributes rejected by the user 
  - $\mathcal{V}_{\mathrm{rej}}$ denotes the items rejected by the user
  
  We will get set of pair $(r', e'_t)$ which it would be $(r'_{pos}, p_u), (r'_{neg}, p_{rej}), (r'_{neg}, v_{rej})$ then we calculate new user embedding $e_{new}$ from `Positive/Negative Translations`

- `Existing users embeddings from TransE` : Take all users $ \textbf{E}_\textbf{U} $ which trained by `transE` 

- `Similarity function` : The goal of finding the highest matching candidate embedding $e_{\text{candidate}}$ involves calculating it using the formula: $$ e_{\text{candidate}} = \arg\max_{e_i \in \textbf{E}_\textbf{U}} f(e_{\text{new}}, \textbf{E}_\textbf{U}) $$ where
  - $ e_{\text{new}} $ denotes as a new embedding vector that you want to match against existing candidate embeddings.
  - $ \textbf{E}_\textbf{U} $ denotes as a set (or vector) of existing candidate user embeddings.
  - $ f(e_{\text{new}}, e_i) $ denotes as a function computes a similarity score or a measure of matching between the new user embedding $ e_{\text{new}} $ and each candidate user embedding $ e_i \in \textbf{E}_\textbf{U} $. Importantly, $ f(e_{\text{new}}, e_i) $ returns a value in the range $[0, 1]$, where higher values indicate a stronger match or similarity between $ e_{\text{new}} $ and $ e_i $.
  
  The expression $ \arg\max_{e_i \in \textbf{E}_\textbf{U}} f(e_{\text{new}}, e_i) $ finds the candidate embedding $ e_i $ from the set $ \textbf{E}_\textbf{U} $ that maximizes the matching function $ f $ with $ e_{\text{new}} $.

</details>

<details>
<summary>Graph Reasoning</summary>

- `Graph Reasoning (GR)`: Given $e_{\text{candidate}}$, the GR agent will generate paths for recommendation according to the trained policy.

</details>

<details>
<summary>Trim</summary>

- `Trim` : After obtaining GR of $e_{candidate}$, we eliminate the nodes of $\mathcal{P}_{\mathrm{rej}}$ and $\mathcal{V}_{\mathrm{rej}}$ 

</details> -->



## Run the baselines

**Overall, how does our technique compare to SOTA techniques?**

```
bash source 02-GREC/run_basline.sh
```

<details>
<summary>Details code</summary>

```bash
echo "------------- 1 : Process the files for Recbole -------------"
# Process the processed files for RecBole (after processing the original files for Graph Reasoning) 
echo "-------------- Formatting Beauty --------------------------"
python3 src/baselines/format_beauty.py \
    --config config_default/beauty/baselines/format.json 
echo "-------------- Formatting CDs --------------------------"
python3 src/baselines/format_cds.py \
    --config config_default/cds/baselines/format.json
echo "-------------- Formatting Cellphones -------------------"
python3 src/baselines/format_cellphones.py \
    --config config_default/cellphones/baselines/format.json
echo "-------------- Formatting Clothing ---------------------"
python3 src/baselines/format_clothing.py \
    --config config_default/clothing/baselines/format.json
echo "--------------------------------------------------------"
# python3 src/baselines/format_coco.py \
#     --config config_default/coco/baselines/format.json
# After this process, all the files from beauty have been standardized into the format needed by RecBole. 
# We follow the same process for the other datasets: 

echo "------------- 2 : Run the baselines -------------"
# To run a baseline on Beauty, choose a yaml config file in config_default/beauty/baselines and run the following:
DATASET_NAMES=("beauty" "cds" "cellphones" "clothing")

# DATASET_NAME=beauty
for DATASET_NAME in "${DATASET_NAMES[@]}"; do
    python3 src/baselines/baseline.py \
        --config config_default/${DATASET_NAME}/baselines/Pop.yaml
    python3 src/baselines/baseline.py \
        --config config_default/${DATASET_NAME}/baselines/ItemKNN.yaml
    python3 src/baselines/baseline.py \
        --config config_default/${DATASET_NAME}/baselines/BPR.yaml
    python3 src/baselines/baseline.py \
        --config config_default/${DATASET_NAME}/baselines/NeuMF.yaml
    python3 src/baselines/baseline.py \
        --config config_default/${DATASET_NAME}/baselines/CFKG.yaml
    python3 src/baselines/baseline.py \
        --config config_default/${DATASET_NAME}/baselines/KGCN.yaml
    python3 src/baselines/baseline.py \
        --config config_default/${DATASET_NAME}/baselines/MKR.yaml
    python3 src/baselines/baseline.py \
        --config config_default/${DATASET_NAME}/baselines/SpectralCF.yaml
done
# This example runs the Pop baseline on the Beauty dataset.
# You can ignore the warning "command line args [--config config_default/baselines/Pop.yaml] will not be used in RecBole". The argument is used properly.
```

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