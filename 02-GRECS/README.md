# Graph Reasoning for Explainable Cold Start Recommendation<!-- omit from toc -->

## Table of Contents<!-- omit from toc -->

- [Datasets](#datasets)
- [Installation](#installation)
- [How to run the Graph Reasoning models](#how-to-run-the-graph-reasoning-models)
- [Additional information about config files](#additional-information-about-config-files)
- [How to run the baselines](#how-to-run-the-baselines)

## Datasets

<details>

<summary>Datasets</summary>
### Amazon datasets

The Amazon datasets can be downloaded at this address: [https://drive.usercontent.google.com/download?id=1CL4Pjumj9d7fUDQb1_leIMOot73kVxKB](https://drive.usercontent.google.com/download?id=1CL4Pjumj9d7fUDQb1_leIMOot73kVxKB). \

Extract the files and place them in their associated folders.

You should get the following structure:

- data/beauty/Amazon_Beauty
- data/cds/Amazon_CDs
- data/cellphones/Amazon_Cellphones
- data/clothing/Amazon_Clothing

### COCO dataset

The COCO dataset is not available for direct download. To get access to it, you need to contact the authors of [COCO: Semantic-Enriched Collection of Online Courses at Scale with Experimental Use Cases](https://link.springer.com/chapter/10.1007/978-3-319-77712-2_133) by email. Extract the file and place it in data/coco/

You should get one folder:

- data/coco/coco/

Note: Because you might get a more recent version of the dataset, some of the characteristics (number of learners, courses, etc... ) might be different.

</details>

## Installation

<details>

<summary>Installation</summary>

### Requirements

Python 3.10

### Install required packages

```bash
pip install -r requirements.txt
```

If you intent to run the skill extractor on the coco dataset, you will need to download en_core_web_lg:

```bash
python -m spacy download en_core_web_lg
```

</details>

## How to run the Graph Reasoning models
<details>

<summary>Graph Reasoning</summary>

### Extract the skills from COCO's course descriptions (for COCO dataset only)

To extract the skills from COCO's course descriptions using SkillNER and connect each user to a skill taught by a course they have taken, run the following

```bash
python src/preprocess/extract_skills.py
```

After this process, the following files have been created in data/coco/coco:

- course_skill.csv
- learner_skill.csv


### Process original files

To Process Amazon's Beauty dataset use the command: 

```bash
python src/preprocess/beauty.py --config config/beauty/graph_reasoning/preprocess.json
```

After this process, all the files from been standardized into the format needed.

The files are saved in the folder data/beauty/Amazon_Beauty_01_01 (the path can be changed in the config file).

We used the same file format as in the original PGPR repository: [https://github.com/orcax/PGPR](https://github.com/orcax/PGPR).

To process the other datasets, use the commands:

```bash
python src/preprocess/cds.py --config config/cds/graph_reasoning/preprocess.json
python src/preprocess/cellphones.py --config config/cellphones/graph_reasoning/preprocess.json
python src/preprocess/clothing.py --config config/clothing/graph_reasoning/preprocess.json
python src/preprocess/coco.py --config config/coco/graph_reasoning/preprocess.json
```

### Dataset Split, Cold users/items, and Knowledge Graph Creation

To split the dataset, get the cold users and items and Create the Knowledge Graph for the Beauty dataset:

```bash
python src/graph_reasoning/make_dataset.py --config config/beauty/graph_reasoning/UPGPR.json
```

The additional files are saved in the folder data/beauty/Amazon_Beauty_01_01 (the path can be changed in the config file).

For the other datasets, you just need to change the config file:

```bash
python src/graph_reasoning/make_dataset.py --config config/cds/graph_reasoning/UPGPR.json
python src/graph_reasoning/make_dataset.py --config config/cellphones/graph_reasoning/UPGPR.json
python src/graph_reasoning/make_dataset.py --config config/clothing/graph_reasoning/UPGPR.json
python src/graph_reasoning/make_dataset.py --config config/coco/graph_reasoning/UPGPR.json
```

### Train the Knowledge Graph Embeddings

To train the Knowledge Graph Embeddings use the command:

```bash
python src/graph_reasoning/train_transe_model.py --config config/beauty/graph_reasoning/UPGPR.json
```

The KG embeddings are saved in data/beauty/Amazon_Beauty_01_01/train_transe_model

For the other datasets, you just need to change the config file:

```bash
python src/graph_reasoning/train_transe_model.py --config config/cds/graph_reasoning/UPGPR.json
python src/graph_reasoning/train_transe_model.py --config config/cellphones/graph_reasoning/UPGPR.json
python src/graph_reasoning/train_transe_model.py --config config/clothing/graph_reasoning/UPGPR.json
python src/graph_reasoning/train_transe_model.py --config config/coco/graph_reasoning/UPGPR.json
```

### Train the RL agent

After the embeddings are trained, optimize the RL agent:

```bash
python src/graph_reasoning/train_agent.py --config config/beauty/graph_reasoning/UPGPR.json
```

The agent is saved in data/beauty/Amazon_Beauty_01_01/agent

For the other datasets, you just need to change the config file:


```bash
python src/graph_reasoning/train_agent.py --config config/cds/graph_reasoning/UPGPR.json
python src/graph_reasoning/train_agent.py --config config/cellphones/graph_reasoning/UPGPR.json
python src/graph_reasoning/train_agent.py --config config/clothing/graph_reasoning/UPGPR.json
python src/graph_reasoning/train_agent.py --config config/coco/graph_reasoning/UPGPR.json
```

### Evaluation
To compute and save the recommendations and metrics:

```bash
python src/graph_reasoning/test_agent.py --config config/beauty/graph_reasoning/UPGPR.json
```

The results are saved in the folder results.

For the other datasets, you just need to change the config file:

```bash
python src/graph_reasoning/test_agent.py --config config/cds/graph_reasoning/UPGPR.json
python src/graph_reasoning/test_agent.py --config config/cellphones/graph_reasoning/UPGPR.json
python src/graph_reasoning/test_agent.py --config config/clothing/graph_reasoning/UPGPR.json
python src/graph_reasoning/test_agent.py --config config/coco/graph_reasoning/UPGPR.json
```

### Config files

In the examples above, we use the config file config/beauty/graph_reasoning/UPGPR.json for running UPGPR on the beauty dataset.

If you want to run another model on beauty use one of the listed config file:

- config/beauty/graph_reasoning/UPGPR.json: UPGPR with the average strategy for cold embeddings assignment   
- config/beauty/graph_reasoning/PGPR.json: PGPR with the average strategy for cold embeddings assignment   
- config/beauty/graph_reasoning/UPGPR_zero.json: UPGPR with the null strategy for cold embeddings assignment   
- config/beauty/graph_reasoning/PGPR_zero.json: PGPR with the null strategy for cold embeddings assignment   


</details>


## How to run the baselines

<details>

<summary>Baselines</summary>


### Process the files for Recbole

Process the processed files for RecBole (after processing the original files for Graph Reasoning )

```bash
python src/baselines/format_beauty.py --config config/beauty/baselines/format.json
```

After this process, all the files from beauty have been standardized into the format needed by RecBole. 

We follow the same process for the other datasets:

```bash
python src/baselines/format_cds.py --config config/cds/baselines/format.json
python src/baselines/format_cellphones.py --config config/cellphones/baselines/format.json
python src/baselines/format_clothing.py --config config/clothing/baselines/format.json
python src/baselines/format_coco.py --config config/coco/baselines/format.json
```

### Run the baselines

To run a baseline on Beauty, choose a yaml config file in config/beauty/baselines and run the following:

```bash
python src/baselines/baseline.py --config config/baselines/Pop.yaml
```

This example runs the Pop baseline on the Beauty dataset.

You can ignore the warning "command line args [--config config/baselines/Pop.yaml] will not be used in RecBole". The argument is used properly.

</details>
