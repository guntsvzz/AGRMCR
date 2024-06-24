import os
import json
import argparse

from easydict import EasyDict as edict


def save_users(datadir, savedir, name):
    """Save users to recbole format

    Args:
        datadir (str): path of the processed dataset
        savedir (str): path of the recbole dataset
        name (str): name of the dataset
    """
    users = []
    with open(os.path.join(datadir, "users.txt"), "r") as f:
        for line in f:
            users.append(line.strip())

    with open(os.path.join(savedir, name + ".user"), "w") as f:
        f.write("user_id:token\n")
        for user_id in users:
            f.write(f"{user_id}\n")
    return users


def save_items(datadir, savedir, name):
    """Save items to recbole format

    Args:
        datadir (str): path of the processed dataset
        savedir (str): path of the recbole dataset
        name (str): name of the dataset
    """

    items = []
    with open(os.path.join(datadir, "product.txt"), "r") as f:
        for line in f:
            items.append(line.strip())

    with open(os.path.join(savedir, name + ".item"), "w") as f:
        f.write("item_id:token\n")
        for item_id in items:
            f.write(f"{item_id}\n")
    return items


def save_interactions(datadir, savedir, name, subset, entities, interacted_items):
    """Save interactions recbole format

    Args:
        datadir (str): path of the processed dataset
        savedir (str): path of the recbole dataset
        name (str): name of the dataset
        subset (str): name of the subset
        entities (dict): dictionary of entities, must contain "users" and "items"
        interacted_items (set): set of items that have been interacted with
    """
    interactions = []
    users = entities["user"]
    items = entities["item"]
    with open(os.path.join(datadir, subset + ".txt"), "r") as f:
        for line in f:
            interactions.append([int(x) for x in line.split()])

    with open(os.path.join(savedir, name + "." + subset + ".inter"), "w") as f:
        f.write("user_id:token\titem_id:token\trating:float\n")
        for user, item in interactions:
            interacted_items.add(items[item])
            f.write(f"{users[user]}\t{items[item]}\t1\n")


def save_item_entity(savedir, name, items, interacted_items):
    """Save item entities to recbole format

    Args:
        savedir (str): path of the recbole dataset
        name (str): name of the dataset
        items (list): list of items
        interacted_items (set): set of items that have been interacted with
    """
    with open(os.path.join(savedir, name + ".link"), "w") as f:
        f.write("item_id:token\tentity_id:token\n")
        for item in items:
            if item in interacted_items:
                f.write(f"{item}\t{item}\n")


def read_entities(datadir, entity_names, entities):
    """Read entities from processed dataset

    Args:
        datadir (str): path of the processed dataset
        entity_names (list): list of entity names
        entities (dict): dictionary of entities

    """
    for entity_name in entity_names:
        entities[entity_name] = []
        with open(os.path.join(datadir, entity_name + ".txt"), "r") as f:
            for line in f:
                entities[entity_name].append(line.strip())


def read_triplets(
    datadir,
    kg_triplets,
    relation_file,
    entities,
    head_entities,
    relation,
    tail_entities,
    interacted_items,
):
    """Read triplets from processed dataset

    Args:
        datadir (str): path of the processed dataset
        kg_triplets (list): list of kg triplets
        relation_file (str): name of the relation file
        entities (dict): dictionary of entities
        head_entities (str): head entity name
        relation (str): relation
        tail_entities (str): tail entity name
        interacted_items (set): set of items that have been interacted with
    """
    with open(os.path.join(datadir, relation_file), "r") as f:
        is_item = head_entities == "item"
        head_entities = entities[head_entities]
        tail_entities = entities[tail_entities]
        for i, line in enumerate(f):
            list_tail_entities = line.strip()
            if tail_entities:
                head_entity = head_entities[int(i)]
                if is_item and head_entity in interacted_items:
                    for tail_entity in list_tail_entities.split():
                        kg_triplets.append(
                            [
                                head_entity,
                                relation,
                                tail_entities[int(tail_entity)],
                            ]
                        )


def read_all_triplets(
    datadir, kg_triplets, entities, relations_files, interacted_items
):
    """Update kg triplets with all relations

    Args:
        datadir (str): path of the processed dataset
        kg_triplets (list): list of kg triplets
        entities (dict): dictionary of entities
        relations_files (dict): dictionary of relation files and their corresponding triplets
    """
    for relation_file, (head_entity, relation, tail_entity) in relations_files.items():
        read_triplets(
            datadir,
            kg_triplets,
            relation_file,
            entities,
            head_entity,
            relation,
            tail_entity,
            interacted_items,
        )


def save_kg_triplets(kg_triplets, savedir, name):
    """Save kg_triplets to file.

    Args:
        kg_triplets (list): list of triplets as a tuple (head_id, relation_id, tail_id)
        savedir (str): path of the recbole dataset
        name (str): name of the dataset
    """
    with open(os.path.join(savedir, name + ".kg"), "w") as f:
        f.write("head_id:token\trelation_id:token\ttail_id:token\n")
        for head_id, relation_id, tail_id in kg_triplets:
            f.write(f"{head_id}\t{relation_id}\t{tail_id}\n")


def format_pgpr_moocube(datadir, savedir, name):
    """Format processed dataset to recbole format

    Args:
        datadir (str): path of the processed dataset
        savedir (str): path of the recbole dataset
        name (str): name of the dataset
    """
    entities = dict()
    entities["user"] = save_users(datadir, savedir, name)
    entities["item"] = save_items(datadir, savedir, name)

    subsets = ["train", "validation", "test"]

    # keep track of the items that have been interacted with, recbole does not support items that have not been interacted with but that are in the kg file
    interacted_items = set()

    for subset in subsets:
        save_interactions(datadir, savedir, name, subset, entities, interacted_items)

    save_item_entity(savedir, name, entities["item"], interacted_items)

    entity_names = ["category", "brand", "vocab", "related_product"]

    read_entities(datadir, entity_names, entities)

    kg_triplets = []
    relations_files = {
        "also_bought_p_p.txt": ("item", "also_bought", "related_product"),
        "also_viewed_p_p.txt": ("item", "also_viewed", "related_product"),
        "bought_together_p_p.txt": ("item", "bought_together", "related_product"),
        "described_as_p_w.txt": ("item", "described_", "vocab"),
        "brand_p_b.txt": ("item", "belong_to", "brand"),
        "category_p_c.txt": ("item", "category_of", "category"),
        "mentioned_by_u_w.txt": ("user", "mentioned", "vocab"),
        "interested_in_u_c.txt": ("user", "interested_in", "category"),
        "like_u_b.txt": ("user", "like", "brand"),
    }

    read_all_triplets(datadir, kg_triplets, entities, relations_files, interacted_items)

    save_kg_triplets(kg_triplets, savedir, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Config file.",
        default="config/clothing/baselines/format.json",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = edict(json.load(f))

    # Creates the folder savedir if it does not exist
    os.makedirs(config.savedir, exist_ok=True)

    format_pgpr_moocube(config.datadir, config.savedir, config.dataset_name)
