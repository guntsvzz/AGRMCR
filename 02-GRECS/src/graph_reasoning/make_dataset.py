from __future__ import absolute_import, division, print_function

import argparse
import random
import wandb
import random
import json

from utils import *
from data_utils import Dataset
# from knowledge_graph_m import KnowledgeGraph
from knowledge_graph_m import KnowledgeGraph
from easydict import EasyDict as edict


def generate_labels(data_dir, filename):
    interaction_file = f"{data_dir}/{filename}"
    user_items = {}  # {uid: [cid,...], ...}
    with open(interaction_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            arr = line.split(" ")
            user_idx = int(arr[0])
            item_idx = int(arr[1])
            if user_idx not in user_items:
                user_items[user_idx] = []
            user_items[user_idx].append(item_idx)
    return user_items


def get_cold_users(user_items, cold_users_prop):
    """Make the cold start users sets."""
    cold_start_users = dict()
    users = list(user_items.keys())
    nb_users = len(users)
    random.shuffle(users)
    cold_start_users["train"] = set(users[: int((1 - 2 * cold_users_prop) * nb_users)])
    cold_start_users["validation"] = set(
        users[
            int((1 - 2 * cold_users_prop) * nb_users) : int(
                (1 - cold_users_prop) * nb_users
            )
        ]
    )
    cold_start_users["test"] = set(users[int((1 - cold_users_prop) * nb_users) :])
    return cold_start_users


def get_cold_items(user_items, cold_items_prop):
    cold_start_items = dict()
    items = set()
    for user in user_items:
        items.update(user_items[user])
    items = list(items)
    nb_items = len(items)
    random.shuffle(items)

    cold_start_items["train"] = set(items[: int((1 - 2 * cold_items_prop) * nb_items)])
    cold_start_items["validation"] = set(
        items[
            int((1 - 2 * cold_items_prop) * nb_items) : int(
                (1 - cold_items_prop) * nb_items
            )
        ]
    )
    cold_start_items["test"] = set(items[int((1 - cold_items_prop) * nb_items) :])
    return cold_start_items


def split_train_test_data_by_user(
    data_dir,
    data_file,
    validation_prop,
    test_prop,
    cold_users_prop,
    cold_items_prop,
):
    user_items = generate_labels(data_dir, data_file)

    train_data = []
    test_data = []
    validation_data = []

    # Make the cold start users sets
    cold_start_users = get_cold_users(user_items, cold_users_prop)
    # Make the cold start items sets
    cold_start_items = get_cold_items(user_items, cold_items_prop)

    # Split the data into train, validation and test sets
    for user in user_items:
        items = user_items[user]
        nb_users_validation = max(1, int(len(items) * validation_prop))
        nb_users_test = max(1, int(len(items) * test_prop))
        # get the list of items for the user for each set
        l_train_data = items[: -(nb_users_validation + nb_users_test)]
        l_validation_data = items[
            -(nb_users_validation + nb_users_test) : -(nb_users_test)
        ]
        l_test_data = items[-(nb_users_test):]

        # remove cold start items from each set
        l_train_data = [
            item for item in l_train_data if item in cold_start_items["train"]
        ]

        l_validation_data = [
            item for item in l_validation_data if item not in cold_start_items["test"]
        ]

        l_test_data = [
            item for item in l_test_data if item not in cold_start_items["validation"]
        ]

        # remove cold start users from each set
        if user in cold_start_users["train"]:
            for c in l_train_data:
                train_data.append(f"{user} {c}\n")
            for c in l_validation_data:
                validation_data.append(f"{user} {c}\n")
            for c in l_test_data:
                test_data.append(f"{user} {c}\n")
        elif user in cold_start_users["validation"]:
            for c in l_validation_data:
                validation_data.append(f"{user} {c}\n")
        else:
            for c in l_test_data:
                test_data.append(f"{user} {c}\n")

    random.shuffle(train_data)
    random.shuffle(test_data)
    random.shuffle(validation_data)

    create_data_file(data_dir, train_data, "train.txt")
    create_data_file(data_dir, validation_data, "validation.txt")
    create_data_file(data_dir, test_data, "test.txt")

    for key in cold_start_users:
        cold_start_users[key] = list(cold_start_users[key])

    for key in cold_start_items:
        cold_start_items[key] = list(cold_start_items[key])

    with open(data_dir + "/cold_start_users.json", "w") as f:
        json.dump(cold_start_users, f, indent=4)

    with open(data_dir + "/cold_start_items.json", "w") as f:
        json.dump(cold_start_items, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Config file.",
        default="config/coco_01_01/UPGPR_10.json",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = edict(json.load(f))

    set_random_seed(config.data_split_seed)
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project_name,
            name=config.wandb_run_name,
            config=config,
        )

    split_train_test_data_by_user(
        config.processed_data_dir,
        data_file=config.data_file,
        validation_prop=config.validation_prop,
        test_prop=config.test_prop,
        cold_users_prop=config.cold_users_prop,
        cold_items_prop=config.cold_items_prop,
    )

    # Create MoocDataset instance for dataset.
    # ========== BEGIN ========== #

    for set_name in ["train", "test", "validation"]:

        print(f"Loading dataset from folder: {config.processed_data_dir}")
        dataset = Dataset(config.processed_data_dir, config.KG_ARGS, set_name)
        save_dataset(config.processed_data_dir, dataset, config.use_wandb) #dataset.pkl

        kg = KnowledgeGraph(
            dataset,
            config.KG_ARGS,
            set_name=set_name,
            use_user_relations=config.use_user_relations,
            use_entity_relations=config.use_entity_relations,
        )
        kg.compute_degrees() 
        save_kg(config.processed_data_dir, kg, config.use_wandb) #kg.pkl
    # =========== END =========== #

    # Genereate train/test labels.
    # ========== BEGIN ========== #
    print("Generate train/test labels.")
    train_labels = generate_labels(config.processed_data_dir, "train.txt")
    test_labels = generate_labels(config.processed_data_dir, "test.txt")
    validation_labels = generate_labels(config.processed_data_dir, "validation.txt")

    save_labels(
        config.processed_data_dir,
        train_labels,
        mode="train",
        use_wandb=config.use_wandb,
    )
    save_labels(
        config.processed_data_dir, test_labels, mode="test", use_wandb=config.use_wandb
    )
    save_labels(
        config.processed_data_dir,
        validation_labels,
        mode="validation",
        use_wandb=config.use_wandb,
    )

    # =========== END =========== #
    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
