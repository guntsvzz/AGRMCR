from __future__ import absolute_import, division, print_function

import os
import json
import argparse
import torch
import torch.optim as optim
import numpy as np

from copy import deepcopy

from utils import *
from data_utils import DataLoader
from transe_model import KnowledgeEmbedding
from easydict import EasyDict as edict


logger = None


def train(config):
    transe_config = config.TRAIN_EMBEDS
    kg_config = config.KG_ARGS

    dataset = load_dataset(config.processed_data_dir, "train")
    dataloader = DataLoader(
        dataset,
        transe_config.batch_size,
        config.use_user_relations, #true
        config.use_entity_relations, #false
    )
    interactions_to_train = transe_config.epochs * dataset.interactions.size

    model = KnowledgeEmbedding(dataset, transe_config, kg_config).to(
        transe_config.device
    )
    
    print("====MODEL====")
    print(model)
    
    logger.info("Parameters:" + str([i[0] for i in model.named_parameters()]))
    optimizer = optim.Adam(model.parameters(), lr=transe_config.lr)
    steps = 0
    smooth_loss = 0.0
    min_val_loss = np.Inf
    patience = transe_config.patience
    epochs_no_improve = 0

    for epoch in range(1, transe_config.epochs + 1):
        dataloader.reset()
        loss = 0
        while dataloader.has_next():
            # Set learning rate.
            lr = transe_config.lr * max(
                1e-4,
                1.0
                - dataloader.finished_interaction_number / float(interactions_to_train),
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Get training batch.
            batch_idxs = dataloader.get_batch() #batch, 9
            batch_idxs = torch.from_numpy(batch_idxs).to(transe_config.device)

            # Train model.
            optimizer.zero_grad()
            train_loss = model(batch_idxs)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), transe_config.max_grad_norm
            )
            optimizer.step()
            smooth_loss += train_loss.item() / transe_config.steps_per_checkpoint
            loss += train_loss.item()

            steps += 1
            if steps % transe_config.steps_per_checkpoint == 0:
                logger.info(
                    "Epoch: {:02d} | ".format(epoch)
                    + "Interactions: {:d}/{:d} | ".format(
                        dataloader.finished_interaction_number, interactions_to_train
                    )
                    + "Lr: {:.5f} | ".format(lr)
                    + "Smooth loss: {:.5f}".format(smooth_loss)
                )
                smooth_loss = 0.0
            if config.quick_run:
                break

        file_name = "{}/transe_model_sd_epoch_{}.ckpt".format(config.log_dir, epoch)
        torch.save(model.state_dict(), file_name)
        if config.use_wandb:
            wandb.log({"Loss": loss})

        if epoch > transe_config.min_epochs:
            if loss < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = loss
                torch.save(model.state_dict(), file_name)
            else:
                epochs_no_improve += 1

            if epochs_no_improve == patience:
                print("Early stopping after {} epochs".format(epoch))
                # set epochs to number of epochs of best model
                transe_config.epochs = int(epoch - patience)
                break

            if epoch == transe_config.epochs:
                print(
                    "Stoppping after {} epochs, best embeddings after {}".format(
                        epoch, int((epoch - epochs_no_improve))
                    )
                )
                transe_config.epochs = int(epoch - epochs_no_improve)


def extract_embeddings(config):
    transe_config = config.TRAIN_EMBEDS
    kg_config = config.KG_ARGS
    """Note that last entity embedding is of size [vocab_size+1, d]."""
    model_file = "{}/transe_model_sd_epoch_{}.ckpt".format(
        config.log_dir, transe_config.epochs
    )
    print("Load embeddings", model_file)
    state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)

    embeds = {}
    embeds_entity = {
        e: state_dict[f"{e}.weight"].cpu().data.numpy()[:-1] for e in kg_config.entities
    }  # Must remove last dummy 'user' with 0 embed.

    embeds_relation = {
        r: (
            state_dict[f"{r}"].cpu().data.numpy()[0],
            state_dict[f"{r}_bias.weight"].cpu().data.numpy(),
        )
        for r in kg_config.item_relation.keys()
    }

    if config.use_user_relations == True:
        embeds_relation.update(
            {
                r: (
                    state_dict[f"{r}"].cpu().data.numpy()[0],
                    state_dict[f"{r}_bias.weight"].cpu().data.numpy(),
                )
                for r in kg_config.user_relation.keys()
            }
        )

    if config.use_entity_relations == True:
        embeds_relation.update(
            {
                r: (
                    state_dict[f"{r}"].cpu().data.numpy()[0],
                    state_dict[f"{r}_bias.weight"].cpu().data.numpy(),
                )
                for r in kg_config.entity_relation.keys()
            }
        )

    embeds_relation.update(
        {
            kg_config.interaction: (
                state_dict[f"{kg_config.interaction}"].cpu().data.numpy()[0],
                state_dict[f"{kg_config.interaction}_bias.weight"].cpu().data.numpy(),
            )
        }
    )

    embeds.update(embeds_entity)
    embeds.update(embeds_relation)

    return embeds


def make_cold_embeds(config, embeds, use_wandb=False):
    transe_config = config.TRAIN_EMBEDS

    # load cold start users
    cold_users_path = os.path.join(config.processed_data_dir, "cold_start_users.json")
    cold_users = json.load(open(cold_users_path, "r"))

    # load cold start items
    cold_items_path = os.path.join(config.processed_data_dir, "cold_start_items.json")
    cold_items = json.load(open(cold_items_path, "r"))

    # set all cold start users embeddings to 0
    tmp_cold_users = cold_users["test"] + cold_users["validation"]
    embeds["user"][tmp_cold_users] = 0

    # set all cold start items embeddings to 0
    tmp_cold_items = cold_items["test"] + cold_items["validation"]
    embeds["item"][tmp_cold_items] = 0

    # if the zero cold start embeddings are used, save the embeddings and exit
    if transe_config["cold_start_embeddings"] == "zero":
        for set_name in ["train", "validation", "test"]:
            save_embed(
                config.processed_data_dir, f"{set_name}_transe_embed.pkl", embeds
            )
    # set all cold start users embeddings to average of their neighbors minus the interaction embedding
    else:
        # nothing to modify for the training set
        save_embed(
            config.processed_data_dir,
            f"train_transe_embed.pkl",
            embeds,
        )
        for set_name in ["validation", "test"]:
            dataset = load_dataset(config.processed_data_dir, set_name)
            tmp_cold_users = cold_users[set_name]
            tmp_cold_items = cold_items[set_name]
            # making a copy of the embeddings to avoid using the modified cold start embeddings in the next iteration
            tmp_embeds = deepcopy(embeds)
            # Compute the embeddings of cold start users
            if transe_config["cold_start_embeddings"] == "avg":
                for user in tmp_cold_users:
                    nb_relations = 0
                    # for each relation, get all related entities embeddings and average them
                    for relation, entity in dataset.data_args.user_relation.items():
                        # get all entities related to user by relation
                        entities = getattr(dataset, relation, None)["data"][user]
                        # get all related entities embeddings and subtract the relation embedding
                        all_related_emb = (
                            embeds[entity[1]][entities] - embeds[relation][0]
                        )
                        nb_relations += all_related_emb.shape[0]
                        # sum all related entities embeddings
                        tmp_embeds["user"][user] += all_related_emb.sum(axis=0)
                    # divide by the number of relations to get the average
                    if nb_relations > 0:
                        tmp_embeds["user"][user] /= nb_relations
            if (
                transe_config["cold_start_embeddings"] == "avg"
                or transe_config["cold_start_embeddings"] == "item_avg"
            ):
                # Compute the embeddings of cold start items
                for item in tmp_cold_items:
                    nb_relations = 0
                    # for each relation, get all related entities embeddings and average them
                    for relation, entity in dataset.data_args.item_relation.items():
                        # get all entities related to item by relation
                        entities = getattr(dataset, relation, None)["data"][item]
                        # get all related entities embeddings and subtract the relation embedding
                        all_related_emb = (
                            embeds[entity[1]][entities] - embeds[relation][0]
                        )
                        nb_relations += all_related_emb.shape[0]
                        # sum all related entities embeddings
                        tmp_embeds["item"][item] += all_related_emb.sum(axis=0)
                    # divide by the number of relations to get the average
                    if nb_relations > 0:
                        tmp_embeds["item"][item] /= nb_relations

            # save the embeddings
            save_embed(
                config.processed_data_dir, f"{set_name}_transe_embed.pkl", tmp_embeds
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Config file.",
        default="config/coco_01_01/UPGPR_10.json",
    )
    parser.add_argument("--seed", type=int, help="Random seed.", default=0)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = edict(json.load(f))

    config.seed = args.seed
    transe_config = config.TRAIN_EMBEDS
    transe_config.use_user_relations = config.use_user_relations
    transe_config.use_entity_relations = config.use_entity_relations

    assert (
        transe_config.min_epochs <= transe_config.epochs
    ), "Minimum number of epochs should be lower than total number of epochs."

    if config.use_wandb:
        wandb.init(
            project=config.wandb_project_name, name=config.wandb_run_name, config=config
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = transe_config.gpu

    transe_config.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    config.log_dir = "{}/{}".format(config.processed_data_dir, transe_config.name)
    if not os.path.isdir(config.log_dir):
        os.makedirs(config.log_dir)

    global logger
    logger = get_logger(config.log_dir + "/train_log.txt")
    # logger.info(config)

    set_random_seed(config.seed)
    train(config)
    embeds = extract_embeddings(config)
    make_cold_embeds(config, embeds)

    if config.use_wandb:
        wandb.finish()

    print("Done!")


if __name__ == "__main__":
    main()
