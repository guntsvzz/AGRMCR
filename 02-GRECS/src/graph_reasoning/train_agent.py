from __future__ import absolute_import, division, print_function

import os
import json
import argparse
import torch
import torch.optim as optim
from easydict import EasyDict as edict

from actor_critic import ActorCritic
from kg_env import BatchKGEnvironment
from utils import *
from validate import *

logger = None


class ACDataLoader(object):
    def __init__(self, uids, batch_size):
        self.uids = np.array(uids)
        self.num_users = len(uids)
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self._rand_perm = np.random.permutation(self.num_users)
        self._start_idx = 0
        self._has_next = True

    def has_next(self):
        return self._has_next

    def get_batch(self):
        if not self._has_next:
            return None
        # Multiple users per batch
        end_idx = min(self._start_idx + self.batch_size, self.num_users)
        batch_idx = self._rand_perm[self._start_idx : end_idx]
        batch_uids = self.uids[batch_idx]
        self._has_next = self._has_next and end_idx < self.num_users
        self._start_idx = end_idx
        return batch_uids.tolist()


def train(config):
    config_agent = config.AGENT
    kg_config = config.KG_ARGS
    env = BatchKGEnvironment(
        config.processed_data_dir,
        kg_config,
        set_name="train",
        max_acts=config_agent.max_acts,
        max_path_len=config_agent.max_path_len,
        state_history=config_agent.state_history,
        reward_function=config_agent.reward,
        mask_first_interaction=config_agent.mask_first_interaction,
        use_pattern=config_agent.use_pattern,
    )
    # only select the users that have interactions
    uids = [
        key
        for key in env.kg.G["user"]
        if env.kg.G["user"][key][env.kg_args.interaction]
    ]
    dataloader = ACDataLoader(uids, config_agent.batch_size)
    model = ActorCritic(
        env.state_dim,
        env.act_dim,
        gamma=config_agent.gamma,
        hidden_sizes=config_agent.hidden,
        modified_policy=config_agent.modified_policy,
        embed_size=env.embed_size,
    ).to(config_agent.device)
    logger.info("Parameters:" + str([i[0] for i in model.named_parameters()]))
    optimizer = optim.Adam(model.parameters(), lr=config_agent.lr)
    avg_reward = 0
    # epochs_no_improve = 0
    # max_hit_rate_validation_set = 0
    # patience = args.patience

    total_losses, total_plosses, total_vlosses, total_entropy, total_rewards = (
        [],
        [],
        [],
        [],
        [],
    )
    step = 0
    model.train()
    for epoch in range(1, config_agent.epochs + 1):
        ### Start epoch ###
        dataloader.reset()
        while dataloader.has_next():
            batch_uids = dataloader.get_batch()
            ### Start batch episodes ###
            batch_state = env.reset(batch_uids)  # numpy array of [bs, state_dim]
            done = False
            first_action = True
            while not done:
                batch_act_mask = env.batch_action_mask(
                    first_action=first_action, dropout=config_agent.act_dropout
                )  # numpy array of size [bs, act_dim]

                batch_act_embeddings = (
                    env.batch_action_embeddings()
                )  # numpy array of size [bs, 2*embed_size, act_dim]
                batch_act_idx = model.select_action(
                    batch_state,
                    batch_act_mask,
                    batch_act_embeddings,
                    config_agent.device,
                )  # int
                batch_state, batch_reward, done = env.batch_step(batch_act_idx)
                model.rewards.append(batch_reward)
                first_action = False
            ### End of episodes ###

            lr = config_agent.lr * max(
                1e-4,
                1.0
                - float(step)
                / (config_agent.epochs * len(uids) / config_agent.batch_size),
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Update policy
            total_rewards.append(np.sum(model.rewards))
            loss, ploss, vloss, eloss = model.update(
                optimizer, config_agent.device, config_agent.ent_weight
            )
            total_losses.append(loss)
            total_plosses.append(ploss)
            total_vlosses.append(vloss)
            total_entropy.append(eloss)
            step += 1

            # Report performance
            if step > 0 and step % 100 == 0:
                avg_reward = np.mean(total_rewards) / config_agent.batch_size
                avg_loss = np.mean(total_losses)
                avg_ploss = np.mean(total_plosses)
                avg_vloss = np.mean(total_vlosses)
                avg_entropy = np.mean(total_entropy)
                (
                    total_losses,
                    total_plosses,
                    total_vlosses,
                    total_entropy,
                    total_rewards,
                ) = ([], [], [], [], [])
                logger.info(
                    "epoch/step={:d}/{:d}".format(epoch, step)
                    + " | loss={:.5f}".format(avg_loss)
                    + " | ploss={:.5f}".format(avg_ploss)
                    + " | vloss={:.5f}".format(avg_vloss)
                    + " | entropy={:.5f}".format(avg_entropy)
                    + " | reward={:.5f}".format(avg_reward)
                )
            if config.quick_run:
                break
        ### END of epoch ###

        tmp_policy_file = "{}/tmp_policy_model_epoch_{}.ckpt".format(
            config.log_dir, epoch
        )
        policy_file = "{}/policy_model_epoch_{}.ckpt".format(config.log_dir, epoch)
        torch.save(model.state_dict(), tmp_policy_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Config file.",
        default="config/coco_01_01/UPGPR_test.json",
    )
    parser.add_argument("--seed", type=int, help="Random seed.", default=0)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = edict(json.load(f))

    config.seed = args.seed

    config_agent = config.AGENT

    if config.use_wandb:
        wandb.init(
            project=config.wandb_project_name, name=config.wandb_run_name, config=config
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = config_agent.gpu
    config_agent.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    config.log_dir = "{}/{}".format(config.processed_data_dir, config_agent.name)
    if not os.path.isdir(config.log_dir):
        os.makedirs(config.log_dir)

    global logger
    logger = get_logger(config.log_dir + "/train_log.txt")

    set_random_seed(config.seed)
    train(config)
    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
