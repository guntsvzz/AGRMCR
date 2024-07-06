from __future__ import absolute_import, division, print_function
import json
from collections import Counter
import os
import argparse
from math import log
from tqdm.auto import tqdm
from easydict import EasyDict as edict
import torch
from functools import reduce
from kg_env_m import BatchKGEnvironment
from actor_critic import ActorCritic
from utils import *
import wandb
from make_cold_start_kg import InitalUserEmbedding

def evaluate(
    topk_matches,
    test_user_products,
    train_user_products,
    use_wandb,
    dir_path,
    result_file_dir,
    k=10,
    min_items=10,
    compute_all=True,
):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of product ids in ascending order.
    """

    # computes the user relations distributions from the kg
    user_relations = Counter()
    kg = load_kg(dir_path, set_name="test")
    for uid, relations in kg.G["user"].items():
        for entities in relations.values():
            user_relations[uid] += len(entities)

    invalid_users = []

    # Metrics for all users
    user_metrics = dict()

    # Compute metrics
    precisions, recalls, ndcgs, hits, hits_at_1, hits_at_3, hits_at_5 = ([], [], [], [], [], [], [], )
    (
        precisions_all,
        recalls_all,
        ndcgs_all,
        hits_all,
        hits_at_1_all,
        hits_at_3_all,
        hits_at_5_all,
    ) = ([], [], [], [], [], [], [])
    test_user_idxs = list(test_user_products.keys())
    for uid in test_user_idxs:
        is_invalid = False
        if uid not in topk_matches or len(topk_matches[uid]) < min_items:
            invalid_users.append(uid)
            is_invalid = True
        pred_list, rel_set = topk_matches.get(uid, []), test_user_products[uid]
        nb_train = len(train_user_products.get(uid, []))
        if len(pred_list) == 0:
            ndcgs_all.append(0.0)
            recalls_all.append(0.0)
            precisions_all.append(0.0)
            hits_all.append(0.0)
            hits_at_1_all.append(0.0)
            hits_at_3_all.append(0.0)
            hits_at_5_all.append(0.0)
            continue

        if is_invalid == False:
            dcg = 0.0
            hit_num = 0.0
            hit_at_1 = 0.0
            hit_at_3 = 0.0
            hit_at_5 = 0.0

            for i in range(len(pred_list)):
                if pred_list[i] in rel_set:
                    dcg += 1.0 / (log(i + 2) / log(2))
                    hit_num += 1
                    if i < 1:
                        hit_at_1 += 1
                    if i < 3:
                        hit_at_3 += 1
                    if i < 5:
                        hit_at_5 += 1
            # idcg
            idcg = 0.0
            for i in range(min(len(rel_set), k)):
                idcg += 1.0 / (log(i + 2) / log(2))
            ndcg = dcg / idcg
            recall = hit_num / len(rel_set)
            precision = hit_num / k
            hit = 1.0 if hit_num > 0.0 else 0.0
            hit_at_1 = 1.0 if hit_at_1 > 0.0 else 0.0
            hit_at_3 = 1.0 if hit_at_3 > 0.0 else 0.0
            hit_at_5 = 1.0 if hit_at_5 > 0.0 else 0.0

            ndcgs.append(ndcg)
            recalls.append(recall)
            precisions.append(precision)
            hits.append(hit)
            hits_at_1.append(hit_at_1)
            hits_at_3.append(hit_at_3)
            hits_at_5.append(hit_at_5)

            ndcgs_all.append(ndcg)
            recalls_all.append(recall)
            precisions_all.append(precision)
            hits_all.append(hit)
            hits_at_1_all.append(hit_at_1)
            hits_at_3_all.append(hit_at_3)
            hits_at_5_all.append(hit_at_5)

        elif compute_all == True:
            dcg_all = 0.0
            hit_num_all = 0.0
            hit_at_1_all = 0.0
            hit_at_3_all = 0.0
            hit_at_5_all = 0.0
            for i in range(len(pred_list)):
                if pred_list[i] in rel_set:
                    dcg_all += 1.0 / (log(i + 2) / log(2))
                    hit_num_all += 1
                    if i < 1:
                        hit_at_1_all += 1
                    if i < 3:
                        hit_at_3_all += 1
                    if i < 5:
                        hit_at_5_all += 1
            # idcg
            idcg_all = 0.0
            for i in range(min(len(rel_set), k)):
                idcg_all += 1.0 / (log(i + 2) / log(2))
            ndcg_all = dcg_all / idcg_all
            recall_all = hit_num_all / len(rel_set)
            precision_all = hit_num_all / k
            hit_all = 1.0 if hit_num_all > 0.0 else 0.0
            hit_at_1_all = 1.0 if hit_at_1_all > 0.0 else 0.0
            hit_at_3_all = 1.0 if hit_at_3_all > 0.0 else 0.0
            hit_at_5_all = 1.0 if hit_at_5_all > 0.0 else 0.0
            ndcgs_all.append(ndcg_all)
            recalls_all.append(recall_all)
            precisions_all.append(precision_all)
            hits_all.append(hit_all)
            hits_at_1_all.append(hit_at_1_all)
            hits_at_3_all.append(hit_at_3_all)
            hits_at_5_all.append(hit_at_5_all)
        else:
            ndcgs_all.append(0.0)
            recalls_all.append(0.0)
            precisions_all.append(0.0)
            hits_all.append(0.0)
            hits_at_1_all.append(0.0)
            hits_at_3_all.append(0.0)
            hits_at_5_all.append(0.0)
        user_metrics[uid] = {
            "ndcg": ndcgs_all[-1] * 100,
            "recall": recalls_all[-1] * 100,
            "hit": hits_all[-1] * 100,
            "precision": precisions_all[-1] * 100,
            "predictions": pred_list,
            "nb_train": nb_train,
            "nb_relations": user_relations[uid],
        }

    avg_precision = np.mean(precisions) * 100
    avg_recall = np.mean(recalls) * 100
    avg_ndcg = np.mean(ndcgs) * 100
    avg_hit = np.mean(hits) * 100
    avg_hit_at_1 = np.mean(hits_at_1) * 100
    avg_hit_at_3 = np.mean(hits_at_3) * 100
    avg_hit_at_5 = np.mean(hits_at_5) * 100

    avg_precision_all = np.mean(precisions_all) * 100
    avg_recall_all = np.mean(recalls_all) * 100
    avg_ndcg_all = np.mean(ndcgs_all) * 100
    avg_hit_all = np.mean(hits_all) * 100
    avg_hit_at_1_all = np.mean(hits_at_1_all) * 100
    avg_hit_at_3_all = np.mean(hits_at_3_all) * 100
    avg_hit_at_5_all = np.mean(hits_at_5_all) * 100

    # print(
    #     "Min items to consider user valid={:.3f} |  Compute metrics for all users={}\n".format(
    #         min_items, compute_all
    #     )
    # )

    # print(
    #     "NDCG={:.3f} |  Recall={:.3f} | HR={:.3f} | Precision={:.3f} | HR@1={:.3f} | HR@3={:.3f} | HR@5={:.3f} | Invalid users={}\n".format(
    #         avg_ndcg,
    #         avg_recall,
    #         avg_hit,
    #         avg_precision,
    #         avg_hit_at_1,
    #         avg_hit_at_3,
    #         avg_hit_at_5,
    #         len(invalid_users),
    #     )
    # )
    print(
        "NDCG={:.3f} |  Recall={:.3f} | HR={:.3f} | Precision={:.3f} | HR@1={:.3f} | HR@3={:.3f} | HR@5={:.3f} | Computed for all users.\n".format(
            avg_ndcg_all,
            avg_recall_all,
            avg_hit_all,
            avg_precision_all,
            avg_hit_at_1_all,
            avg_hit_at_3_all,
            avg_hit_at_5_all,
        )
    )
    filename = os.path.join(result_file_dir, "metrics.json")

    metrics_all = {
        "ndcg": avg_ndcg_all,
        "recall": avg_recall_all,
        "hit": avg_hit_all,
        "precision": avg_precision_all,
    }

    json.dump(metrics_all, open(filename, "w"), indent=4)

    filename = os.path.join(result_file_dir, "user_metrics.json")
    json.dump(user_metrics, open(filename, "w"), indent=4)

    if use_wandb:
        wandb.save(filename)

    return avg_precision, avg_recall, avg_ndcg, avg_hit


def evaluate_validation(topk_matches, test_user_products):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of product ids in ascending order.
    """
    invalid_users = []
    # Compute metrics
    hits = []
    test_user_idxs = list(test_user_products.keys())
    for uid in test_user_idxs:
        if uid not in topk_matches or len(topk_matches[uid]) < 1:
            invalid_users.append(uid)
            continue
        pred_list, rel_set = topk_matches[uid], test_user_products[uid]
        if len(pred_list) == 0:
            continue

        hit_num = 0.0
        for i in range(len(pred_list)):
            if pred_list[i] in rel_set:
                hit_num += 1

        hit = 1.0 if hit_num > 0.0 else 0.0
        hits.append(hit)

    avg_hit = np.mean(hits) * 100

    print(" HR={:.3f} | Invalid users={}\n".format(avg_hit, len(invalid_users)))

    return avg_hit

def batch_beam_search_cold_start(env, model, kg_config, uids, device, topk=[10, 3, 1], policy=0, user_pref_embed=None):
    def _batch_acts_to_masks(batch_acts):
        batch_masks = []
        for acts in batch_acts:
            num_acts = len(acts)
            act_mask = np.zeros(env.act_dim, dtype=np.uint8)
            act_mask[:num_acts] = 1
            # if first_action:
            #     for i, act in enumerate(acts):
            #         if act[0] == env.kg_args.interaction:
            #             act_mask[i] = 0
            batch_masks.append(act_mask)
        return np.vstack(batch_masks)

    state_pool = env.reset(uids, user_pref_embed, args.embeds_type)  # numpy of [bs, dim]
    path_pool = env._batch_path  # list of list, size=bs
    probs_pool = [[] for _ in uids]
    model.eval()
    for hop in range(len(topk)):
        # first_action = hop == 1
        state_tensor = torch.FloatTensor(state_pool).to(device)
        acts_pool = env._batch_get_actions(path_pool, False, user_pref_embed)  # list of list, size=bs
        actmask_pool = _batch_acts_to_masks(acts_pool)  # numpy of [bs, dim]
        actmask_tensor = torch.ByteTensor(actmask_pool).to(device)
        batch_act_embeddings = env.batch_action_embeddings(
            path_pool, acts_pool
        )  # numpy array of size [bs, 2*embed_size, act_dim]
        embeddings = torch.ByteTensor(batch_act_embeddings).to(device)
        probs, _ = model(
            (state_tensor, actmask_tensor, embeddings)
        )  # Tensor of [bs, act_dim]
        probs = probs + actmask_tensor.float()  # In order to differ from masked actions
        topk_probs, topk_idxs = torch.topk(
            probs, topk[hop], dim=1
        )  # LongTensor of [bs, k]
        topk_idxs = topk_idxs.detach().cpu().numpy()
        topk_probs = topk_probs.detach().cpu().numpy()

        new_path_pool, new_probs_pool = [], []
        for row in range(topk_idxs.shape[0]):
            path = path_pool[row]
            probs = probs_pool[row]
            for idx, p in zip(topk_idxs[row], topk_probs[row]):
                if idx >= len(acts_pool[row]):  # act idx is invalid
                    continue
                relation, next_node_id = acts_pool[row][idx]  # (relation, next_node_id)
                if relation == kg_config.self_loop:
                    next_node_type = path[-1][1]
                else:
                    next_node_type = kg_config.kg_relation[path[-1][1]][relation]
                new_path = path + [(relation, next_node_type, next_node_id)]
                new_path_pool.append(new_path)
                new_probs_pool.append(probs + [p])
        path_pool = new_path_pool
        probs_pool = new_probs_pool
        if hop < len(topk) - 1:  # no need to update state at the last hop
            state_pool = env._batch_get_state(path_pool)
    return path_pool, probs_pool


def batch_beam_search(env, model, kg_config, uids, device, topk=[10, 3, 1], policy=0):
    def _batch_acts_to_masks(batch_acts):
        batch_masks = []
        for acts in batch_acts:
            num_acts = len(acts)
            act_mask = np.zeros(env.act_dim, dtype=np.uint8)
            act_mask[:num_acts] = 1
            # if first_action:
            #     for i, act in enumerate(acts):
            #         if act[0] == env.kg_args.interaction:
            #             act_mask[i] = 0
            batch_masks.append(act_mask)
        return np.vstack(batch_masks)

    state_pool = env.reset(uids)  # numpy of [bs, dim]
    path_pool = env._batch_path  # list of list, size=bs
    probs_pool = [[] for _ in uids]
    model.eval()
    for hop in range(len(topk)):
        # first_action = hop == 1
        state_tensor = torch.FloatTensor(state_pool).to(device)
        acts_pool = env._batch_get_actions(path_pool, False)  # list of list, size=bs
        actmask_pool = _batch_acts_to_masks(acts_pool)  # numpy of [bs, dim]
        actmask_tensor = torch.ByteTensor(actmask_pool).to(device)
        batch_act_embeddings = env.batch_action_embeddings(
            path_pool, acts_pool
        )  # numpy array of size [bs, 2*embed_size, act_dim]
        embeddings = torch.ByteTensor(batch_act_embeddings).to(device)
        probs, _ = model(
            (state_tensor, actmask_tensor, embeddings)
        )  # Tensor of [bs, act_dim]
        probs = probs + actmask_tensor.float()  # In order to differ from masked actions
        topk_probs, topk_idxs = torch.topk(
            probs, topk[hop], dim=1
        )  # LongTensor of [bs, k]
        topk_idxs = topk_idxs.detach().cpu().numpy()
        topk_probs = topk_probs.detach().cpu().numpy()

        new_path_pool, new_probs_pool = [], []
        for row in range(topk_idxs.shape[0]):
            path = path_pool[row]
            probs = probs_pool[row]
            for idx, p in zip(topk_idxs[row], topk_probs[row]):
                if idx >= len(acts_pool[row]):  # act idx is invalid
                    continue
                relation, next_node_id = acts_pool[row][idx]  # (relation, next_node_id)
                if relation == kg_config.self_loop:
                    next_node_type = path[-1][1]
                else:
                    next_node_type = kg_config.kg_relation[path[-1][1]][relation]
                new_path = path + [(relation, next_node_type, next_node_id)]
                new_path_pool.append(new_path)
                new_probs_pool.append(probs + [p])
        path_pool = new_path_pool
        probs_pool = new_probs_pool
        if hop < len(topk) - 1:  # no need to update state at the last hop
            state_pool = env._batch_get_state(path_pool)
    return path_pool, probs_pool

# Cold start user
def update_paths_with_uid(paths, uid):
    updated_paths = []
    for path in paths:
        updated_path = [(path[0][0], path[0][1], uid)] + path[1:]
        updated_paths.append(updated_path)
    return updated_paths
        
def predict_paths(
    policy_file, path_file, config, config_agent, kg_config, set_name="test"
):
    print("Predicting paths...")
    env = BatchKGEnvironment(
        config.processed_data_dir,
        kg_config,
        set_name=set_name,
        max_acts=config_agent.max_acts,
        max_path_len=config_agent.max_path_len,
        state_history=config_agent.state_history,
        reward_function=config_agent.reward,
        mask_first_interaction=True,
        use_pattern=config_agent.use_pattern,
    )
    pretrain_sd = torch.load(policy_file, map_location=torch.device("cpu"))
    model = ActorCritic(
        env.state_dim,
        env.act_dim,
        gamma=config_agent.gamma,
        hidden_sizes=config_agent.hidden,
        modified_policy=config_agent.modified_policy,
        embed_size=env.embed_size,
    ).to(config_agent.device)
    model_sd = model.state_dict()
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)


    if 'test' in set_name:
    # if set_name in ['test', 'test_cold_start', 'test_cold_start_trim', 'test_cold_start_mix', 'test_cold_start_mix_trim']:
        test_labels = load_labels(config.processed_data_dir, "test")
    else:
        test_labels = load_labels(config.processed_data_dir, set_name)
    test_uids = list(test_labels.keys())

    if set_name in ['test', 'test_cold_start']:
        batch_size = 16
        start_idx = 0
        all_paths, all_probs = [], []
        pbar = tqdm(total=len(test_uids))
        while start_idx < len(test_uids):
            end_idx = min(start_idx + batch_size, len(test_uids))
            batch_uids = test_uids[start_idx:end_idx]
            paths, probs = batch_beam_search(
                env,
                model,
                kg_config,
                batch_uids,
                config_agent.device,
                topk=config_agent.topk,
                policy=config_agent.modified_policy,
            )
            all_paths.extend(paths)
            all_probs.extend(probs)
            start_idx = end_idx
            pbar.update(batch_size)

        cold_start_uids = {}
    else:
        #######################################Loading user preference#######################################
        if args.domain is not None:
            user_pref = load_user_pref(config.processed_data_dir, args.domain)
        # if args.set_name in ['test', 'test_cold_start', 'test_cold_start_trim', 'test_cold_start_mix', 'test_cold_start_mix_trim']: 
        cold_start_uids = {}
        init_embed = InitalUserEmbedding(set_name="test", config=config)
        for idx in tqdm(range(len(user_pref))):
            user_id = user_pref[str(idx)]['idx_user']
            target_item = user_pref[str(idx)]['idx_item']
            user_acc_feature = user_pref[str(idx)]['user_acc_feature']
            user_rej_feature = user_pref[str(idx)]['user_rej_feature']
            user_rej_items = user_pref[str(idx)]['user_rej_items']
            
            user_preferred = init_embed.user_preference_config(
                user_acc_feature = user_acc_feature, 
                user_rej_feature = user_rej_feature, 
                user_rej_items = user_rej_items, 
            )
            
            user_pref_emb = init_embed.embeds['user'][user_pref[str(idx)]['idx_user']]
            
            idx_cand_user, cand_user_emb = init_embed.distance(user_pref_emb, top_k=6) #N+1 because it wil remove later
            user_preferred['related_user'] = idx_cand_user
            cold_start_uids[user_pref[str(idx)]['idx_user']] = user_preferred
            # break
        print('all_user_pref', len(cold_start_uids))
        #####################################################################################################

        # Convert lists to sets for fast membership checking
        test_uids_set = set(test_uids)
        cold_start_set = set(cold_start_uids)
        # len(test_uids), len(all_user_pref)
        # Extract elements in test_uids but not in all_user_pref
        extracted_uids = test_uids_set - cold_start_set
        # Convert the result back to a list if needed
        non_cold_start_uids = list(extracted_uids)
        # len(non_cold_start_uids)
        assert len(test_uids) == len(cold_start_uids) + len(non_cold_start_uids)

        batch_size = 16
        start_idx = 0
        all_paths, all_probs = [], []
        pbar = tqdm(total=len(non_cold_start_uids))
        # Non-cold start user
        while start_idx < len(non_cold_start_uids):
            end_idx = min(start_idx + batch_size, len(non_cold_start_uids))
            batch_uids = non_cold_start_uids[start_idx:end_idx]
            paths, probs = batch_beam_search(
                env,
                model,
                kg_config,
                batch_uids,
                config_agent.device,
                topk=config_agent.topk,
                policy=config_agent.modified_policy,
            )
            all_paths.extend(paths)
            all_probs.extend(probs)
            start_idx = end_idx
            pbar.update(batch_size)
                    
        start_idx = 0
        for uid in tqdm(cold_start_uids):
            batch_uids = cold_start_uids[uid]['related_user'][1:]
            paths, probs = batch_beam_search_cold_start(
                env,
                model,
                kg_config,
                batch_uids,
                config_agent.device,
                topk=config_agent.topk,
                policy=config_agent.modified_policy,
                user_pref_embed = user_pref_emb #adding user_pref 
            )

            updated_paths = update_paths_with_uid(paths, uid)
            
            all_paths.extend(updated_paths)
            all_probs.extend(probs)
        
    predicts = {"paths": all_paths, "probs": all_probs}
    pickle.dump(predicts, open(path_file, "wb"))
    if config.use_wandb:
        wandb.save(path_file)
        
    return cold_start_uids

def evaluate_paths(
    dir_path,
    path_file,
    train_labels,
    test_labels,
    kg_config,
    use_wandb,
    result_file_dir,
    set_name="test",
    validation=False,
    trim=False,
    cold_start_uids={}
):
    embeds = load_embed(dir_path, set_name)
    user_embeds = embeds["user"]
    interaction_embeds = embeds[kg_config.interaction][0]
    item_embeds = embeds["item"]
    scores = np.dot(user_embeds + interaction_embeds, item_embeds.T)

    # 1) Get all valid paths for each user, compute path score and path probability.
    results = pickle.load(open(path_file, "rb"))
    pred_paths = {uid: {} for uid in test_labels}
    for path, probs in zip(results["paths"], results["probs"]):
        if path[-1][1] != "item":
            continue
        uid = path[0][2]
        if uid not in pred_paths:
            continue
        pid = path[-1][2]
        # 2) Triming item which are assosiacted with user_rej_items     
        if uid in cold_start_uids.keys():
            if (pid in cold_start_uids[uid]['non-purchase']) and (trim is True): 
                continue  # Skip this item if it's in the user_rej_items list
        if pid not in pred_paths[uid]:
            pred_paths[uid][pid] = []
        path_score = scores[uid][pid]
        path_prob = reduce(lambda x, y: x * y, probs)
        pred_paths[uid][pid].append((path_score, path_prob, path))
        
    # 3) Pick best path for each user-product pair, also remove pid if it is in train set.
    best_pred_paths = {}
    for uid in pred_paths:
        train_pids = set(train_labels.get(uid, []))
        # if len(train_pids) == 0:
        #     continue
        best_pred_paths[uid] = []
        for pid in pred_paths[uid]:
            if pid in train_pids:
                continue
            # Get the path with highest probability
            sorted_path = sorted(pred_paths[uid][pid], key=lambda x: x[1], reverse=True)
            best_pred_paths[uid].append(sorted_path[0])

    path_patterns = {}
    for uid in best_pred_paths:
        for path in best_pred_paths[uid]:
            path_pattern = path[2]
            pattern_key = ""
            for node in path_pattern:
                pattern_key += node[0] + "_" + node[1] + "-->"
            path_patterns[pattern_key] = path_patterns.get(pattern_key, 0) + 1

    print(path_patterns)
    filename = os.path.join(result_file_dir, "patterns.json")
    json.dump(path_patterns, open(filename, "w"), indent=4)

    cold_start_users_path = os.path.join(dir_path, "cold_start_users.json")
    cold_start_users = json.load(open(cold_start_users_path, "r"))
    cold_start_users = set(cold_start_users["train"])

    cold_path_patterns = {}
    for uid in best_pred_paths:
        if uid not in cold_start_users:
            for path in best_pred_paths[uid]:
                path_pattern = path[2]
                pattern_key = ""
                for node in path_pattern:
                    pattern_key += node[0] + "_" + node[1] + "-->"
                cold_path_patterns[pattern_key] = (
                    cold_path_patterns.get(pattern_key, 0) + 1
                )

    print(cold_path_patterns)

    filename = os.path.join(result_file_dir, "cold_patterns.json")
    json.dump(cold_path_patterns, open(filename, "w"), indent=4)

    # computes the item distribution from train_labels
    item_distribution = Counter()
    for uid in train_labels:
        item_distribution.update(train_labels[uid])

    filename = os.path.join(result_file_dir, "item_distribution.json")
    json.dump(item_distribution, open(filename, "w"), indent=4)

    # 3) Compute top 10 recommended products for each user.
    sort_by = "score"
    pred_labels = {}
    for uid in best_pred_paths:
        if sort_by == "score":
            sorted_path = sorted(
                best_pred_paths[uid], key=lambda x: (x[0], x[1]), reverse=True
            )
        elif sort_by == "prob":
            sorted_path = sorted(
                best_pred_paths[uid], key=lambda x: (x[1], x[0]), reverse=True
            )
        top_pids = [p[-1][2] for _, _, p in sorted_path]  # from largest to smallest

        pred_labels[uid] = top_pids[:10]  # change order to from smallest to largest!

    if validation == True:
        return evaluate_validation(pred_labels, test_labels)

    else:
        evaluate(
            pred_labels,
            test_labels,
            train_labels,
            use_wandb,
            config.processed_data_dir,
            result_file_dir=result_file_dir,
            min_items=10,
            compute_all=True,
        )


def test(config, set_name):
    config_agent = config.AGENT
    kg_config = config.KG_ARGS
    
    policy_file = config_agent.log_dir + "/tmp_policy_model_epoch_{}.ckpt".format(
        config_agent.epochs
    )
    path_file = config_agent.log_dir + f"/policy_paths_epoch_{config_agent.epochs}_cold_start_{set_name}.pkl"

    train_labels = load_labels(config.processed_data_dir, "train")
    test_labels = load_labels(config.processed_data_dir, "test")

    dataset_name = config.processed_data_dir.split("/")[-1]

    model_name = (
        "UPGPR_len_"
        + str(config_agent.max_path_len)
        + "_"
        + config.AGENT.reward
        + "_"
        + config.TRAIN_EMBEDS.cold_start_embeddings
        + "_mask_"
        + str(config.AGENT.mask_first_interaction)
        + "_max_cold_concept_"
        + str(kg_config.max_nb_cold_entities)
        + "_topk_"
        + "_".join(map(str, config_agent.topk))
        + "_set_name_"
        + "_".join(set_name)
    )

    config_agent.result_file_dir = os.path.join(
        config_agent.result_file_dir, dataset_name, model_name, str(config.seed)
    )

    os.makedirs(
        config_agent.result_file_dir,
        exist_ok=True,
    )     
    
    if config_agent.run_path:
        cold_start_uids = predict_paths(
            policy_file, 
            path_file, 
            config, 
            config_agent, 
            kg_config,
            set_name = set_name
        )
    if config_agent.run_eval:
        evaluate_paths(
            config.processed_data_dir,
            path_file,
            train_labels,
            test_labels,
            kg_config,
            config.use_wandb,
            config_agent.result_file_dir,
            # set_name = "test",
            validation=False,
            trim=args.trim,
            cold_start_uids=cold_start_uids
        )

def load_user_pref(path, domain):
    user_pref_path = os.path.join(path)
    # Load JSON data from a file
    user_pref = json.load(open(f'{user_pref_path}/user_preference_{domain}.json', 'r'))
    return user_pref

if __name__ == "__main__":
    boolean = lambda x: (str(x).lower() == "true")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Config file.", default="config/beauty/graph_reasoning/UPGPR.json", )
    parser.add_argument("--seed", type=int, help="Random seed.", default=0)
    parser.add_argument(
        "--set_name", 
        type=str, 
        help="Set name.", 
        default="test", 
        choices=['train','test', 'test_cold_start', 'test_cold_start_trim', 'test_cold_start_mix', 'test_cold_start_mix_trim']
    )
    parser.add_argument(
        '--domain', 
        type=str, 
        default=None, 
        choices=['Beauty','Cellphones', 'Clothing', 'CDs', None],
        help='One of {CDs, Beauty, Clothing, Cellphones, None}.'
    )
    parser.add_argument("--trim", action="store_true", help="Triming or not", default=False)
    parser.add_argument(
        "--embeds_type", 
        help="mix : user preference + other, past: only other, None: only user preference", 
        default=None,
        choices=['mix','past', None],
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = edict(json.load(f))

    config.seed = args.seed
    config_agent = config.AGENT

    if config.use_wandb:
        wandb.init(
            project=config.wandb_project_name,
            name=config.wandb_run_name,
            config=config,
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = config_agent.gpu
    config_agent.device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

    if config_agent.early_stopping == True:
        with open("early_stopping.txt", "r") as f:
            config_agent.epochs = int(f.read())

    config_agent.log_dir = config.processed_data_dir + "/" + config_agent.name
    test(config, args.set_name)

    if config.use_wandb:
        wandb.finish()
