
import math
import random
import numpy as np
import os
import sys
from tqdm import tqdm
# sys.path.append('..')
from Graph_generate.amazon_graph import AmazonGraph
from Graph_generate.amazon_data_process import AmazonDataset
from collections import namedtuple
import argparse
from itertools import count, chain
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import *
from sum_tree import SumTree

#TODO select env
from RL.env_binary_question import BinaryRecommendEnv
from RL.env_enumerated_question import EnumeratedRecommendEnv
# from RL.env_binary_question_pgpr import BinaryRecommendEnvPGPR
from RL.env_binary_question_pgpr_feature import BinaryRecommendEnvPGPR
from RL.RL_evaluate import dqn_evaluate
from RL_model import Agent, ReplayMemoryPER
from gcn import GraphEncoder
import time
import warnings
import json

warnings.filterwarnings("ignore")
EnvDict = {
    LAST_FM: BinaryRecommendEnv,
    LAST_FM_STAR: BinaryRecommendEnv,
    YELP: EnumeratedRecommendEnv,
    YELP_STAR: BinaryRecommendEnv,
    AMAZON: BinaryRecommendEnv,
    AMAZON_STAR: BinaryRecommendEnv,
    BEAUTY : BinaryRecommendEnvPGPR,
    CELLPHONES : BinaryRecommendEnvPGPR, 
    CLOTH : BinaryRecommendEnvPGPR, 
    CDS : BinaryRecommendEnvPGPR
    }

FeatureDict = {
    LAST_FM: 'feature',
    LAST_FM_STAR: 'feature',
    YELP: 'large_feature',
    YELP_STAR: 'feature',
    AMAZON: 'feature',
    AMAZON_STAR: 'feature',
    BEAUTY : 'feature',
    CELLPHONES : 'feature',
    CLOTH : 'feature',
    CDS : 'feature'
}

def evaluate(args, kg, dataset, filename):
    test_env = EnvDict[args.data_name](
        kg, 
        dataset, 
        args.data_name, 
        args.embed, 
        seed=args.seed, 
        max_turn=args.max_turn,
        cand_num=args.cand_num, 
        cand_item_num=args.cand_item_num, 
        attr_num=args.attr_num, 
        mode=args.mode,
        ask_num=args.ask_num, 
        entropy_way=args.entropy_method,
        fm_epoch=args.fm_epoch, 
        domain=args.domain
    )
    
    set_random_seed(args.seed)
    memory = ReplayMemoryPER(args.memory_size) #10000
    embed = torch.FloatTensor(
        np.concatenate((test_env.ui_embeds, test_env.feature_emb, np.zeros((1,test_env.ui_embeds.shape[1]))), axis=0))
    gcn_net = GraphEncoder(
        device=args.device, 
        entity=embed.size(0), 
        emb_size=embed.size(1), 
        kg=kg, 
        embeddings=embed, 
        fix_emb=args.fix_emb, 
        seq=args.seq, 
        gcn=args.gcn, 
        hidden_size=args.hidden
    ).to(args.device)
    agent = Agent(
        device=args.device, 
        memory=memory, 
        state_size=args.hidden, 
        action_size=embed.size(1), 
        hidden_size=args.hidden, 
        gcn_net=gcn_net, 
        learning_rate=args.learning_rate, 
        l2_norm=args.l2_norm, 
        PADDING_ID=embed.size(0)-1
    )
    print('Staring loading rl model in epoch {}'.format(args.load_rl_epoch))
    print(args.data_name)
    print(filename)
    print(args.load_rl_epoch)
    
    agent.load_model(
        data_name=args.data_name, 
        filename=filename, 
        epoch_user=args.load_rl_epoch)

    tt = time.time()
    start = tt

    SR5, SR10, SR15, AvgT, Rank = 0, 0, 0, 0, 0
    SR_turn_15 = [0]* args.max_turn
    turn_result = []
    result = []
    user_size = test_env.ui_array.shape[0]
    print('User-Item Pair')
    print(test_env.ui_array)
    print('User size in UI_test: ', user_size)
    test_filename = 'Evaluate-epoch-{}-'.format(args.load_rl_epoch) + filename
    
    ###BEAUTY, CELLPHONES, CLOTH, CDS
    if args.data_name in [BEAUTY, CELLPHONES, CLOTH, CDS]:  
        if args.domain == 'Beauty':
            if args.eval_num == 1:
                test_size = int(0.1 * user_size)
            else:
                # test_size = int(0.01 * user_size)
                test_size = user_size
        elif args.domain == 'CDs':
            if args.eval_num == 1:
                test_size = int(0.1 * user_size)
            else:
                # test_size = int(0.003 * user_size) 
                test_size = user_size
        elif args.domain == 'Clothing':
            if args.eval_num == 1:
                test_size = int(0.1 * user_size)
            else:
                # test_size = int(0.01 * user_size)
                test_size = user_size
        elif args.domain == 'Cellphones':
            if args.eval_num == 1:
                test_size = int(0.1 * user_size)
            else:
                # test_size = int(0.01 * user_size)
                test_size = user_size
        user_size = test_size
    
    print('The select Test size : ', user_size)
    user_preferences = {}
    # At the beginning of your code, initialize an empty dictionary to store the user preferences
    for user_num in tqdm(range(user_size), desc='User Sampling'):  #user_size
        # TODO uncommend this line to print the dialog process
        blockPrint()
        print('\n================test tuple:{}===================='.format(user_num))
        success = False
        while not success:
            try:
                state, cand, action_space = test_env.reset()  # Reset environment and record the starting state
                success = True  # If reset succeeds, mark success as True to exit the loop
            except Exception as e:
                # test_env.increment_test_num()
                print(f"Error occurred durievaluateng reset: {e}")
                print("Retrying...")
                time.sleep(0.5)  # Wait for 1 second before retrying (adjust as needed)
        # state, cand, action_space = test_env.reset()  # Reset environment and record the starting state
        is_last_turn = False
        for t in count():  # user  dialog
            if t == 14:
                is_last_turn = True
            action, sorted_actions = agent.select_action(state, cand, action_space, is_test=True, is_last_turn=is_last_turn)
            next_state, next_cand, action_space, reward, done = test_env.step(action.item(), sorted_actions)
            reward = torch.tensor([reward], device=args.device, dtype=torch.float)
            if done:
                next_state = None
            state = next_state
            cand = next_cand
            if done:
                enablePrint()
                print(f'Turn: {t}, Reward: {reward.item()}')
                if reward.item() == 1:  # recommend successfully
                    SR_turn_15 = [v+1 if i>t  else v for i, v in enumerate(SR_turn_15) ]
                    if t < 5:
                        SR5 += 1
                        SR10 += 1
                        SR15 += 1
                    elif t < 10:
                        SR10 += 1
                        SR15 += 1
                    else:
                        SR15 += 1
                    Rank += (1/math.log(t+3,2) + (1/math.log(t+2,2)-1/math.log(t+3,2))/math.log(done+1,2))
                else:
                    Rank += 0
                AvgT += t+1
                break

        if (user_num+1) % args.observe_num == 0 and user_num >= 0:
            SR = [
                SR5/args.observe_num, 
                SR10/args.observe_num, 
                SR15/args.observe_num, 
                AvgT / args.observe_num, 
                Rank / args.observe_num
            ]
            SR_TURN = [i/args.observe_num for i in SR_turn_15]
            print('Total evalueation epoch_uesr:{}'.format(user_num + 1))
            print('Takes {} seconds to finish {}% of this task'.format(str(time.time() - start),
                                                                       float(user_num) * 100 / user_size))
            print('SR5:{}, SR10:{}, SR15:{}, AvgT:{}, Rank:{} '
                  'Total epoch_uesr:{}'.format(SR5 / args.observe_num, SR10 / args.observe_num, SR15 / args.observe_num,
                                                AvgT / args.observe_num, Rank / args.observe_num, user_num + 1))
            result.append(SR)
            turn_result.append(SR_TURN)
            SR5, SR10, SR15, AvgT, Rank = 0, 0, 0, 0, 0
            SR_turn_15 = [0] * args.max_turn
            tt = time.time()
        
        # user's profile
        user_acc_feature = test_env.user_acc_feature
        user_rej_feature = test_env.user_rej_feature
        user_rej_items = list(set(test_env.user_rej_items))
        cand_items = test_env.cand_items #.tolist()
        idx_user = test_env.ui_array[user_num][0].tolist()
        idx_item = test_env.ui_array[user_num][1].tolist()
        # print("user's profile")
        # print('user_id =', idx_user)
        # print('target_item =', idx_item)
        # print('user_acc_feature =', user_acc_feature)
        # print('user_rej_feature =', user_rej_feature)
        # print('user_rej_items =', user_rej_items)
        # print('Number of cand_items', len(cand_items))
        
        # Store user profile in the dictionary
        user_preferences[str(user_num)] = {
            "idx_user": idx_user,
            "idx_item": idx_item,
            "user_acc_feature": user_acc_feature,
            "user_rej_feature": user_rej_feature,
            "user_rej_items": user_rej_items
        }
        
        # enablePrint()
    
    # At the end of your code, save the dictionary to a JSON file
    with open(TMP_DIR[args.data_name] + f'/user_preference_{args.domain}.json' , 'w') as f:
        json.dump(user_preferences, f)
        
    SR5_mean = np.mean(np.array([item[0] for item in result]))
    SR10_mean = np.mean(np.array([item[1] for item in result]))
    SR15_mean = np.mean(np.array([item[2] for item in result]))
    AvgT_mean = np.mean(np.array([item[3] for item in result]))
    Rank_mean = np.mean(np.array([item[4] for item in result]))
    SR_all = [SR5_mean, SR10_mean, SR15_mean, AvgT_mean, Rank_mean]
    save_rl_mtric(
        dataset=args.data_name, 
        filename=filename, 
        epoch=user_num, 
        SR=SR_all, 
        spend_time=time.time() - start,
        mode='test'
    )
    save_rl_mtric(
        dataset=args.data_name, 
        filename=test_filename, 
        epoch=user_num, 
        SR=SR_all, 
        spend_time=time.time() - start,
        mode='test'
    )  # save RL SR
    print('save test evaluate successfully!')

    SRturn_all = [0] * args.max_turn
    for i in range(len(SRturn_all)):
        SRturn_all[i] = np.mean(np.array([item[i] for item in turn_result]))
    print('success turn:{}'.format(SRturn_all))
    print('SR5:{}, SR10:{}, SR15:{}, AvgT:{}, Rank:{}'.format(SR5_mean, SR10_mean, SR15_mean, AvgT_mean, Rank_mean))
    PATH = TMP_DIR[args.data_name] + '/RL-log-merge/' + test_filename + '.txt'
    with open(PATH, 'a') as f:
        #f.write('Training epocch:{}\n'.format(i_episode))
        f.write('===========Test Turn===============\n')
        f.write('Testing {} user tuples\n'.format(user_num))
        for i in range(len(SRturn_all)):
            f.write('Testing SR-turn@{}: {}\n'.format(i, SRturn_all[i]))
        f.write('================================\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-seed', type=int, default=1, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', '-me', type=int, default=50000, help='the number of RL train epoch')
    parser.add_argument('--fm_epoch', type=int, default=0, help='the epoch of FM embedding')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate.')
    parser.add_argument('--l2_norm', type=float, default=1e-6, help='l2 regularization.')
    parser.add_argument('--hidden', type=int, default=100, help='number of samples')
    parser.add_argument('--memory_size', type=int, default=50000, help='size of memory ')

    parser.add_argument('--data_name', type=str, default=AMAZON, choices=[BEAUTY, CELLPHONES, CLOTH, CDS, AMAZON, AMAZON_STAR, LAST_FM, LAST_FM_STAR, YELP, YELP_STAR],
                        help='One of {BEAUTY, CELLPHONES, CLOTH, CDS, AMAZON, AMAZON_STAR, LAST_FM, LAST_FM_STAR, YELP, YELP_STAR}.')
    parser.add_argument('--domain', type=str, default='Beauty', choices=['Beauty','Cellphones', 'Clothing', 'CDs'],
                        help='One of {CDs, Beauty, Clothing, Cellphones,}.')
    parser.add_argument('--entropy_method', type=str, default='weight_entropy', help='entropy_method is one of {entropy, weight entropy}')
    # Although the performance of 'weighted entropy' is better, 'entropy' is an alternative method considering the time cost.
    parser.add_argument('--max_turn', type=int, default=15, help='max conversation turn')
    parser.add_argument('--cand_len_size', type=int, default=20, help='binary state size for the length of candidate items')
    parser.add_argument('--attr_num', type=int, help='the number of attributes')
    parser.add_argument('--mode', type=str, default='test', help='the mode in [train, test, test_cold_start]')
    parser.add_argument('--ask_num', type=int, default=1, help='the number of features asked in a turn')
    parser.add_argument('--observe_num', type=int, default=5, help='the number of epochs to save RL model and metric')
    parser.add_argument('--load_rl_epoch', type=int, default=0, help='the epoch of loading RL model')

    parser.add_argument('--sample_times', type=int, default=100, help='the epoch of sampling')
    parser.add_argument('--max_steps', type=int, default=100, help='max training steps')
    parser.add_argument('--eval_num', type=int, default=10, help='the number of epochs to save RL model and metric')
    parser.add_argument('--cand_num', type=int, default=10, help='candidate sampling number')
    parser.add_argument('--cand_item_num', type=int, default=10, help='candidate item sampling number')
    parser.add_argument('--fix_emb', type=bool, default=True, help='fix embedding or not')
    parser.add_argument('--embed', type=str, default='None', help='pretrained embeddings', choices=['transe', None])
    parser.add_argument('--seq', type=str, default='transformer', choices=['rnn', 'transformer', 'mean'], help='sequential learning method')
    parser.add_argument('--gcn', action='store_false', help='use GCN or not')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print(args.device)
    print('data_set:{}'.format(args.data_name))
    if args.data_name == AMAZON:
        kg = AmazonGraph(args.domain)    
    else:
        kg = load_kg(args.data_name, mode='test')
        print(kg.G.keys())
        print('Number of user',len(kg.G['user']))
        print('Number of item',len(kg.G['item']))
        print('Number of category',len(kg.G['category']))
    #reset attr_num
    feature_name = FeatureDict[args.data_name]
    feature_length = len(kg.G[feature_name].keys())
    print('dataset:{}, feature_length:{}'.format(args.data_name, feature_length))
    args.attr_num = feature_length  # set attr_num  = feature_length
    print('args.attr_num:', args.attr_num)
    print('args.entropy_method:', args.entropy_method)

    if args.data_name == AMAZON:
        dataset = AmazonDataset(args.domain)    
    else:
        dataset = load_dataset(args.data_name, mode='test')
        print('Number of user', getattr(dataset, 'user').vocab_size)
        print('Number of item', getattr(dataset, 'item').vocab_size)
        print('Number of category', getattr(dataset, 'category').vocab_size)
    
    filename = 'train-data-{}-RL-cand_num-{}-cand_item_num-{}-embed-{}-seq-{}-gcn-{}'.format(
        args.data_name, args.cand_num, args.cand_item_num, args.embed, args.seq, args.gcn)
    evaluate(args, kg, dataset, filename)

if __name__ == '__main__':
    main()