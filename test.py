from PPO import *
from utils import *
import torch
import argparse
from params import configs
import time
import numpy as np
from envs import Env
from ProgramEnv import ProgEnv


def test(modelPath, pars):
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configs.torch_seed)
    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
              num_layers=configs.num_layers,
              neighbor_pooling_type=configs.neighbor_pooling_type,
              input_dim=configs.input_dim,
              hidden_dim=configs.hidden_dim,
              num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
              num_mlp_layers_actor=configs.num_mlp_layers_actor,
              hidden_dim_actor=configs.hidden_dim_actor,
              num_mlp_layers_critic=configs.num_mlp_layers_critic,
              hidden_dim_critic=configs.hidden_dim_critic)
    ppo.policy.load_state_dict(torch.load(modelPath))
    ppo.policy.eval();
    env = ProgEnv(*pars)
    device = torch.device(configs.device)
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size([1, env.action_space.n, env.action_space.n]),
                             n_nodes=env.action_space.n,
                             device=device)
    adj, fea, candidate, mask = env.reset()
    rewards = 0
    actions = []
    times = []
    while True:
        fea_tensor = torch.from_numpy(np.copy(fea)).to(device).float()
        adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse().float()
        candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device).float()
        mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
        with torch.no_grad():
            pi, _ = ppo.policy(x=fea_tensor,
                          graph_pool=g_pool_step,
                          padded_nei=None,
                          adj=adj_tensor,
                          candidate=candidate_tensor.unsqueeze(0),
                          mask=mask_tensor.unsqueeze(0))
            action = greedy_select_action(pi, candidate)
        adj, fea, reward, done, candidate, mask, startTime = env.step(action.item())
        rewards += reward
        actions.append(action.item())
        times.append(startTime)
        if done:
            break
    ActSeq = []
    ModeSeq = []
    TimeSeq = times
    mode_Number = env.Activity_mode_Num
    cum_sum = np.cumsum(mode_Number) - 1
    for act in actions:
        activity = np.where(cum_sum >= act)[0][0].item()
        mode = cum_sum[int(activity)] - act + 1
        ActSeq.append(activity)
        ModeSeq.append(mode)
    return rewards, ActSeq, ModeSeq, TimeSeq


if __name__ == '__main__':
    modelpath = './log/summary/20211118-1709/PPO-ProgramEnv-seed-200.pth'
    pars = (
        configs.filepath, configs.Target_T, configs.price_renew, configs.price_non, configs.penalty0, configs.penalty1)
    rewards, ActSeq, ModeSeq, TimeSeq = test(modelpath, pars)
    print('================================================')
    print('============== The Final Reward is: ============')
    print('{}'.format(np.round(rewards, 3)))
    print('======= The Final Activity Sequence is: ========')
    print(ActSeq)
    print('========= The Final Mode Sequence is: ==========')
    print(ModeSeq)
    print('========= The Final Time Sequence is: ==========')
    print(TimeSeq)
    print('================================================')



