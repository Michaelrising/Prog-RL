from PPO import *
from utils import *
import torch
import argparse
from params import configs
import time
import numpy as np
from envs import Env
from ProgramEnv import ProgEnv


def test(modelPath):
    model = torch.load(modelPath)
    model.eval();
    env = ProgEnv(filename=configs.filepath)
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
            pi, _ = model(x=fea_tensor,
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
        activity = np.where(cum_sum<=act)[0][-1].item()
        mode = act - activity + 1
        ActSeq.append(activity)
        ModeSeq.append(mode)
    return rewards, ActSeq, ModeSeq, TimeSeq

if __name__ == '__main__':
    modelpath = ''
    test(modelpath)