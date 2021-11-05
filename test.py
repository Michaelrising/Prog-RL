from utils import *
import torch
import argparse
from params import configs
import time
import numpy as np
from envs import Env


class PPO:
    def __init__(self,
                 lr,
                 gamma,
                 k_epochs,
                 eps_clip,
                 n_j,
                 n_m,
                 num_layers,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 ):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy = ActorCritic(n_j=n_j,
                                  n_m=n_m,
                                  num_layers=num_layers,
                                  learn_eps=False,
                                  neighbor_pooling_type=neighbor_pooling_type,
                                  input_dim=input_dim,
                                  hidden_dim=hidden_dim,
                                  num_mlp_layers_feature_extract=num_mlp_layers_feature_extract,
                                  num_mlp_layers_actor=num_mlp_layers_actor,
                                  hidden_dim_actor=hidden_dim_actor,
                                  num_mlp_layers_critic=num_mlp_layers_critic,
                                  hidden_dim_critic=hidden_dim_critic,
                                  device=device)
        self.policy_old = deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=configs.decay_step_size,
                                                         gamma=configs.decay_ratio)

        self.V_loss_2 = nn.MSELoss()

    def update(self, memories, n_tasks, g_pool):

        vloss_coef = configs.vloss_coef
        ploss_coef = configs.ploss_coef
        entloss_coef = configs.entloss_coef

        rewards_all_env = []
        adj_mb_t_all_env = []
        fea_mb_t_all_env = []
        candidate_mb_t_all_env = []
        mask_mb_t_all_env = []
        a_mb_t_all_env = []
        old_logprobs_mb_t_all_env = []
        for i in range(len(memories)):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memories[i].r_mb), reversed(memories[i].done_mb)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            rewards = torch.tensor(rewards, dtype=torch.float).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            rewards_all_env.append(rewards)
            adj_mb_t_all_env.append(aggr_obs(torch.stack(memories[i].adj_mb).to(device), n_tasks))
            fea_mb_t = torch.stack(memories[i].fea_mb).to(device)
            fea_mb_t = fea_mb_t.reshape(-1, fea_mb_t.size(-1))
            fea_mb_t_all_env.append(fea_mb_t)
            candidate_mb_t_all_env.append(torch.stack(memories[i].candidate_mb).to(device).squeeze())
            mask_mb_t_all_env.append(torch.stack(memories[i].mask_mb).to(device).squeeze())
            a_mb_t_all_env.append(torch.stack(memories[i].a_mb).to(device).squeeze())
            old_logprobs_mb_t_all_env.append(torch.stack(memories[i].logprobs).to(device).squeeze().detach())

        mb_g_pool = g_pool_cal(g_pool, torch.stack(memories[0].adj_mb).to(device).shape, n_tasks, device)

        for _ in range(self.k_epochs):
            loss_sum = 0
            vloss_sum = 0
            for i in range(len(memories)):
                pis, vals = self.policy(x=fea_mb_t_all_env[i],
                                        graph_pool=mb_g_pool,
                                        adj=adj_mb_t_all_env[i],
                                        candidate=candidate_mb_t_all_env[i],
                                        mask=mask_mb_t_all_env[i],
                                        padded_nei=None)
                logprobs, ent_loss = eval_actions(pis.squeeze(), a_mb_t_all_env[i])
                ratios = torch.exp(logprobs - old_logprobs_mb_t_all_env[i].detach())
                advantages = rewards_all_env[i] - vals.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                v_loss = self.V_loss_2(vals.squeeze(), rewards_all_env[i])
                p_loss = - torch.min(surr1, surr2)
                ent_loss = - ent_loss.clone()
                loss = vloss_coef * v_loss + ploss_coef * p_loss + entloss_coef * ent_loss
                loss_sum += loss
                vloss_sum += v_loss
            self.optimizer.zero_grad()
            loss_sum.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        if configs.decayflag:
            self.scheduler.step()
        return loss_sum.mean().item(), vloss_sum.mean().item()


def aggr_obs(obs_mb, n_node):
    idxs = obs_mb.coalesce().indices()
    vals = obs_mb.coalesce().values()
    new_idx_row = idxs[1] + idxs[0] * n_node
    new_idx_col = idxs[2] + idxs[0] * n_node
    idx_mb = torch.stack((new_idx_row, new_idx_col))
    adj_batch = torch.sparse.FloatTensor(indices=idx_mb,
                                         values=vals,
                                         size=torch.Size([obs_mb.shape[0] * n_node,
                                                          obs_mb.shape[0] * n_node]),
                                         ).to(obs_mb.device)
    return adj_batch


def g_pool_cal(graph_pool_type, batch_size, n_nodes, device):
    if graph_pool_type == 'average':
        elem = torch.full(size=(batch_size[0]*n_nodes, 1),
                          fill_value=1 / n_nodes,
                          dtype=torch.float32,
                          device=device).view(-1)
    else:
        elem = torch.full(size=(batch_size[0] * n_nodes, 1),
                          fill_value=1,
                          dtype=torch.float32,
                          device=device).view(-1)
    idx_0 = torch.arange(start=0, end=batch_size[0],
                         device=device,
                         dtype=torch.long)
    idx_0 = idx_0.repeat(n_nodes, 1).t().reshape((batch_size[0]*n_nodes, 1)).squeeze()

    idx_1 = torch.arange(start=0, end=n_nodes*batch_size[0],
                         device=device,
                         dtype=torch.long)
    idx = torch.stack((idx_0, idx_1))
    graph_pool = torch.sparse.FloatTensor(idx, elem,
                                          torch.Size([batch_size[0],
                                                      n_nodes*batch_size[0]])
                                          ).to(device)
    return graph_pool


device = torch.device(configs.device)
parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument('--Pn_j', type=int, default=100, help='Number of jobs of instances to test')
parser.add_argument('--Pn_m', type=int, default=20, help='Number of modes instances to test')
parser.add_argument('--Nn_j', type=int, default=20, help='Number of jobs on which to be loaded net are trained')
parser.add_argument('--Nn_m', type=int, default=20, help='Number of modes on which to be loaded net are trained')
parser.add_argument('--low', type=int, default=1, help='LB of duration')
parser.add_argument('--high', type=int, default=99, help='UB of duration')
parser.add_argument('--seed', type=int, default=200, help='Seed for validate set generation')
params = parser.parse_args()

N_JOBS_P = params.Pn_j
N_MODES_P = params.Pn_m
LOW = params.low
HIGH = params.high
SEED = params.seed
N_JOBS_N = params.Nn_j
N_MODES_N = params.Nn_m

env = Env(n_j=N_JOBS_P, n_m=N_MODES_P)

ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
          n_j=N_JOBS_P,
          n_m=N_MODES_P,
          num_layers=configs.num_layers,
          neighbor_pooling_type=configs.neighbor_pooling_type,
          input_dim=configs.input_dim,
          hidden_dim=configs.hidden_dim,
          num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
          num_mlp_layers_actor=configs.num_mlp_layers_actor,
          hidden_dim_actor=configs.hidden_dim_actor,
          num_mlp_layers_critic=configs.num_mlp_layers_critic,
          hidden_dim_critic=configs.hidden_dim_critic)
path = './weights/{}.pth'.format(str(N_JOBS_N) + '_' + str(N_MODES_N) + '_' + str(LOW) + '_' + str(HIGH))
ppo.policy.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                         batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
                         n_nodes=env.number_of_tasks,
                         device=device)

from uniform_instance_gen import uni_instance_gen
np.random.seed(SEED)

dataLoaded = np.load('./DataGen/generatedData' + str(N_JOBS_P) + '_' + str(N_MODES_P) + '_Seed' + str(SEED) + '.npy')
dataset = []

for i in range(dataLoaded.shape[0]):
    dataset.append((dataLoaded[i][0], dataLoaded[i][1]))

def test(dataset):
    result = []
    t1 = time.time()
    for i, data in enumerate(dataset):
        adj, fea, candidate, mask = env.reset(data)
        ep_reward = - env.max_endTime
        while True:
            fea_tensor = torch.from_numpy(fea).to(device)
            adj_tensor = torch.from_numpy(adj).to(device)
            candidate_tensor = torch.from_numpy(candidate).to(device)
            mask_tensor = torch.from_numpy(mask).to(device)
            with torch.no_grad():
                pi, _ = ppo.policy(x=fea_tensor,
                                   graph_pool=g_pool_step,
                                   padded_nei=None,
                                   adj=adj_tensor,
                                   candidate=candidate_tensor.unsqueeze(0),
                                   mask=mask_tensor.unsqueeze(0))
                action = greedy_select_action(pi, candidate)

            adj, fea, reward, done, candidate, mask = env.step(action)
            ep_reward += reward

            if done:
                break
        print('Instance' + str(i + 1) + ' makespan:', -ep_reward + env.posRewards)
        result.append(-ep_reward + env.posRewards)
    t2 = time.time()
    print(t2 - t1)
    file_writing_obj = open('./' + 'drltime_' + str(N_JOBS_N) + 'x' + str(N_MODES_N) + '_' + str(N_JOBS_P) + 'x' + str(N_MODES_P) + '.txt', 'w')
    file_writing_obj.write(str((t2 - t1)/len(dataset)))
    np.save('drlResult_' + str(N_JOBS_N) + 'x' + str(N_MODES_N) + '_' + str(N_JOBS_P) + 'x' + str(N_MODES_P) + '_Seed' + str(SEED), np.array(result, dtype=np.single))


if __name__ == '__main__':
    import cProfile
    cProfile.run('test(dataset)', filename='restats')