from torch.distributions.categorical import Categorical


def select_action(p, cadidate, memory):
    dist = Categorical(p.squeeze())
    s = dist.sample()
    if memory is not None: memory.logprobs.append(dist.log_prob(s))
    return cadidate[s], s


def eval_actions(p, actions):
    softmax_dist = Categorical(p)
    ret = softmax_dist.log_prob(actions).reshape(-1)
    entropy = softmax_dist.entropy().mean()
    return ret, entropy


def greedy_select_action(p, candidate):
    _, index = p.squeeze().max(0)
    action = candidate[index]
    return action


def sample_select_action(p, candidate):
    dist = Categorical(p.squeeze())
    s = dist.sample()
    return candidate[s]
