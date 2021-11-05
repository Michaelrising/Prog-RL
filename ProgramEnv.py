from ActConstraints import Constraints, ReadInfo
import numpy as np
import gym
from gym import spaces
from collections import deque
import pandas as pd
from gym.utils import EzPickle
from params import configs
from tool import permissibleLeftShift


class ProgEnv(Constraints, ReadInfo):
    def __init__(self, filename):
        super().__init__(filename)
        self.steps = 0
        self.action_num = sum(self.Activity_mode_Num)
        self.action_space = spaces.Discrete(self.action_num)
        self.price_renewable_resource = 100 * np.ones_like(self.Renewable_resource)
        self.price_nonrenewable_resource = 10 * np.ones_like(self.Nonrenewable_resource)
        self.actSeq = []
        self.modeSeq = []
        self.timeSeq = []
        self.actionSeq = []
        self.crtTime = 0
        self.activities = list(self.SsLimit.keys()) # string
        self.done = False
        self.stateGraph = np.ones((self.action_space.n, self.action_space.n)) * (-np.inf) # non-direction graph
        self.actStatus = np.zeros(self.action_space.n)
        self.candidate = np.arange(self.action_space.n)
        self.mask = np.full(shape=self.action_space.n, fill_value=0, dtype=bool)
        self.penalty_coeff0 = 4000
        self.penalty_coeff1 = 100
        # self.actionPairs = []

    def actionDetermine(self, action):
        # decide the action represents which activity and which mode
        assert action < self.action_num
        temp = 0
        for activity in range(len(self.Activity_mode_Num)): # act starts from 0 mode starts from 1
            for mode in np.arange(1, self.Activity_mode_Num[activity] + 1):
                if action == temp:
                    return int(activity), int(mode)
                temp += 1

    def resetStateGraph(self):
        for activity in self.activities:
            temp = self.SsLimit[activity]
            for key in list(temp.keys()):
                for mode_act in np.arange(1, self.Activity_mode_Num[int(activity)] + 1): # 1, 2, 3 ...
                    for mode_key in np.arange(1, self.Activity_mode_Num[int(key)] + 1):# 1, 2, 3 ...
                        pcol = sum(self.Activity_mode_Num[:(int(activity))]) + mode_act - 1
                        prow = sum(self.Activity_mode_Num[:(int(key))]) + mode_key - 1
                        self.stateGraph[prow, pcol] = temp[key][mode_act-1, mode_key-1] # 1 if temp[key].mean() > 0 else -1
                # self.stateGraph[int(activity), int(activity)] = 0


    def BuildStateGraph(self, action):
        # self.stateGraph[self.crtAct, self.crtAct] = 1  # after the activity is taken, the stateGraph[i,i] = 1
        action_limit = sum(self.Activity_mode_Num[:self.crtAct])+ np.arange(0, self.Activity_mode_Num[self.crtAct])
        # temp = self.SsLimit[str(self.crtAct)]
        for act_col in action_limit:
            if act_col != action:
                self.stateGraph[:, act_col] = - np.inf
                self.stateGraph[act_col] = - np.inf
        return action_limit
        # for key in list(temp.keys()):
        #     if int(key) not in self.actSeq:
        #         for act_col in action_limit:
        #             prow = sum(self.Activity_mode_Num[:(self.crtAct)]) + self.crtMode - 1
        #             self.stateGraph[:, act_col] = - np.inf # temp[key][mode_key, self.crtMode]
        #             self.stateGraph[act_col] = - np.inf
        #     else:
        #         slicing = np.where(int(key) == self.actSeq)[0]
        #         mode = self.modeSeq[slicing]
        #         self.stateGraph[int(key), int(self.crtAct)] = temp[key][int(mode), int(self.crtMode)].item()
        # for act, mode in zip(self.actSeq, self.modeSeq):
        #     temp = self.SsLimit[str(act)]
        #     if str(self.crtAct) in list(temp.keys()):
        #         self.stateGraph[int(self.crtAct), int(act)] = temp[str(self.crtAct)][int(self.crtMode), int(mode)].item()

    def updateActStatus(self, action, action_limit):
        # update the activity status
        duration = self.get_current_duration(self.crtMode, self.crtAct)
        self.actStatus[action] = 1. / duration if duration > 0 else 1  # activity starts then the status changes to 1/duration from 0
        _, endTimeSeq, durationSeq = self.getProjectTime(self.timeSeq[:-1], self.modeSeq[:-1], self.actSeq[:-1])
        pastAction = np.array(self.actSeq[:-1])
        pastTime =  np.array(self.timeSeq[:-1])
        finishactMask = pastAction[endTimeSeq <= self.crtTime]
        progressactMask = pastAction[endTimeSeq > self.crtTime] #actMask[]
        self.actStatus[finishactMask] = 1  # when activity done, then the status changes to 1
        self.actStatus[progressactMask] = (self.crtTime - pastTime[endTimeSeq > self.crtTime]) / durationSeq[endTimeSeq > self.crtTime]
        for act in action_limit:
            if act != action:
                self.actStatus[act] = -1

    def startTimeDetermine(self, graph):
        assert len(self.timeSeq) + 1 == graph.shape[0]
        position = -1 # self.actSeq.index(self.crtAct) # -1
        OutDegree, InDegree = graph[:-1, -1], graph[-1, :-1]
        latestTime = np.array(self.timeSeq) - OutDegree
        earliestTime = np.array(self.timeSeq) + InDegree
        if earliestTime.max() <= latestTime.min():
            startTime = earliestTime.max() + 1 if earliestTime.max() > -1000 else self.crtTime + 1 # greedy choice of start time
            return startTime
        return 0

    def potential(self):
        projectStatus = self.actStatus[self.actStatus != -1].sum()/len(self.activities) * 100 # the percentage that the project is complicated
        return projectStatus


    def step(self, action):
        potential0 = self.potential()
        self.actionSeq.append(action)
        self.crtAct, self.crtMode = self.actionDetermine(action)

        self.actSeq.append(self.crtAct)
        self.modeSeq.append(self.crtMode)
        # have to determine the earliest crtAct start time
        # determine whether first activity is feasible
        if self.steps == 0:
            self.crtTime = 0
            self.timeSeq.append(self.crtTime)
            tempM = np.ones(len(self.activities)) * (-5000)
            for activity in self.activities:
                temp = self.SsLimit[activity]  # dict
                if str(self.crtAct) in list(temp.keys()):
                    tempM[int(activity)] = temp[str(self.crtAct)][:, int(self.crtMode)].min().item()

            if (tempM > 0).any():
                feasible = False
                self.done = True
            else:
                mask_limit = self.BuildStateGraph(action)
                self.mask[mask_limit] = True
        else:
            mask_limit = self.BuildStateGraph(action)
            self.mask[mask_limit] = True
            PastActGraph = self.stateGraph[self.actionSeq][: ,self.actionSeq] # find the past graph for judging the feasibility  and other determination
            feasible = self.JudgeFeasibility(PastActGraph)
            resource1 = self.Is_Renewable_Resource_Feasible(self.crtMode, self.crtAct, self.crtTime)
            resource2 = self.Is_NonRenewable_Resource_Feasible(self.crtMode, self.crtAct)
            feasible = bool(feasible and resource1 and resource2)
            if feasible:
                startTime = self.startTimeDetermine(PastActGraph)
                self.crtTime = startTime
                self.timeSeq.append(self.crtTime)
                self.updateActStatus(action, mask_limit) # update the activity status, for the done activity set to 1, for the on-progress activity, set as (ctrTime - startTime)/duration
            else:
                self.done = True

        # assign reward
        if not self.done:
            # penalty for resource used
            potential1 = self.potential()
            renewR = self.get_current_mode_using_renewable_resource(self.crtMode, self.crtAct)
            NonrenewR = self.get_current_mode_using_nonrenewable_resource(self.crtMode, self.crtAct)
            reward = potential1 - potential0 - np.dot(renewR, self.price_renewable_resource.reshape(-1)) - np.dot(NonrenewR, self.price_nonrenewable_resource.reshape(-1))

        # determine whether all the activity are on-progress and finished or not yet
        if len(self.actSeq) == len(self.activities):
            self.done = True
            T, _, _ = self.getProjectTime(self.timeSeq, self.modeSeq, self.actSeq)
            self.crtTime = T

        if self.done:
            if (self.actStatus == 0).any(): # the project is not finished, but done since infeasibility of the activity or the resource constraints
                reward = - self.penalty_coeff0 * len(self.actStatus[self.actStatus == 0])
            else:
                potential1 = self.potential()
                renewR = self.get_current_mode_using_renewable_resource(self.crtMode, self.crtAct)
                NonrenewR = self.get_current_mode_using_nonrenewable_resource(self.crtMode, self.crtAct)
                reward = potential1 - potential0 - np.dot(renewR, self.price_renewable_resource) - np.dot(NonrenewR,self.price_nonrenewable_resource)
        self.steps += 1
        # self.posRewards +=reward
        return self.stateGraph, self.actStatus, reward, self.done, self.candidate, self.mask


    def reset(self):
        self.actSeq = []
        self.modeSeq = []
        self.timeSeq = []
        self.crtTime = 0
        self.done = False
        self.stateGraph = np.ones((self.action_space.n, self.action_space.n)) * (-np.inf)  # non-direction graph
        self.actStatus = np.zeros(self.action_space.n)
        self.resetStateGraph()
        self.candidate = np.arange(self.action_space.n)
        self.mask = np.full(shape=self.action_space.n, fill_value=0, dtype=bool)
        return self.stateGraph, self.actStatus, self.candidate, self.mask



# if __name__ == '__main__':
#     filepath = r'./test.sch'
#     env = ProgEnv(filepath)
#     s, a, c, m = env.reset()
#     env.step(0)
#     env.step(1)
#     env.step(3)




