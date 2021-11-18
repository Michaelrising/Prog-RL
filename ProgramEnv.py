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
        self.price_renewable_resource = .1 * np.ones_like(self.Renewable_resource)
        self.price_nonrenewable_resource = .2 * np.ones_like(self.Nonrenewable_resource)
        self.actSeq = []
        self.modeSeq = []
        self.timeSeq = []
        self.actionSeq = []
        self.crtTime = 0
        self.activities = list(self.SsLimit.keys()) # string
        self.done = False
        self.stateGraph = np.zeros((self.action_space.n, self.action_space.n))# np.ones((self.action_space.n, self.action_space.n)) * (-np.inf) # non-direction graph
        self.actStatus = np.zeros(self.action_space.n)
        self.timeStatus = np.zeros(self.action_space.n)
        self.candidate = np.arange(self.action_space.n)
        self.pastMask = np.full(shape=self.action_space.n, fill_value=0, dtype=bool)
        self.penalty_coeff0 = 5
        self.penalty_coeff1 = 2
        self.lastTime = 0
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
        # initial graph for the whole state space is len(action) x len(action), so in the test is 13x13.
        # we replace with 1e-8 for 0 weight link, and for -inf we replace with 0
        for activity in self.activities:
            temp = self.SsLimit[activity]
            for key in list(temp.keys()):
                for mode_act in np.arange(1, self.Activity_mode_Num[int(activity)] + 1): # 1, 2, 3 ...
                    for mode_key in np.arange(1, self.Activity_mode_Num[int(key)] + 1):# 1, 2, 3 ...
                        pcol = sum(self.Activity_mode_Num[:(int(activity))]) + mode_act - 1
                        prow = sum(self.Activity_mode_Num[:(int(key))]) + mode_key - 1
                        self.stateGraph[prow, pcol] = temp[key][mode_act-1, mode_key-1] if abs(temp[key][mode_act-1, mode_key-1]) else 1e-8 # if 0 then we set it as 1e-8

    def BuildStateGraph(self, action):
        # after one action is taken, the explicit activity and corresponding mode are determined,
        # then we have to modify te graph for current state: for each mode that will not be taken in the future,
        # replace the whole column and row with 0
        action_limit = sum(self.Activity_mode_Num[:self.crtAct])+ np.arange(0, self.Activity_mode_Num[self.crtAct])
        # temp = self.SsLimit[str(self.crtAct)]
        for act_col in action_limit:
            if act_col != action:
                self.stateGraph[:, act_col] = 0 # - np.inf #### set as -M ####
                self.stateGraph[act_col] = 0 # - np.inf
        return action_limit

    def updateActStatus(self, action, action_limit):
        # update the activity status
        # update the activity status, for the done activity set to 1,
        # for the on-progress activity, set as (ctrTime - startTime)/duration

        duration = self.get_current_duration(self.crtMode, self.crtAct)
        if not self.done:
            self.actStatus[action] = 1. / duration if duration > 0 else 1  # activity starts then the status changes to 1/duration from 0
        else:
            self.actStatus[action] = 0
        _, endTimeSeq, durationSeq = self.getProjectTime(self.timeSeq[:-1], self.modeSeq[:-1], self.actSeq[:-1])
        pastAction = np.array(self.actSeq[:-1], dtype=np.int)
        pastTime =  np.array(self.timeSeq[:-1])
        finishActMask = pastAction[endTimeSeq <= self.crtTime]
        progressActMask = pastAction[endTimeSeq > self.crtTime] #actMask[]
        self.actStatus[finishActMask] = 1  # when activity done, then the status changes to 1
        self.actStatus[progressActMask] = (self.crtTime - pastTime[endTimeSeq > self.crtTime]) / durationSeq[endTimeSeq > self.crtTime]
        for act in action_limit:
            if act != action:
                self.actStatus[act] = -1

    def startTimeDetermine(self, graph): # fix !!!!!!
        assert len(self.timeSeq) + 1 == graph.shape[0]
        position = -1 # self.actSeq.index(self.crtAct) # -1

        if len(self.timeSeq) != 0:
            OutDegree, InDegree = graph[:-1, -1], graph[-1, :-1]
            OutDegree[OutDegree == 0] = -50000
            InDegree[InDegree == 0] = -50000
            latestTime = np.array(self.timeSeq) - OutDegree
            earliestTime = np.array(self.timeSeq) + InDegree
            if int(earliestTime.max()) <= int(latestTime.min()):
                greedy_startTime = int(earliestTime.max()) + 1 if earliestTime.max() > - 1000 else self.crtTime + 1 # greedy choice of start time
                end_startTime = max(latestTime.min(), greedy_startTime)
                return greedy_startTime, end_startTime
            else:
                return max(int(earliestTime.max()), latestTime.min()), max(int(earliestTime.max()), latestTime.min())
        return 0, 0

    def potential(self):
        # the percentage that the project is complicated is set as the potential function
        # which is to decide the reward
        projectStatus = self.actStatus[self.actStatus != -1].sum()/len(self.activities) * 100
        return projectStatus

    def feasibleAction(self):
        feasibleMask = np.full(shape=self.action_space.n, fill_value=0, dtype=bool)
        for ii in range(self.action_space.n):
            tempmask = (self.stateGraph[ii, ~ self.pastMask] > 0).any()
            if tempmask:
                feasibleMask[ii] = True
        return feasibleMask

    def step(self, action):
        startTime = self.crtTime + 1
        potential0 = self.potential()
        self.actionSeq.append(action)
        self.crtAct, self.crtMode = self.actionDetermine(action)
        self.actSeq.append(self.crtAct)
        self.modeSeq.append(self.crtMode)
        mask_limit = self.BuildStateGraph(action)
        self.pastMask[mask_limit] = True
        PastActGraph = self.stateGraph[self.actionSeq][:, self.actionSeq]
        actionFeasible = self.JudgeFeasibility(PastActGraph, self.crtAct, self.crtMode)
        nonRenewFeasible = self.Is_NonRenewable_Resource_Feasible(self.crtMode, self.crtAct)
        if actionFeasible and nonRenewFeasible:
            greedy_startTime, latest_startTime = self.startTimeDetermine(PastActGraph)
            #
            RenewFeasible = self.Is_Renewable_Resource_Feasible(self.modeSeq, self.actSeq, self.timeSeq, greedy_startTime)
            if not RenewFeasible:
                for t in np.arange(greedy_startTime + 1, latest_startTime + 1):
                    if self.Is_Renewable_Resource_Feasible(self.modeSeq, self.actSeq, self.timeSeq, t):
                        startTime = t # greedy choice
                        break
                    if t == latest_startTime and not self.Is_Renewable_Resource_Feasible(self.modeSeq, self.actSeq, self.timeSeq, t):
                        startTime = max(latest_startTime, self.crtTime + 1)
                        self.done = True
            else:
                startTime = max(greedy_startTime, startTime)
        else:
            RenewFeasible = False
            self.done = True
        self.crtTime = startTime
        self.timeSeq.append(self.crtTime)
        self.updateActStatus(action, mask_limit)

        # determine whether all the activity are on-progress and finished or not yet
        if len(self.actSeq) == len(self.activities):
            self.done = True
            T, _, _ = self.getProjectTime(self.timeSeq, self.modeSeq, self.actSeq)
            self.crtTime = T
        # assign reward
        # feasible reward
        f_reward = actionFeasible * 2 + nonRenewFeasible * 1 + RenewFeasible * 1
        reward = f_reward
        potential1 = self.potential()
        renewR = self.get_current_mode_using_renewable_resource(self.crtMode, self.crtAct)
        NonrenewR = self.get_current_mode_using_nonrenewable_resource(self.crtMode, self.crtAct)
        if bool(not self.done) or (self.actStatus != 0).all(): # not done or all activities are finished
            # penalty for resource used
            renew_Penalty = np.dot(renewR, self.price_renewable_resource.reshape(-1))
            nonrenew_Penalty = np.dot(NonrenewR, self.price_nonrenewable_resource.reshape(-1))
            time_penalty = (self.crtTime - self.lastTime)*1
            reward += potential1 - potential0 - time_penalty - renew_Penalty - nonrenew_Penalty
        elif not actionFeasible:
            reward += - self.penalty_coeff0 * len(self.actStatus[self.actStatus == 0])
        else:
            reward += - self.penalty_coeff1 * len(self.actStatus[self.actStatus == 0])

        self.steps += 1
        feasibleMask = self.feasibleAction()
        mask = feasibleMask + self.pastMask
        self.timeStatus[action] = self.crtTime
        self.timeStatus[mask_limit] = -1
        self.lastTime  = self.crtTime
        fea = np.concatenate((self.actStatus, self.timeStatus)).reshape(-1,2)
        return self.stateGraph, fea, reward, self.done, self.candidate, mask, self.crtTime

    def reset(self):
        self.resetResource()
        self.steps = 0
        self.actSeq = []
        self.modeSeq = []
        self.timeSeq = []
        self.actionSeq = []
        self.crtTime = 0
        self.lastTime = 0
        self.done = False
        self.stateGraph = np.zeros((self.action_space.n, self.action_space.n)) #np.ones((self.action_space.n, self.action_space.n)) * (-np.inf)  # non-direction graph
        self.actStatus = np.zeros(self.action_space.n)
        self.timeStatus = np.zeros(self.action_space.n)
        self.resetStateGraph()
        self.pastMask = np.full(shape=self.action_space.n, fill_value=0, dtype=bool)
        feasibleMask = self.feasibleAction()
        mask = feasibleMask + self.pastMask
        self.candidate = np.arange(self.action_space.n)
        fea = np.concatenate((self.actStatus, self.timeStatus)).reshape(-1, 2)
        return self.stateGraph, fea, self.candidate, mask