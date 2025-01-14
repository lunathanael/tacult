import logging
import math

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)
import torch

from .utils import to_obs

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            
            obs = to_obs(canonicalBoard)
            obs = np.expand_dims(obs, axis=0)
            pred = self.nnet.predict(obs)
            self.Ps[s], v = pred[0][0], pred[1][0]

            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v


class VectorizedMCTS():
    """
    This class handles the Vectorized MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = [{} for _ in range(args.numEnvs)]  # stores Q values for s,a (as defined in the paper)
        self.Nsa = [{} for _ in range(args.numEnvs)]  # stores #times edge s,a was visited
        self.Ns = [{} for _ in range(args.numEnvs)]  # stores #times board s was visited
        self.Ps = [{} for _ in range(args.numEnvs)]  # stores initial policy (returned by neural net)

        self.Es = [{} for _ in range(args.numEnvs)]  # stores game.getGameEnded ended for board s
        self.Vs = [{} for _ in range(args.numEnvs)]  # stores game.getValidMoves for board s

    def reset(self, i):
        self.Qsa[i] = {}
        self.Nsa[i] = {}
        self.Ns[i] = {}
        self.Ps[i] = {}

        self.Es[i] = {}
        self.Vs[i] = {}

    def getActionProbs(self, canonicalBoards, temps):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard for each environment.

        Returns:
            probs: for each environment, a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoards)

        _s = [
            self.game.stringRepresentation(canonicalBoard)
            for canonicalBoard in canonicalBoards
        ]
        _counts = [
            [
                self.Nsa[i].get((s, a), 0)
                for a in range(self.game.getActionSize())
            ]
            for i, s in enumerate(_s)
        ]

        _counts = np.array(_counts)
        _probs = np.zeros((self.args.numEnvs, self.game.getActionSize()))

        for i in range(self.args.numEnvs):
            counts = _counts[i]
            temp = temps[i]

            if temp == 0:
                bestAs = np.array(np.argwhere(counts == np.max(counts))).ravel()
                bestA = np.random.choice(bestAs)
                _probs[i][bestA] = 1
            else:
                counts = counts ** (1. / temp)
                counts_sum = float(np.sum(counts))
                _probs[i] = counts / counts_sum
        return _probs


    def search(self, canonicalBoards):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        terminalMask = np.zeros(self.args.numEnvs, dtype=bool)
        for i in range(self.args.numEnvs):
            if canonicalBoards[i] is None:
                terminalMask[i] = True
        return self._search(canonicalBoards, terminalMask)

    
    def _search(self, canonicalBoards, terminalMask):
        _v = np.zeros(self.args.numEnvs)
        _s = [
            self.game.stringRepresentation(canonicalBoard) if not terminalMask[i] else None
            for i, canonicalBoard in enumerate(canonicalBoards)
        ]

        for i, s in enumerate(_s):
            if terminalMask[i]:
                continue
            if s not in self.Es[i]:
                self.Es[i][s] = self.game.getGameEnded(canonicalBoards[i], 1)
            if self.Es[i][s] != 0:
                terminalMask[i] = True
                _s[i] = None
                _v[i] = self.Es[i][s]

        if np.all(terminalMask):
            return -_v
        
        _pis, _vs = self.maskedBatchPrediction(canonicalBoards, terminalMask)

        for i, s in enumerate(_s):
            if terminalMask[i]:
                continue
            if s not in self.Ps[i]:
                # leaf node
                self.Ps[i][s], v = _pis[i], _vs[i]
                valids = self.game.getValidMoves(canonicalBoards[i], 1)
                self.Ps[i][s] = self.Ps[i][s] * valids  # masking invalid moves
                sum_Ps_s = np.sum(self.Ps[i][s])
                if sum_Ps_s > 0:
                    self.Ps[i][s] /= sum_Ps_s  # renormalize
                else:
                    # if all valid moves were masked make all valid moves equally probable

                    # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                    # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                    log.error("All valid moves were masked, doing a workaround.")
                    self.Ps[i][s] = self.Ps[i][s] + valids
                    self.Ps[i][s] /= np.sum(self.Ps[i][s])

                self.Vs[i][s] = valids
                self.Ns[i][s] = 0

                terminalMask[i] = True
                _s[i] = None
                _v[i] = v

        if np.all(terminalMask):
            return -_v
        
        _next_s = [None] * self.args.numEnvs
        _a = [None] * self.args.numEnvs

        for i, s in enumerate(_s):
            if terminalMask[i]:
                continue
            valids = self.Vs[i][s]
            cur_best = -float('inf')
            best_act = -1

            # pick the action with the highest upper confidence bound
            for a in range(self.game.getActionSize()):
                if valids[a]:
                    if (s, a) in self.Qsa[i]:
                        u = self.Qsa[i][(s, a)] + self.args.cpuct * self.Ps[i][s][a] * math.sqrt(self.Ns[i][s]) / (
                                1 + self.Nsa[i][(s, a)])
                    else:
                        u = self.args.cpuct * self.Ps[i][s][a] * math.sqrt(self.Ns[i][s] + EPS)  # Q = 0 ?

                    if u > cur_best:
                        cur_best = u
                        best_act = a

            a = best_act
            next_s, next_player = self.game.getNextState(canonicalBoards[i], 1, a)
            next_s = self.game.getCanonicalForm(next_s, next_player)

            _next_s[i] = next_s
            _a[i] = a

        _v += self._search(_next_s, terminalMask)

        for i, s in enumerate(_s):
            if s is None:
                continue
            v = _v[i]
            a = _a[i]
            if (s, a) in self.Qsa[i]:
                self.Qsa[i][(s, a)] = (self.Nsa[i][(s, a)] * self.Qsa[i][(s, a)] + v) / (self.Nsa[i][(s, a)] + 1)
                self.Nsa[i][(s, a)] += 1

            else:
                self.Qsa[i][(s, a)] = v
                self.Nsa[i][(s, a)] = 1

            self.Ns[i][s] += 1
            terminalMask[i] = True
        return -_v
    
    def maskedBatchPrediction(self, canonicalBoards, terminalMask):
        obs = np.array([
            to_obs(canonicalBoard) if (canonicalBoard is not None) else torch.zeros(2, 9, 9, dtype=torch.float32)
            for canonicalBoard in canonicalBoards
        ])
        active_obs = obs[~terminalMask]
        active_pi, active_v = self.nnet.predict(active_obs)

        full_pi = np.zeros((self.args.numEnvs, self.game.getActionSize()))
        full_pi[~terminalMask] = active_pi
        full_v = np.zeros(self.args.numEnvs)
        full_v[~terminalMask] = active_v.ravel()
        return full_pi, full_v


