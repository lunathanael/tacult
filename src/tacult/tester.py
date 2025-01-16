import logging

import coloredlogs

from tacult.utac_game import UtacGame as Game
from tacult.utils import dotdict
from tacult.base.coach import Coach
from tacult.network import UtacNNet
from tacult.utac_nn import UtacNN
from tacult.base import NNetWrapper
from tacult.base.mcts import MCTS, RawMCTS
from tacult.base.arena import Arena
import numpy as np
import torch
log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

_args = dotdict({
    'numMCTSSims': 2,          # Number of games moves for MCTS to simulate.
    'cpuct': 1,
    'cuda': False,
    'model_file': ('./temp/run1','best.pt'),
    'ignore_optimizer': True,
    'numRollouts': 50,

    'lr': 0.001, # redundancy
    'steps_per_epoch': 10,
    'epochs': 10,
    'batch_size': 512,
    'numIters': 10,
    'num_games': 10,
    'verbose': False,

    'lr': 0.001,
    'dropout': 0.3,
    'cuda': False,
    'maxlenOfQueue': 200000,
    'numItersForTrainExamplesHistory': 20,
    'numEnvs': 1,
    'minNumEps': 1,
    'tempThreshold': 100,
})


def main(args=_args):
    log.info('Loading %s...', Game.__name__)
    g = Game()

    # mcts = RawMCTS(g, args)
    # act = mcts.getActionProb(g.getInitBoard(), temp=1)
    # print(act)
    # act = np.array(act).reshape(9, 9)
    # print(act.round(2))
    # print(mcts.getCurrentEvaluation(g.getInitBoard()))
    log.info('Loading %s...', UtacNNet.__name__)
    nnet = NNetWrapper(UtacNNet(args.cuda, onnx_export=True), args)

    # log.info('Loading checkpoint "%s/%s"...', args.model_file[0], args.model_file[1])
    # nnet.load_checkpoint(args.model_file[0], args.model_file[1])

    board = g.getInitBoard()

    current_player = 1
    for i in range(30000):
        moves = g.getValidMoves(board, current_player)
        action = np.random.choice(np.where(moves)[0])
        board, current_player = g.getNextState(board, current_player, action)
        if g.getGameEnded(board, current_player) != 0:
            board = g.getInitBoard()
            current_player = 1

    board.print()

    pi = np.zeros(9*9)
    pi[action] = 1

    sym = g.getSymmetries(board, pi)
    for b, p in sym:
        b.print()
        print(p.reshape(9, 9))
    
    print(g._get_obs(board))
    return
    # nnet = UtacNN(args)

    # coach = Coach(g, nnet, args)
    # q = coach.generateTrainExamples()
    # print([x[2] for x in q])
    
    # canonical_board = g.getCanonicalForm(board, 1)
    # obs = g._get_obs(canonical_board).unsqueeze(0)
    _o = {"0":0,"1":0,"2":0,"3":0,"4":0,"5":0,"6":0,"7":0,"8":0,"9":0,"10":1,"11":0,"12":0,"13":0,"14":0,"15":0,"16":1,"17":0,"18":0,"19":0,"20":1,"21":0,"22":-1,"23":0,"24":0,"25":-1,"26":0,"27":0,"28":0,"29":0,"30":-1,"31":0,"32":-1,"33":0,"34":0,"35":0,"36":0,"37":-1,"38":0,"39":0,"40":1,"41":0,"42":0,"43":1,"44":0,"45":0,"46":0,"47":-1,"48":-1,"49":-1,"50":-1,"51":-1,"52":1,"53":0,"54":-1,"55":0,"56":0,"57":0,"58":0,"59":1,"60":0,"61":1,"62":0,"63":0,"64":1,"65":-1,"66":0,"67":1,"68":0,"69":1,"70":1,"71":1,"72":1,"73":-1,"74":0,"75":1,"76":0,"77":0,"78":0,"79":-1,"80":-1,"81":0,"82":0,"83":0,"84":0,"85":0,"86":0,"87":0,"88":0,"89":0,"90":0,"91":0,"92":0,"93":0,"94":0,"95":0,"96":0,"97":0,"98":0,"99":0,"100":0,"101":0,"102":0,"103":0,"104":0,"105":0,"106":0,"107":0,"108":0,"109":0,"110":0,"111":1,"112":1,"113":1,"114":0,"115":0,"116":0,"117":0,"118":0,"119":0,"120":1,"121":1,"122":1,"123":0,"124":0,"125":0,"126":0,"127":0,"128":0,"129":1,"130":1,"131":1,"132":0,"133":0,"134":0,"135":0,"136":0,"137":0,"138":1,"139":1,"140":1,"141":1,"142":1,"143":1,"144":0,"145":0,"146":0,"147":1,"148":1,"149":1,"150":1,"151":1,"152":1,"153":0,"154":0,"155":0,"156":1,"157":1,"158":1,"159":1,"160":1,"161":1,"162":0,"163":0,"164":0,"165":0,"166":0,"167":0,"168":0,"169":0,"170":0,"171":0,"172":0,"173":0,"174":0,"175":0,"176":0,"177":0,"178":0,"179":0,"180":0,"181":0,"182":0,"183":0,"184":0,"185":0,"186":0,"187":0,"188":0,"189":0,"190":0,"191":0,"192":-1,"193":-1,"194":-1,"195":0,"196":0,"197":0,"198":0,"199":0,"200":0,"201":-1,"202":-1,"203":-1,"204":0,"205":0,"206":0,"207":0,"208":0,"209":0,"210":-1,"211":-1,"212":-1,"213":0,"214":0,"215":0,"216":0,"217":0,"218":0,"219":1,"220":1,"221":1,"222":1,"223":1,"224":1,"225":0,"226":0,"227":0,"228":1,"229":1,"230":1,"231":1,"232":1,"233":1,"234":0,"235":0,"236":0,"237":1,"238":1,"239":1,"240":1,"241":1,"242":1,"243":1,"244":1,"245":1,"246":1,"247":1,"248":1,"249":1,"250":1,"251":1,"252":1,"253":0,"254":1,"255":1,"256":1,"257":1,"258":1,"259":0,"260":1,"261":1,"262":1,"263":0,"264":1,"265":0,"266":1,"267":1,"268":0,"269":1,"270":1,"271":1,"272":1,"273":0,"274":0,"275":0,"276":1,"277":1,"278":1,"279":1,"280":0,"281":1,"282":0,"283":0,"284":0,"285":1,"286":0,"287":1,"288":1,"289":1,"290":0,"291":0,"292":0,"293":0,"294":0,"295":0,"296":1,"297":0,"298":1,"299":1,"300":0,"301":0,"302":0,"303":0,"304":0,"305":0,"306":1,"307":0,"308":0,"309":0,"310":0,"311":0,"312":0,"313":0,"314":0,"315":0,"316":0,"317":1,"318":0,"319":0,"320":0,"321":0,"322":0,"323":0}

    obs = list(_o.values())
    obs = torch.tensor(obs).reshape(1, 4, 9, 9).float()

    print(obs)
    # obs[:,0,:,:] *= -1
    # obs[:,2,:,:] *= -1
    # print(obs)

    pred = nnet.predict(obs)
    print(pred)
    pi, v = pred
    print(pi.reshape(1, 9, 9).round(2))

    mask = obs[:,-1,:,:].flatten()
    print(mask.reshape(1, 9, 9))

    pi = pi * mask.numpy()
    print(pi.reshape(1, 9, 9).round(2))

    sum_pi = np.sum(pi, keepdims=True)
    pi = pi / sum_pi
    print(pi.reshape(1, 9, 9).round(2))
    print(v)
    # mcts = MCTS(g, nnet, args)
    # raw_mcts = RawMCTS(g, args)

    # arena = Arena(lambda x: np.argmax(mcts.getActionProb(x, temp=0)), lambda x: np.argmax(raw_mcts.getActionProb(x, temp=0)), g)
    # a = arena.playGames(args.num_games, verbose=False)
    # print(a)

def test(args=_args):
    main(args)


if __name__ == "__main__":
    main()