import json, glob, ast
import matplotlib.pyplot as plt
import numpy as np
import pandas
from matplotlib.backends.backend_pdf import PdfPages

ewma = pandas.stats.moments.ewma

def compatibleMatrix(data):

    minor = [len(d) for d in data]
    #print data
    minor = min(minor)
    #print "This is the min", minor
    reshaped_data = np.array([d[:minor] for d in data])
    return reshaped_data
def workersMeanReward(path, game):
    total_rewards = list()
    files = glob.glob(path+'*/'+game+'/*')
    #print "OOHH SHIT THIS IS HERE", files
    for f in files:
        print f
        #print f
        total_rewards.append(getData(f, 'r'))

    #print "LENGTH ->>>>>>>>>", len(total_rewards)
    total_rewards = compatibleMatrix(total_rewards)
    total_rewards = np.array(total_rewards)
    total_rewards = np.average(total_rewards, axis=0)
    #print total_rewards
    return total_rewards

def workersLength(path, game):
    total_rewards = list()
    files = glob.glob(path+'*/'+game+'/*')

    for f in files:
        #print f
        total_rewards.append(np.cumsum(getData(f, 'l')))
        #print total_rewards

    #print "LENGTH ->>>>>>>>>", len(total_rewards)
    total_rewards = compatibleMatrix(total_rewards)
    total_rewards = np.array(total_rewards)
    total_rewards = np.average(total_rewards, axis=0)
    #print total_rewards
    return total_rewards

def fix_name(paths):
    return [(name.split('/')[-2]) for name in paths]

def getMin(data):
    mini = [x[-1] for x in data]
    #print mini
    return min(mini)

def plotData(data, length, paths, lines, colors):
    mini = getMin(length)
    #print paths
    name = paths[0].split('/')[-4]+'_from_'+paths[0].split('/')[-3]
    paths = fix_name(paths)
    fig = plt.figure()

    for k, (d, l) in enumerate(zip(data, length)):
        plt.plot(l ,ewma(d, 20), linestyle=lines[k], color=colors[k])

    plt.legend(paths, loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.grid()
    plt.xlim(0.0,mini)
    #print name,"NAMEEEE MOTHE FUCKER"
    fig.savefig(name+'.pdf')
    plt.show()



def getData(path, key):
    with open(path, 'r') as f:
        #print f
        #data = json.load(f)
        total = 0
        rewards = list()
        for line in f:
            D = ast.literal_eval(line)
            if D.has_key(key):
                total+=D['l']
                rewards.append(D[key])
                #print type(D['r'])
        #print total
        return np.array(rewards)

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_dir', help='Log dir to plot', type=str, default='Breakout')
    parser.add_argument('--env', help='environments ID to plot', type=str, default='NoFrameskip-v4')
    args = parser.parse_args()


    # MAXOUT
    #all_games =  ["Asterix", "Asterix", "BeamRider", "BeamRider", \
    #             "DemonAttack", "DemonAttack", "Enduro", "Enduro", \
    #             "Pong", "Pong", "SpaceInvaders", "SpaceInvaders"
    #]
    #all_paths = ["Asterix/aste_beam_maxout", "Asterix/aste_endu_maxout", "BeamRider/aste_beam_maxout", "BeamRider/beam_endu_maxout", \
    #             "DemonAttack/demo_pong_maxout", "DemonAttack/demo_space_maxout", "Enduro/aste_endu_maxout", "Enduro/beam_endu_maxout", \
    #             "Pong/demo_pong_maxout", "Pong/space_pong_maxout", "SpaceInvaders/demo_space_maxout", "SpaceInvaders/space_pong_maxout"
    #]

    # LWTA
    #all_games =  ["DemonAttack", "DemonAttack", "Pong", "Pong", "SpaceInvaders", "SpaceInvaders"]
    #all_paths = [ "DemonAttack/demo_pong_lwta", "DemonAttack/demo_space_lwta",
    #             "Pong/demo_pong_lwta", "Pong/space_pong_lwta", "SpaceInvaders/demo_space_lwta", "SpaceInvaders/space_pong_lwta"
    #]

    # MIXED
    all_games =  ["DemonAttack", "DemonAttack", "Pong", "Pong", "SpaceInvaders", "SpaceInvaders"]
    all_paths = [ "DemonAttack/demo_pong_mix", "DemonAttack/demo_space_mix", "Pong/demo_pong_mix", \
                  "Pong/space_pong_mix", "SpaceInvaders/demo_space_mix", "SpaceInvaders/space_pong_mix"]
    colors = ["black", "gray", "blue", "red"]
    lines = [':', '-.', '--', '-']
    for current, game2plot in zip(all_paths, all_games):
        paths = glob.glob("../log/{}/*/".format(current))
        #print paths
        data = list()
        length = list()
        log_dir = args.log_dir.split('/')[0]
        for game in paths:
            #print("SHIIIIT", game)
            data.append(workersMeanReward(game, game2plot+args.env))
            #print "This is the data", data
            length.append(workersLength(game, game2plot+args.env))
        #print length
        plotData(data, length, paths, lines, colors)

if __name__ == '__main__':
    main()
