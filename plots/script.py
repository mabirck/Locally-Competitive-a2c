import json, glob, ast
import matplotlib.pyplot as plt
import numpy as np
import pandas
from matplotlib.backends.backend_pdf import PdfPages

ewma = pandas.stats.moments.ewma

def compatibleMatrix(data):

    minor = [len(d) for d in data]
    #print minor
    minor = min(minor)
    #print "This is the min", minor
    reshaped_data = np.array([d[:minor] for d in data])
    return reshaped_data
def workersMeanReward(path, game):
    total_rewards = list()
    files = glob.glob(path+'*/'+game+'/*')

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
    return [name.split('/')[-2] for name in paths]

def getMin(data):
    mini = [x[-1] for x in data]
    print mini
    return min(mini)

def plotData(data, length, paths):
    mini = getMin(length)
    print paths
    name = paths[0].split('/')[-1]
    paths = fix_name(paths)
    fig = plt.figure()

    for d, l in zip(data, length):
        plt.plot(l ,ewma(d, 50))

    plt.legend(paths, loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.grid()
    plt.xlim(0.0,mini)
    print name
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

    paths = glob.glob("../log/{}/*/".format(args.log_dir))
    #print paths
    data = list()
    length = list()

    for game in paths:
        #print(game)
        data.append(workersMeanReward(game, args.log_dir+args.env))
        length.append(workersLength(game, args.log_dir+args.env))
    #print length
    plotData(data, length, paths)

if __name__ == '__main__':
    main()
