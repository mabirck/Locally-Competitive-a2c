import json, glob, ast
import matplotlib.pyplot as plt
import numpy as np

def compatibleMatrix(data):

    minor = min([len(d) for d in data])
    print "This is the min", minor
    reshaped_data = np.array([d[:minor] for d in data])
    return reshaped_data
def workersMeanReward(path):
    total_rewards = list()
    files = glob.glob(path+'*')

    for f in files:
        print f
        total_rewards.append(getData(f))

    print "LENGTH ->>>>>>>>>", len(total_rewards)
    total_rewards = compatibleMatrix(total_rewards)
    total_rewards = np.array(total_rewards)
    total_rewards = np.average(total_rewards, axis=0)
    print total_rewards
    return total_rewards


def plotData(data, paths):

    for d in data:
        plt.plot(d)
    plt.legend(paths, loc='best')
    plt.show()


def getData(path):
    with open(path, 'r') as f:
        #print f
        #data = json.load(f)
        rewards = list()
        for line in f:
            D = ast.literal_eval(line)
            if D.has_key('r'):
                rewards.append(D['r'])
                #print type(D['r'])
        return np.array(rewards)

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_dir', help='Log dir to plot', type=str, default='Breakout')
    parser.add_argument('--env', help='environments ID to plot', type=str, default='NoFrameskip-v4')
    args = parser.parse_args()

    paths = glob.glob("../log/{}/*/{}/".format(args.log_dir, args.log_dir+args.env))
    print paths
    data = list()
    for game in paths:
        print(game)
        data.append(workersMeanReward(game))

    plotData(data, paths)

if __name__ == '__main__':
    main()
