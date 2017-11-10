#!/usr/bin/env python
import os, logging, gym
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from a2c import learn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind
from policies import CnnPolicy, LstmPolicy, LnLstmPolicy
import tensorflow as tf



def train(args, env_id, num_frames, seed, policy, lrschedule, num_cpu):
    num_timesteps = int(num_frames / 4 * 1.1)
    if len(env_id) > 1:
        num_timesteps = int(num_timesteps * len(env_id))
    DM_STYLE = "NoFrameskip-v4"
    # divide by 4 due to frameskip, then do a little extras so episodes end
    def make_env(rank):
        def _thunk():
            #########################################################
            ########### HARDCODED TO BE 8 CPUS #####################
            #######################################################
            index = rank//8
            MT = ""
            if len(env_id) > 1:
                short_names = [e[:4] for e in env_id]
                short = "_".join(short_names)
                MT = "{}_{}/".format(short+'_'+args.head, args.act_func+'_'+str(args.seed))
            else:
                MT = "{}_{}/".format(env_id[index]+DM_STYLE+'_'+args.head, args.act_func+'_'+str(args.seed))

            ### CREATING MULTIPLE GAMES ENVS ###
            if(len(env_id) == 1): # <<--- CHECKING WHETHER IS MULTITASK
                PATH = './{}/{}{}'.format(args.log_dir,MT, env_id[index]+DM_STYLE)
                env = gym.make(env_id[index]+DM_STYLE)
            else:
                #print(index)
                PATH = './{}/{}{}'.format(args.log_dir,MT,env_id[index]+DM_STYLE)
                env = gym.make(env_id[index]+DM_STYLE)

            env.seed(seed + rank)
            if not(os.path.exists(PATH)):
                os.makedirs(PATH)

            env = bench.Monitor(env, "{}/{}.monitor.json".format(PATH,rank))
            gym.logger.setLevel(logging.WARN)
            return wrap_deepmind(env)
        return _thunk

    if len(env_id) > 1:
        num_cpu*=len(env_id)
    set_global_seeds(seed)

    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    actionMasks = []
    heads=list()
    #NECESSARY PROCEDURE TO GET ACTION MASKS
    for e in range(num_cpu):
        #print(e, num_cpu)
        env.remotes[e].send(('get_spaces', None))
        actionMasks.append(env.remotes[e].recv()[0].n)
        heads.append(e // 8)
        #print(e//8)
    heads = set(heads)
    heads = list(heads)


    print(actionMasks)
    print(heads)

    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy

    learn(policy_fn, env, actionMasks, heads, seed, args.act_func, args.dropout, args.head, total_timesteps=num_timesteps, lrschedule=lrschedule)
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #######REMEMBER COOL THING TO MAKE ARGPARSE RECEIVE LIST OF STRINGS####################################
    parser.add_argument('--env', help='environments ID', nargs='*', default=['Breakout'])##
    ####################################################################################################
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--head', help='OUTPUT STYLE', choices=['multi', 'single'], default='single')
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='lstm')
    parser.add_argument('--act_func', help='Activation Function', choices=['maxout', 'relu', 'lwta', 'maxout_lwta', 'fully_maxout'], default='relu')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--million_frames', help='How many frames to train (/ 1e6). '
        'This number gets divided by 4 due to frameskip', type=int, default=100)
    parser.add_argument('--log_dir', help='Log dir', type=str, default='log')
    parser.add_argument('--exp', help='Exploration Strategies', choices=['ent', 'thompson'], default='ent')
    parser.add_argument('--dropout', help='Exploration Strategies', type=float, default=0.1)

    args = parser.parse_args()
    train(args, args.env, num_frames=1e6 * args.million_frames, seed=args.seed,
        policy=args.policy, lrschedule=args.lrschedule, num_cpu=8)

if __name__ == '__main__':
    main()
