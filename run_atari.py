#!/usr/bin/env python
import os, logging, gym
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from a2c import learn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind
from policies import CnnPolicy, LstmPolicy, LnLstmPolicy




def train(args, env_id, num_frames, seed, policy, lrschedule, num_cpu):
    num_timesteps = int(num_frames / 4 * 1.1)
    if len(env_id) > 1:
        num_timesteps = int(num_timesteps * 2)
    DM_STYLE = "NoFrameskip-v4"
    # divide by 4 due to frameskip, then do a little extras so episodes end
    def make_env(rank):
        def _thunk():
            MT = ""
            if len(env_id) > 1:
                MT = "{}_{}_{}/".format(env_id[0][:4], env_id[1][:4], args.act_func)
            else:
                MT = "{}_{}/".format(env_id[0]+DM_STYLE, args.act_func)

            ### CREATING MULTIPLE GAMES ENVS ###
            if(rank < num_cpu//2 or len(env_id) == 1): # <<--- CHECKING WHETHER IS MULTITASK
                PATH = './{}/{}{}'.format(args.log_dir,MT, env_id[0]+DM_STYLE)
                env = gym.make(env_id[0]+DM_STYLE)
            else:
                PATH = './{}/{}{}'.format(args.log_dir,MT,env_id[1]+DM_STYLE)
                env = gym.make(env_id[1]+DM_STYLE)

            env.seed(seed + rank)
            if not(os.path.exists(PATH)):
                os.makedirs(PATH)

            env = bench.Monitor(env, "{}/{}.monitor.json".format(PATH,rank))
            gym.logger.setLevel(logging.WARN)
            return wrap_deepmind(env)
        return _thunk

    if len(env_id)   > 1:
        num_cpu*=2
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    learn(policy_fn, env, seed, args.act_func, args.dropout, total_timesteps=num_timesteps, lrschedule=lrschedule)
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #######REMEMBER COOL THING TO MAKE ARGPARSE RECEIVE LIST OF STRINGS####################################
    parser.add_argument('--env', help='environments ID', nargs='*', default=['Breakout'])##
    ####################################################################################################
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='lstm')
    parser.add_argument('--act_func', help='Activation Function', choices=['maxout', 'relu', 'lwta'], default='lwta')
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
