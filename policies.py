import numpy as np
import tensorflow as tf
from utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, sample, check_shape, multi_lstm
from baselines.common.distributions import make_pdtype
import baselines.common.tf_util as U
import gym
from  tfmaxout.maxout import max_out
from tfmaxout.lwta import lwta


class LnLstmPolicy(object):
    def __init__(self, sess,act_f, ob_space, ac_space, nenv, nsteps, nstack, nlstm=256, reuse=False):

        ################################################################
        C = 1
        if act_f == "relu":
            act_f = tf.nn.relu()
        elif act_f =="maxout":
            act_f = max_out()
            # Constant which multiplies model by 2 in case of maxout
            C = 2
        ################################################################

        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32*C, rf=8, stride=4, act=act_f, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64*C, rf=4, stride=2, act=act_f, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, act=act_f, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, act=act_f, init_scale=np.sqrt(2))
            xs = batch_to_seq(h4, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', act=lambda x:x)
            vf = fc(h5, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            a, v, s = sess.run([a0, v0, snew], {X:ob, S:state, M:mask})
            return a, v, s

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class LstmPolicy(object):
    def __init__(self, sess, act_f, ob_space, ac_space, actionMasks, heads, multi_contr, nenv, nsteps, nstack, nlstm=256, reuse=False):

        ## PREPARING INDEX THINGS #########################################
        lstm_heads = [[index]*8 for index in heads]
        lstm_heads = [item for sublist in lstm_heads for item in sublist]

        index = [[index]*8*nsteps for index in heads]
        index = [item for sublist in index for item in sublist]
        index = [ [k, i] for k, i in enumerate(index)]
        ################################################################
        C = 1
        F = 1
        if act_f == "relu":
            act_conv = tf.nn.relu
            act_f = tf.nn.relu
        elif act_f =="maxout":
            act_conv = max_out
            act_f = tf.nn.relu
            # Constant which multiplies model by 2 in case of maxout
            C = 2
        elif act_f == "lwta":
            act_conv = tf.nn.relu
            act_f = lwta
        elif act_f == "maxout_lwta":
            act_conv = max_out
            act_f = lwta
            C = 2
        elif act_f == "fully_maxout":
            act_conv = max_out
            act_f = max_out
            C = 2
            F = 2
        ################################################################
        keep_prob = tf.placeholder(tf.float32)
        ################################################################


        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        # USING 18 WHICH IS THE MAX
        #nact = ac_space.n
        nact = 18
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        #print(nenv//8, "SHIIIIT")
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        if(multi_contr == "multi"):
            S = tf.placeholder(tf.float32, [nenv, nlstm*2 * nenv//8]) #states
        else:
            S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states

        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32*C, rf=8, stride=4, act=act_conv, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64*C, rf=4, stride=2, act=act_conv, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64*C, rf=3, stride=1, act=act_conv, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512*F, act = act_f, init_scale=np.sqrt(2))
            xs = batch_to_seq(h4, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)

            if multi_contr == "single":
                print("LETS BUILD A SINGLE HEAD MODEL...")
                #### TAKE CARE SINGLE LSTM ####################
                h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
                h5 = seq_to_batch(h5)
                #############################################
                pi = fc(h5, 'pi', nact, act=lambda x:x)
                vf = fc(h5, 'v', 1, act=lambda x:x)
            elif multi_contr == "multi":
                print("LETS BUILD A MULTI HEAD MODEL...")

                #### TAKE CARE MUITPLE LSTM (TREATING DIFFERENTE CONTEXT VECTORS AND HIDDEN)
                h5, snew = multi_lstm(xs, ms, S, 'lstm1', nh=nlstm, nenvs = nenv//8, index=lstm_heads, nsteps=nsteps)

                h5 = seq_to_batch(h5)

                pi = fc(h5, 'pi', nenv//8 * nact, act=lambda x:x)
                pi = tf.reshape(pi, (nenv*nsteps, nenv//8, 18), 'pi_rs')

                vf = fc(h5, 'v', nenv//8 * 1, act=lambda x:x)
                vf = tf.reshape(vf, [nenv*nsteps, nenv//8, 1], 'vf_rs')



                pi = tf.gather_nd(pi, index, name='pi_gather')
                vf = tf.gather_nd(vf, index, name='vf_gather')

            print("FINAL PI and FINAL", pi, vf)


        print("This is my pi", pi)
        print("This is my vf", vf)




        mask = [[item]*nsteps for item in actionMasks]
        mask = [item for sublist in mask for item in sublist]

        v0 = vf[:,0]
        a0 = sample(pi, mask)
        self.initial_state = np.zeros((nenv, nlstm*2*nenv//8), dtype=np.float32)

        def step(ob, state, mask, drop):
            a, v, s = sess.run([a0, v0, snew], {X:ob, S:state, M:mask, keep_prob:drop})
            return a, v, s

        def value(ob, state, mask, drop):
            return sess.run(v0, {X:ob, S:state, M:mask, keep_prob:drop})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.keep_prob = keep_prob
        self.nenv = nenv//8

class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            #h4_mx = max_out(h4, num_units=256)
            #pi = fc(h4_mx, 'pi', nact, act=lambda x:x)
            pi = fc(h4, 'pi', nact, act=lambda x:x)
            vf = fc(h4, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = [] #not stateful

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X:ob})
            return a, v, [] #dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
