#!/bin/python
import tensorflow as tf

class DQNModel:
    def __init__(self, n_actions,
                 n_features,
                 learning_rate=0.01,
                 gamma=0.9,
                 epsilon=0.9,
                 update_freq=300,
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_freq = update_freq

        self.learn_step_cnt = 0
        self.build_model()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def terminal_info_feat(self):
        raise NotImplementedError("Not implemented yet")

    def terminal_action_feat(self):
        raise NotImplementedError("Not implemented yet")

    def gen_info_feat(self, info):
        raise NotImplementedError("Not implemented yet")

    def gen_action_feat(self, action):
        raise NotImplementedError("Not implemented yet")

    def build_model(self):
        raise NotImplementedError("Not implemented yet")

    def update_model(self, experiences):
        raise NotImplemented("Not implemented yet")

    def predict_q(self,info):
        raise NotImplementedError("Not implemented yet")

