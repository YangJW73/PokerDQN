#!/bin/python
import algorithm

import numpy as np
import tensorflow as tf

class SevenKingModel(algorithm.DQNModel):
    def terminal_info_feat(self):
        raise NotImplementedError("Not implemented yet")

    def terminal_action_feat(self):
        raise NotImplementedError("Not implemented yet")

    def gen_info_feat(self, info):
        raise NotImplementedError("Not implemented yet")

    def gen_action_feat(self, action):
        raise NotImplementedError("Not implemented yet")

    def build_model(self):
        self.state = tf.placeholder(tf.float32, [None, self.n_features], name='state')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')
        # self.reward = tf.placeholder(tf.float32, [None, ], name='reward')
        # self.action = tf.placeholder(tf.int32, [None, ], name='action')

        w_init, b_init = tf.random_normal_initializer(), tf.constant_initializer()

        # build evaluate net
        with tf.variable_scope('eval_net'):
            e_layer_1 = tf.layers.dense(self.state, 20, tf.nn.relu, kernel_initializer=w_init,
                                        bias_initializer=b_init, name='e_layer_1')
            self.q_eval = tf.layers.dense(e_layer_1, self.n_actions, kernel_initializer=w_init,
                                          bias_initializer=b_init, name='q_eval')

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        # build target net
        self.next_state = tf.placeholder(tf.float32, [None, self.n_features], name='next_state')

        with tf.variable_scope('target_net'):
            t_layer_1 = tf.layers.dense(self.next_state, 20, tf.nn.relu, kernel_initializer=w_init,
                                        bias_initializer=b_init, name='t_layer_1')
            self.q_next = tf.layers.dense(t_layer_1, self.n_actions, kernel_initializer=w_init,
                                          bias_initializer=b_init, name='q_next')

    def update_model(self, experiences):
        if self.learn_step_cnt % self.update_freq == 0:
            self.sess.run(self.target_replace_op)
            print('update target network params\n')

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.next_state: experiences[:].next_info_feat,
                self.state: experiences[:].info_feat,
            }
        )

        q_target = q_eval.copy()
        action_ind = experiences[:].action_feat
        reward = experiences[:].reward
        q_target[:, action_ind] = reward + self.gamma * np.max(q_next, axis=1)

        self.sess.run([self.train_op, self.loss],
                      feed_dict={
                          self.state: experiences[:].info_feat,
                          self.q_target: q_target
                      })

        self.learn_step_cnt += 1

    def predict_q(self, info):
        q_eval = self.sess.run(self.q_eval,
                      feed_dict={
                          self.state: info
                      })

        return np.argmax(q_eval)
