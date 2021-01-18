import numpy as np
import tensorflow as tf
import random, os
from copy import deepcopy


def fc(x, scope, nh, act=tf.nn.relu, init_scale=1.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=0))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x, w) + b
        h = act(z)
        return h


class Estimator:
    def __init__(self,
                 sess,
                 action_dim,
                 state_dim,
                 n_valid_node,
                 scope="estimator",
                 summaries_dir=None):
        self.sess = sess
        self.n_valid_node = n_valid_node
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.scope = scope
        self.T = 144

        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            value_loss = self._build_value_model()

            with tf.variable_scope("policy"):
                actor_loss, entropy = self._build_policy()

            self.loss = actor_loss + .5 * value_loss - 10 * entropy

        # Summaries
        self.summaries = tf.summary.merge([
            tf.summary.scalar("value_loss", self.value_loss),
            tf.summary.scalar("value_output", tf.reduce_mean(self.value_output)),
        ])

        self.policy_summaries = tf.summary.merge([
            tf.summary.scalar("policy_loss", self.policy_loss),
            tf.summary.scalar("adv", tf.reduce_mean(self.tfadv)),
            tf.summary.scalar("entropy", self.entropy),
        ])

        if summaries_dir:
            summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            self.summary_writer = tf.summary.FileWriter(summary_dir)

        self.neighbors_list = [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]]

    def _build_value_model(self):

        self.state = X = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name="X")

        # The TD target value
        self.y_pl = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="y")
        self.loss_lr = tf.placeholder(tf.float32, None, "learning_rate")

        # 3 layers network.
        l1 = fc(X, "l1", 128, act=tf.nn.relu)
        l2 = fc(l1, "l2", 64, act=tf.nn.relu)
        l3 = fc(l2, "l3", 32, act=tf.nn.relu)

        self.value_output = fc(l3, "value_output", 1, act=tf.nn.relu)
        self.value_loss = tf.reduce_mean(tf.squared_difference(self.y_pl, self.value_output))
        self.value_train_op = tf.train.AdamOptimizer(self.loss_lr).minimize(self.value_loss)

        return self.value_loss

    def _build_policy(self):

        self.policy_state = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name="P")
        self.ACTION = tf.placeholder(shape=[None, self.action_dim], dtype=tf.float32, name="action")
        self.tfadv = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='advantage')
        self.neighbor_mask = tf.placeholder(shape=[None, self.action_dim], dtype=tf.float32, name="neighbormask")

        l1 = fc(self.policy_state, "l1", 128, act=tf.nn.relu)
        l2 = fc(l1, "l2", 64, act=tf.nn.relu)
        l3 = fc(l2, "l3", 32, act=tf.nn.relu)
        # avoid valid_logits are all zeros
        self.logits = logits = fc(l3, "logits", self.action_dim, act=tf.nn.relu) + 1
        self.valid_logits = logits * self.neighbor_mask

        self.softmaxprob = tf.nn.softmax(tf.log(self.valid_logits + 1e-8))
        self.logsoftmaxprob = tf.nn.log_softmax(self.softmaxprob)

        self.neglogprob = - self.logsoftmaxprob * self.ACTION
        self.actor_loss = tf.reduce_mean(tf.reduce_sum(self.neglogprob * self.tfadv, axis=1))
        self.entropy = - tf.reduce_mean(self.softmaxprob * self.logsoftmaxprob)

        self.policy_loss = self.actor_loss - 0.01 * self.entropy
        self.policy_train_op = tf.train.AdamOptimizer(self.loss_lr).minimize(self.policy_loss)
        return self.actor_loss, self.entropy

    def predict(self, s):
        value_output = self.sess.run(self.value_output, {self.state: s})

        return value_output

    def action(self, s, ava_node, context, epsilon):

        value_output = self.sess.run(self.value_output, {self.state: s}).flatten()
        action_tuple = []
        valid_prob = []

        # For training policy gradient.
        action_choosen_mat = []
        policy_state = []
        curr_state_value = []
        next_state_ids = []

        grid_ids = [x for x in range(self.n_valid_node)]

        self.valid_action_mask = np.zeros((self.n_valid_node, self.action_dim))
        for i in range(len(ava_node)):
            for j in ava_node[i]:
                self.valid_action_mask[i][j] = 1
        curr_neighbor_mask = deepcopy(self.valid_action_mask)

        self.valid_neighbor_node_id = [[i for i in range(self.action_dim)], [i for i in range(self.action_dim)]]

        # compute policy probability.
        action_probs = self.sess.run(self.softmaxprob, {self.policy_state: s,
                                                        self.neighbor_mask: curr_neighbor_mask})
        curr_neighbor_mask_policy = []
        # sample action.
        for idx, grid_valid_idx in enumerate(grid_ids):
            action_prob = action_probs[idx]

            # action probability for state value function
            valid_prob.append(action_prob)
            if int(context[idx]) == 0:
                continue
            curr_action_indices_temp = np.random.choice(self.action_dim, int(context[idx]),
                                                        p=action_prob / np.sum(action_prob))
            curr_action_indices = [0] * self.action_dim
            for kk in curr_action_indices_temp:
                curr_action_indices[kk] += 1

            self.valid_neighbor_grid_id = self.valid_neighbor_node_id
            for curr_action_idx, num_driver in enumerate(curr_action_indices):
                if num_driver > 0:
                    end_node_id = int(self.valid_neighbor_node_id[grid_valid_idx][curr_action_idx])
                    action_tuple.append(end_node_id)

                    # for training
                    temp_a = np.zeros(self.action_dim)
                    temp_a[curr_action_idx] = 1
                    action_choosen_mat.append(temp_a)
                    policy_state.append(s[idx])
                    curr_state_value.append(value_output[idx])
                    next_state_ids.append(self.valid_neighbor_grid_id[grid_valid_idx][curr_action_idx])
                    curr_neighbor_mask_policy.append(curr_neighbor_mask[idx])

        return action_tuple, np.stack(valid_prob), \
               np.stack(policy_state), np.stack(action_choosen_mat), curr_state_value, \
               np.stack(curr_neighbor_mask_policy), next_state_ids

    def compute_advantage(self, curr_state_value, next_state_ids, next_state, node_reward, gamma):
        # compute advantage
        advantage = []
        node_reward = node_reward.flatten()
        qvalue_next = self.sess.run(self.value_output, {self.state: next_state}).flatten()
        for idx, next_state_id in enumerate(next_state_ids):
            temp_adv = sum(node_reward) + gamma * sum(qvalue_next) - curr_state_value[idx]
            advantage.append(temp_adv)
        return advantage

    def compute_targets(self, valid_prob, next_state, node_reward, gamma):
        # compute targets
        targets = []
        node_reward = node_reward.flatten()
        qvalue_next = self.sess.run(self.value_output, {self.state: next_state}).flatten()

        for idx in np.arange(self.n_valid_node):
            grid_prob = valid_prob[idx][self.valid_action_mask[idx] > 0]
            curr_grid_target = np.sum(
                grid_prob * (sum(node_reward) + gamma * sum(qvalue_next)))
            targets.append(curr_grid_target)

        return np.array(targets).reshape([-1, 1])

    def initialization(self, s, y, learning_rate):
        sess = self.sess
        feed_dict = {self.state: s, self.y_pl: y, self.loss_lr: learning_rate}
        _, value_loss = sess.run([self.value_train_op, self.value_loss], feed_dict)
        return value_loss

    def update_policy(self, policy_state, advantage, action_choosen_mat, curr_neighbor_mask, learning_rate,
                      global_step):
        sess = self.sess
        feed_dict = {self.policy_state: policy_state,
                     self.tfadv: advantage,
                     self.ACTION: action_choosen_mat,
                     self.neighbor_mask: curr_neighbor_mask,
                     self.loss_lr: learning_rate}
        summaries, _, loss = sess.run([self.policy_summaries, self.policy_train_op, self.policy_loss], feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
            self.summary_writer.flush()
        return loss

    def update_value(self, s, y, learning_rate, global_step):
        sess = self.sess
        feed_dict = {self.state: s, self.y_pl: y, self.loss_lr: learning_rate}
        summaries, _, loss = sess.run([self.summaries, self.value_train_op, self.value_loss], feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
            self.summary_writer.flush()
        return loss


class policyReplayMemory:
    def __init__(self, memory_size, batch_size):
        self.states = []
        self.neighbor_mask = []
        self.actions = []
        self.rewards = []
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.current = 0
        self.curr_lens = 0

    # Put data in policy replay memory
    def add(self, s, a, r, mask):
        if self.curr_lens == 0:
            self.states = s
            self.actions = a
            self.rewards = r
            self.neighbor_mask = mask
            self.curr_lens = self.states.shape[0]

        elif self.curr_lens <= self.memory_size:
            self.states = np.concatenate((self.states, s), axis=0)
            self.neighbor_mask = np.concatenate((self.neighbor_mask, mask), axis=0)
            self.actions = np.concatenate((self.actions, a), axis=0)
            self.rewards = np.concatenate((self.rewards, r), axis=0)
            self.curr_lens = self.states.shape[0]
        else:
            new_sample_lens = s.shape[0]
            index = random.randint(0, self.curr_lens - new_sample_lens)
            self.states[index:(index + new_sample_lens)] = s
            self.actions[index:(index + new_sample_lens)] = a
            self.rewards[index:(index + new_sample_lens)] = r
            self.neighbor_mask[index:(index + new_sample_lens)] = mask

    # Take a batch of samples
    def sample(self):
        if self.curr_lens <= self.batch_size:
            return [self.states, self.actions, np.array(self.rewards), self.neighbor_mask]
        indices = random.sample(list(range(0, self.curr_lens)), self.batch_size)
        batch_s = self.states[indices]
        batch_a = self.actions[indices]
        batch_r = self.rewards[indices]
        batch_mask = self.neighbor_mask[indices]
        return [batch_s, batch_a, batch_r, batch_mask]

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.neighbor_mask = []
        self.curr_lens = 0


class ReplayMemory:
    def __init__(self, memory_size, batch_size):
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []

        self.batch_size = batch_size
        self.memory_size = memory_size
        self.current = 0
        self.curr_lens = 0

    # Put data in policy replay memory
    def add(self, s, a, r, next_s):
        if self.curr_lens == 0:
            self.states = s
            self.actions = a
            self.rewards = r
            self.next_states = next_s
            self.curr_lens = self.states.shape[0]

        elif self.curr_lens <= self.memory_size:
            self.states = np.concatenate((self.states, s), axis=0)
            self.next_states = np.concatenate((self.next_states, next_s), axis=0)
            self.actions = np.concatenate((self.actions, a), axis=0)
            self.rewards = np.concatenate((self.rewards, r), axis=0)
            self.curr_lens = self.states.shape[0]
        else:
            new_sample_lens = s.shape[0]
            index = random.randint(0, self.curr_lens - new_sample_lens)
            self.states[index:(index + new_sample_lens)] = s
            self.actions[index:(index + new_sample_lens)] = a
            self.rewards[index:(index + new_sample_lens)] = r
            self.next_states[index:(index + new_sample_lens)] = next_s

    # Take a batch of samples
    def sample(self):
        if self.curr_lens <= self.batch_size:
            return [self.states, self.actions, self.rewards, self.next_states]
        indices = random.sample(list(range(0, self.curr_lens)), self.batch_size)
        batch_s = self.states[indices]
        batch_a = self.actions[indices]
        batch_r = self.rewards[indices]
        batch_mask = self.next_states[indices]
        return [batch_s, batch_a, batch_r, batch_mask]

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.curr_lens = 0


class ModelParametersCopier():
    def __init__(self, estimator1, estimator2):
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        self.update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            self.update_ops.append(op)

    def make(self, sess):
        sess.run(self.update_ops)
