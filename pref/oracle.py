import collections
import math

import gym
import numpy as np
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import random
from random import sample
from pref.utils import save_pickle, load_pickle


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        if j < len(sizes) - 2:
            layers += [nn.Dropout(0.5 if j > 0 else 0.2)]
    return nn.Sequential(*layers)


class HumanRewardNetwork(nn.Module):
    def __init__(self, obs_size, hidden_sizes=(64, 64)):
        super(HumanRewardNetwork, self).__init__()
        self.linear_relu_stack = mlp([obs_size] + list(hidden_sizes) + [1], activation=nn.LeakyReLU)
        self.tanh = nn.Tanh()

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return self.tanh(logits)


class HumanCritic:
    LEARNING_RATE = 0.0003
    BUFFER_SIZE = 1e5
    BATCH_SIZE = 10

    def __init__(self,
                 obs_size=3,
                 action_size=2,
                 maximum_segment_buffer=1000000,
                 maximum_preference_buffer=3500,
                 training_epochs=10,
                 batch_size=32,
                 hidden_sizes=(64, 64),
                 traj_k_lenght=100,
                 weight_decay=0.0,
                 learning_rate=0.0003,
                 regularize=False,
                 env_name=None,
                 custom_oracle=False,
                 seed=12345):
        print("created")
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ===BUFFER===
        self.segments = [None] * maximum_segment_buffer  # lists are very fast for random access
        self.pairs = [None] * maximum_preference_buffer
        self.critical_points = [None] * maximum_segment_buffer
        self.maximum_segment_buffer, self.maximum_preference_buffer, self.maximum_critical_points_buffer = maximum_segment_buffer, maximum_preference_buffer, maximum_segment_buffer
        self.segments_index, self.pairs_index, self.critical_points_index = 0, 0, 0
        self.segments_size, self.pairs_size, self.critical_points_size = 0, 0, 0
        self.segments_max_k_len = traj_k_lenght

        # === MODEL ===
        self.obs_size = obs_size
        self.action_size = action_size
        self.SIZES = hidden_sizes
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.init_model()  # creates model


        # === DATASET TRAINING ===
        self.training_epochs = training_epochs  # Default keras fit
        self.batch_size = batch_size  # Default keras fit

        self.writer = None  # SummaryWriter(working_path + reward_model_name)
        self.loss = nn.CrossEntropyLoss(reduction='sum')
        self.updates = 0

        self.pos_discount_start_multiplier = 1.0  # Ex: 1 * discount (1 * 2) = 2
        self.pos_discount = 0.5
        self.min_pos_discount = 0.01

        self.neg_discount_start_multiplier = 1.0  # Ex: 1 - discount ( 1 - 0.9) = 0.1
        self.neg_discount = 0.5
        self.min_neg_discount = 0.01

        self.punishments_given = 0
        self.approvements_given = 0

        self.regularize = regularize
        self.custom_oracle = custom_oracle
        self.oracle_reward_function = self.get_oracle_reward_function(env_name)

        if wandb.run is not None:
            wandb.define_metric("oracle/pairs")
            wandb.define_metric("oracle/*", step_metric="oracle/pairs")
            wandb.define_metric("oracle/total", summary="mean")
            wandb.define_metric("oracle/closeness", summary="mean")
            wandb.define_metric("oracle/towards_goal", summary="mean")
            wandb.define_metric("oracle/end_wall_closeness", summary="mean")
            wandb.define_metric("oracle/char_closeness", summary="mean")
            wandb.define_metric("oracle/true_rew", summary="mean")

            wandb.define_metric("reward/updates")
            wandb.define_metric("reward/*", step_metric="reward/updates")

    def update_params(self,
                      maximum_segment_buffer=1000000,
                      maximum_preference_buffer=3500,
                      batch_size=32,
                      hidden_sizes=(64, 64),
                      traj_k_lenght=100,
                      learning_rate=0.0003,
                      weight_decay=0.00001):

        # ===BUFFER===
        self.segments = [None] * maximum_segment_buffer  # lists are very fast for random access
        self.pairs = [None] * maximum_preference_buffer
        self.critical_points = [None] * maximum_segment_buffer
        self.maximum_segment_buffer, self.maximum_preference_buffer, self.maximum_critical_points_buffer = maximum_segment_buffer, maximum_preference_buffer, maximum_segment_buffer
        self.segments_index, self.pairs_index, self.critical_points_index = 0, 0, 0
        self.segments_size, self.pairs_size, self.critical_points_size = 0, 0, 0
        self.segments_max_k_len = traj_k_lenght

        self.SIZES = hidden_sizes
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.init_model()

        self.batch_size = batch_size

        self.updates = 0
        self.punishments_given = 0
        self.approvements_given = 0

    def get_oracle_reward_function(self, env_name):
        if not self.custom_oracle:
            print("real reward")
            return self.get_query_results_reward

        if env_name == 'LunarLanderContinuous-v2':
            return self.get_query_results_reward_lunar_lander
        elif env_name == 'Pendulum-v1':
            return self.get_query_results_reward_pendulum
        elif env_name == 'Walker2d-v3':
            return self.get_query_results_reward_walker
        elif env_name == 'Hopper-v3':
            return self.get_query_results_reward_hopper
        elif env_name == 'unity-env':
            return self.get_query_results_reward_socialnav
        elif env_name == 'Social-Nav-v1':
            print("Uses Social-Nav-v1 reward oracle")
            return self.get_query_results_reward_socialnav_v2
        else:
            return self.get_query_results_reward

    def init_model(self):
        # ==MODEL==
        self.reward_model = HumanRewardNetwork(self.obs_size[0] + self.action_size, self.SIZES)

        # ==OPTIMIZER==
        self.optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def clear_segment(self):
        self.segments = [None] * self.maximum_segment_buffer
        self.segments_size, self.segments_index = 0, 0

    def add_segment(self, o, total_reward):
        assert len(o) <= self.segments_max_k_len
        self.segments[self.segments_index] = [o, total_reward]
        self.segments_size = min(self.segments_size + 1, self.maximum_segment_buffer)
        self.segments_index = (self.segments_index + 1) % self.maximum_segment_buffer

    def add_segment_human(self, o):
        assert len(o) <= self.segments_max_k_len
        self.segments[self.segments_index] = [o]
        self.segments_size = min(self.segments_size + 1, self.maximum_segment_buffer)
        self.segments_index = (self.segments_index + 1) % self.maximum_segment_buffer

    def add_critical_points(self, min_index, max_index):
        self.critical_points[self.critical_points_index] = [min_index, max_index]
        self.critical_points_size = min(self.critical_points_size + 1, self.maximum_critical_points_buffer)
        self.critical_points_index = (self.critical_points_index + 1) % self.maximum_critical_points_buffer

    def add_pairs(self, o0, o1, preference):
        self.pairs[self.pairs_index] = [o0, o1, preference]
        self.pairs_size = min(self.pairs_size + 1, self.maximum_preference_buffer)
        self.pairs_index = (self.pairs_index + 1) % self.maximum_preference_buffer

    def add_pairs_with_critical_points(self, o0, o1, preference, critical_points):
        self.pairs[self.pairs_index] = [o0, o1, preference, critical_points]
        self.pairs_size = min(self.pairs_size + 1, self.maximum_preference_buffer)
        self.pairs_index = (self.pairs_index + 1) % self.maximum_preference_buffer

    def random_sample_segments(self, number_of_sampled_segments=64):
        idxs = sample(range(self.segments_size), number_of_sampled_segments)
        return [self.segments[idx] for idx in idxs]

    def save_buffers(self, path="", env_name="", save_name="buffer"):
        save_pickle(self.segments, path + "segments_" + env_name + save_name)
        save_pickle(self.segments_size, path + "segments_size_" + env_name + save_name)
        save_pickle(self.segments_index, path + "segments_index_" + env_name + save_name)
        save_pickle(self.pairs, path + "pairs_" + env_name + save_name)
        save_pickle(self.pairs_size, path + "pairs_size_" + env_name + save_name)
        save_pickle(self.pairs_index, path + "pairs_index_" + env_name + save_name)
        save_pickle(self.critical_points, path + "critical_points_" + env_name + save_name)
        save_pickle(self.critical_points_size, path + "critical_points_size_" + env_name + save_name)
        save_pickle(self.critical_points_index, path + "critical_points_index_" + env_name + save_name)

    def load_buffers(self, path="", env_name="", load_name="buffer"):
        self.segments = load_pickle(path + "segments_" + env_name + load_name)
        self.segments_size = load_pickle(path + "segments_size_" + env_name + load_name)
        self.segments_index = load_pickle(path + "segments_index_" + env_name + load_name)
        self.pairs = load_pickle(path + "pairs_" + env_name + load_name)
        self.pairs_size = load_pickle(path + "pairs_size_" + env_name + load_name)
        self.pairs_index = load_pickle(path + "pairs_index_" + env_name + load_name)
        self.pairs = load_pickle(path + "critical_points_" + env_name + load_name)
        self.pairs_size = load_pickle(path + "critical_points_size_" + env_name + load_name)
        self.pairs_index = load_pickle(path + "critical_points_index_" + env_name + load_name)

    def get_queries_to_be_trained(self, number_of_queries=100, experiments=50, truth=100):
        queries = []
        for _ in range(experiments):
            segments = self.random_sample_segments(number_of_sampled_segments=number_of_queries * 2)  # this sampling obviously can be smarter
            for idx in range(0, len(segments), 2):
                query = self.oracle_reward_function(segments[idx], segments[idx + 1], truth)
                queries.append(query)
        return queries


    def get_query_results_reward_with_critical_points(self, segment1, segment2, critical_points, truth):
        total_reward_1 = segment1[-1]
        total_reward_2 = segment2[-1]
        truth_percentage = truth / 100.0
        fakes_percentage = 1 - truth_percentage
        if total_reward_1 > total_reward_2:
            preference = [1, 0] if fakes_percentage < random.random() else [0, 1]
            point = critical_points[0] if preference[0] == 1 else critical_points[1]
        elif total_reward_1 < total_reward_2:
            preference = [0, 1] if fakes_percentage < random.random() else [1, 0]
            point = critical_points[1] if preference[1] == 1 else critical_points[0]
        elif abs(total_reward_1 - total_reward_2) < 1:
            preference = [0.5, 0.5]
            point = [-1, -1]  # None is preferred and therefore give no additional point
        else:
            raise "Error computing preferences"
        return [segment1, segment2, preference, point]

    def train_dataset(self, dataset, meta_data=None, epochs_override=-1, loss_threshold=15):
        epochs = epochs_override if epochs_override != -1 else self.training_epochs
        losses = collections.deque(maxlen=10)
        for epoch in range(1, epochs + 1):
            running_loss = 0
            running_accuracy = 0

            for step, (o1, o2, prefs) in enumerate(dataset):
                loss = 0.0
                self.optimizer.zero_grad()

                o1_unrolled = torch.reshape(o1, [-1, self.obs_size[0] + self.action_size])
                o2_unrolled = torch.reshape(o2, [-1, self.obs_size[0] + self.action_size])
                r1_unrolled = self.reward_model(o1_unrolled)
                r2_unrolled = self.reward_model(o2_unrolled)
                r1_rolled = torch.reshape(r1_unrolled, o1.shape[0:2])
                r2_rolled = torch.reshape(r2_unrolled, o2.shape[0:2])

                rs1 = torch.sum(r1_rolled, dim=1)
                rs2 = torch.sum(r2_rolled, dim=1)

                rss = torch.stack([rs1, rs2])
                rss = torch.t(rss)

                preds = torch.softmax(rss, dim=0)
                preds_correct = torch.eq(torch.argmax(prefs, 1), torch.argmax(preds, 1)).type(torch.float32)
                accuracy = torch.mean(preds_correct)

                prefs = torch.max(prefs, 1)[1]  # TODO: Think about how to deal with this
                loss = self.loss(rss, prefs)

                running_loss += loss.detach().numpy().item()
                running_accuracy += accuracy

                if meta_data is not None:
                    meta_data['loss'].append(loss.detach().numpy())
                    meta_data['accuracy'].append(accuracy.numpy())

                reporting_interval = (self.training_epochs // 10) if self.training_epochs >= 10 else 1
                if epoch % reporting_interval == 0 and step == len(dataset) - 1:
                    print("Epoch %d , Training loss (for one batch) at step %d: %.4f, Accuracy %.4f" % (epoch, step, float(loss), float(accuracy)))
                    print("Seen so far: %s samples" % ((step + 1) * self.batch_size))

                loss.backward()
                self.optimizer.step()

            episode_loss = (running_loss / len(dataset))
            episode_accuracy = (running_accuracy / len(dataset))

            if self.writer:
                self.writer.add_scalar("reward/loss", episode_loss, self.updates)
                self.writer.add_scalar("reward/accuracy", episode_accuracy, self.updates)
            if wandb.run is not None:
                wandb.log({"reward/loss": episode_loss,
                           "reward/accuracy": episode_accuracy,
                           "reward/updates": self.updates
                           })
            self.updates += 1

            losses.append(episode_loss)
            if sum(losses) / len(losses) < 1.0 and epoch > epochs // 2:
                break

        return meta_data

    def train_dataset_with_critical_points(self, dataset, meta_data, epochs_override=-1):
        max_regularization_sum = 0
        reg_sum = 1
        for i in range(5):
            max_regularization_sum += reg_sum
            reg_sum = reg_sum * 0.5

        epochs = epochs_override if epochs_override != -1 else self.training_epochs
        self.reward_model.train(True)
        for epoch in range(1, epochs + 1):

            running_loss = 0
            running_accuracy = 0
            running_regularization_loss_punishment = 0
            running_regularization_loss_approve = 0
            episode_approve_reward = 0
            episode_punishment_reward = 0
            episode_punishments = 0
            episode_approvements = 0

            for step, (o1, o2, prefs, critical_points) in enumerate(dataset):
                loss = 0.0
                self.optimizer.zero_grad()
                o1_unrolled = torch.reshape(o1, [-1, self.obs_size[0] + self.action_size])
                o2_unrolled = torch.reshape(o2, [-1, self.obs_size[0] + self.action_size])

                r1_unrolled = self.reward_model(o1_unrolled)
                r2_unrolled = self.reward_model(o2_unrolled)

                r1_rolled = torch.reshape(r1_unrolled, o1.shape[0:2])
                r2_rolled = torch.reshape(r2_unrolled, o2.shape[0:2])

                rs1 = torch.sum(r1_rolled, dim=1)
                rs2 = torch.sum(r2_rolled, dim=1)
                rss = torch.stack([rs1, rs2])
                rss = torch.t(rss)

                preds = torch.softmax(rss, dim=0)
                preds_correct = torch.eq(torch.argmax(prefs, 1), torch.argmax(preds, 1)).type(torch.float32)
                accuracy = torch.mean(preds_correct)

                #loss_fn = nn.CrossEntropyLoss(reduction="sum")
                loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
                approve_reward, punishment_reward, n_approve, n_punishment = self.get_critical_points_rewards(critical_points, prefs, r1_rolled,
                                                                                     r2_rolled)
                approve_reward = (approve_reward / n_approve)
                punishment_reward = (punishment_reward / n_punishment)
                episode_approve_reward += approve_reward
                episode_punishment_reward += punishment_reward
                episode_punishments += n_punishment
                episode_approvements += n_approve

                # L1 regularization to create sparse representation

                l1_lambda = 0.0001
                l1_norm = sum(abs(p).sum() for p in self.reward_model.parameters())

                if self.regularize:

                    #regularization_approve = abs(max_regularization_sum - approve_reward)  # 2 = 1 + 0.5 + 0.5^2 + 0.5^3 geometric sum
                    #regularization_punishment = abs(max_regularization_sum + punishment_reward)
                    running_regularization_loss_approve += approve_reward
                    running_regularization_loss_punishment += punishment_reward

                    #running_regularization_loss_punishment += punishment_reward
                    #running_regularization_loss_approve += approve_reward
                    #prefs = torch.max(prefs, 1)[1]  # TODO: Seems to be a problem with UnityEnv
                    loss = loss_fn(rss, prefs) - approve_reward * 10 + punishment_reward * 0.5 #+ l1_lambda * l1_norm
                else:
                    #prefs = torch.max(prefs, 1)[1]  # TODO: Seems to be a problem with UnityEnv
                    loss = loss_fn(rss, prefs) #+ l1_lambda * l1_norm
                running_loss += loss.detach().numpy().item()
                running_accuracy += accuracy

                if meta_data is not None:
                    meta_data['loss'].append(loss.detach().numpy())
                    meta_data['accuracy'].append(accuracy.numpy())

                reporting_interval = (self.training_epochs // 10) if self.training_epochs >= 10 else 1
                if epoch % reporting_interval == 0 and step == len(dataset) - 1:
                    print("Epoch %d , Training loss (for one batch) at step %d: %.4f, Accuracy %.4f" % (epoch, step, float(loss), float(accuracy)))
                    print("Seen so far: %s samples" % ((step + 1) * self.batch_size))

                loss.backward()
                self.optimizer.step()

            episode_loss = (running_loss / len(dataset))
            episode_accuracy = (running_accuracy / len(dataset))
            episode_regularization_loss_punishment = (running_regularization_loss_punishment / len(dataset))
            episode_regularization_loss_approve = (running_regularization_loss_approve / len(dataset))
            if self.writer:
                self.writer.add_scalar("reward/loss", episode_loss, self.updates)
                self.writer.add_scalar("reward/approvements_reward", episode_approve_reward / len(dataset), self.updates)
                self.writer.add_scalar("reward/punishment_reward", episode_punishment_reward / len(dataset), self.updates)
                self.writer.add_scalar("reward/accuracy", episode_accuracy, self.updates)
                self.writer.add_scalar("reward/punishment_loss", episode_regularization_loss_punishment, self.updates)
                self.writer.add_scalar("reward/approve_loss", episode_regularization_loss_approve, self.updates)
            if wandb.run is not None:
                wandb.log({"reward/loss": episode_loss,
                           "reward/accuracy": episode_accuracy,
                           "reward/approvements_reward": episode_approve_reward / len(dataset),
                           "reward/punishment_reward": episode_punishment_reward / len(dataset),
                           "reward/punishment_loss": episode_regularization_loss_punishment,
                           "reward/approve_loss": episode_regularization_loss_approve,
                           "reward/punishments": episode_punishments,
                           "reward/approvements": episode_approvements,
                           "reward/updates": self.updates
                           })
            self.updates += 1
        self.reward_model.train(False)
        return meta_data

    def get_critical_points_rewards(self, critical_points, prefs, r1_rolled, r2_rolled):
        critical_points_discounted_reward_punishment = torch.zeros_like(r1_rolled)
        critical_points_discounted_reward_approve = torch.zeros_like(r1_rolled)
        for i in range(len(prefs)):
            if prefs[i][0] == 1:
                critical_points_discounted_reward_punishment[i] = r2_rolled[i] * critical_points[i, :, 0]
                critical_points_discounted_reward_approve[i] = r1_rolled[i] * critical_points[i, :, 1]
            if prefs[i][1] == 1:
                critical_points_discounted_reward_punishment[i] = r1_rolled[i] * critical_points[i, :, 0]
                critical_points_discounted_reward_approve[i] = r2_rolled[i] * critical_points[i, :, 1]

        punishments_in_batch = torch.sum(critical_points[:, :, 0] == 1).item()
        approvements_in_batch = torch.sum(critical_points[:, :, 1] == 1).item()

        punishments_in_batch = punishments_in_batch if punishments_in_batch != 0 else 1
        approvements_in_batch = approvements_in_batch if approvements_in_batch != 0 else 1

        punishment_reward = torch.sum(critical_points_discounted_reward_punishment) #/ punishments_in_batch
        approve_reward = torch.sum(critical_points_discounted_reward_approve) #/ approvements_in_batch
        return approve_reward, punishment_reward, approvements_in_batch, punishments_in_batch

    def generate_data_for_training(self, queries):
        queries = np.array(queries, dtype=object)
        o1, o2, prefs = queries[:, 0, 0], queries[:, 1, 0], queries[:, 2]
        o1 = [np.stack(segments) for segments in o1]
        o2 = [np.stack(segments) for segments in o2]
        prefs = np.asarray(prefs).astype('float32')
        return o1, o2, prefs

    def generate_data_for_training_human(self, queries):
        queries = np.array(queries, dtype=object)
        o1, o2, prefs = queries[:, 0], queries[:, 1], [pref for pref in queries[:, 2]]
        o1 = [np.stack(segments) for segments in o1]
        o2 = [np.stack(segments) for segments in o2]
        prefs = np.asarray(prefs).astype('float32')
        return o1, o2, prefs

    def generate_data_for_training_with_critical_points(self, queries):
        queries = np.array(queries, dtype=object)
        o1, o2, prefs, critical_points = queries[:, 0, 0], queries[:, 1, 0], queries[:, 2], queries[:, 3]
        o1 = [np.stack(segments) for segments in o1]
        o2 = [np.stack(segments) for segments in o2]

        critical_points = self.generate_critical_point_segment(critical_points)
        prefs = np.asarray(prefs).astype('float32')
        return o1, o2, prefs, critical_points

    def generate_critical_point_segment(self, critical_points):
        rolled_critical_points = [[[0, 0] for _ in range(self.segments_max_k_len)] for _ in range(len(critical_points))]
        for i in range(len(critical_points)):
            neg_index = critical_points[i][0]
            pos_index = critical_points[i][1]

            if pos_index != -1:
                current_pos_discount = self.pos_discount_start_multiplier
                for j in reversed(range(max(0, pos_index - 5), pos_index + 1)):
                    rolled_critical_points[i][j][1] = max(current_pos_discount, self.min_pos_discount)
                    current_pos_discount *= self.pos_discount

            if neg_index != -1:
                current_neg_discount = self.neg_discount_start_multiplier
                for j in reversed(range(max(0, neg_index - 5), neg_index + 1)):
                    rolled_critical_points[i][j][0] = max(current_neg_discount, self.min_neg_discount)
                    current_neg_discount *= self.neg_discount
        critical_points = np.asarray(rolled_critical_points).astype('float32')
        return critical_points

    def save_reward_model(self, env_name="LunarLanderContinuous-v2"):
        torch.save(self.reward_model.state_dict(), env_name)

    def load_reward_model(self, env_name="LunarLanderContinuous-v2"):
        print("loading:" + "models/reward_model/" + env_name)
        self.reward_model.load_state_dict(torch.load(env_name))

    def get_all_preference_pairs(self):
        pairs = [self.pairs[idx] for idx in range(self.pairs_size)]
        obs1, obs2, prefs = self.generate_data_for_training(pairs)
        return obs1, obs2, prefs

    def get_all_preference_pairs_human(self):
        pairs = [self.pairs[idx] for idx in range(self.pairs_size)]
        obs1, obs2, prefs = self.generate_data_for_training_human(pairs)
        return obs1, obs2, prefs

    def get_all_preference_pairs_with_critical_points(self):
        pairs = [self.pairs[idx] for idx in range(self.pairs_size)]
        obs1, obs2, prefs, critical_points = self.generate_data_for_training_with_critical_points(pairs)
        return obs1, obs2, prefs, critical_points

    def generate_preference_pairs(self, trajectories, number_of_queries=200, truth=100):
        for _ in range(number_of_queries):
            segments = self.random_sample_batch_segments(trajectories, number_of_sampled_segments=2)
            query = self.oracle_reward_function(segments[0], segments[1], truth)
            self.add_pairs(query[0], query[1], query[2])

    def generate_preference_pairs_human(self, segments, pref):
        query_pref = [-1, -1]
        if pref == 1:
            query_pref = [1, 0]
        elif pref == 2:
            query_pref = [0, 1]
        else:
            query_pref = [0.5, 0.5]
        self.add_pairs(segments[0], segments[1], query_pref)

    def generate_preference_pairs_with_critical_points(self, trajectories, critical_points, number_of_queries=200, truth=100):
        for _ in range(number_of_queries):
            segments, points = self.random_sample_batch_segments_with_critical_points(trajectories, critical_points, number_of_sampled_segments=2)
            query = self.oracle_reward_function(segments[0], segments[1], truth, points)
            self.add_pairs_with_critical_points(query[0], query[1], query[2], query[3])

    def generate_preference_pairs_information_based(self, trajectories, critical_points, number_of_queries=200, truth=100, uncertain_ratio=0.7):
        rewards = []
        for i in range(len(trajectories)):
            trajectory = trajectories[i][0]
            trajectory = torch.as_tensor(trajectory, dtype=torch.float32)
            rew_unrolled = self.reward_model(trajectory)
            rs = torch.sum(rew_unrolled).item()
            rewards.append([rs, i])
        mean = sum([rew[0] for rew in rewards]) / len(rewards)
        rewards = sorted(rewards, key=lambda rew: abs(rew[0] - mean))

        indexes = set()

        uniform_segments = round(number_of_queries * 5 * (1-uncertain_ratio))
        idxs = sample(range(len(trajectories)), uniform_segments)
        for idx in idxs:
            indexes.add(idx)

        uncertain_segments = round(number_of_queries * 5 * uncertain_ratio)
        for i in range(uncertain_segments):
            indexes.add(rewards[i][1])

        indexes = list(indexes)
        trajectories = [trajectories[idx] for idx in indexes]
        critical_points = [critical_points[idx] for idx in indexes]

        for _ in range(number_of_queries):
            segments, points = self.random_sample_batch_segments_with_critical_points(trajectories, critical_points, number_of_sampled_segments=2)
            query = self.oracle_reward_function(segments[0], segments[1], truth, points)
            self.add_pairs_with_critical_points(query[0], query[1], query[2], query[3])


    def random_sample_batch_segments(self, trajectories, number_of_sampled_segments=64):
        idxs = sample(range(len(trajectories)), number_of_sampled_segments)
        return [trajectories[idx] for idx in idxs]

    def random_sample_batch_segments_and_videos(self, trajectories, video_names, number_of_sampled_segments=64):
        idxs = sample(range(len(trajectories)), number_of_sampled_segments)
        return [trajectories[idx] for idx in idxs], [video_names[idx] for idx in idxs], idxs

    def random_sample_batch_segments_with_critical_points(self, trajectories, critical_points, number_of_sampled_segments=64):
        idxs = sample(range(len(trajectories)), number_of_sampled_segments)
        return [trajectories[idx] for idx in idxs], [critical_points[idx] for idx in idxs]

    def get_query_results_reward_lunar_lander(self, segment1, segment2, truth, critical_points):
        total_reward_1_right = sum([-0.3 if transition[0] < 0 else 0 for transition in segment1[0]])
        total_reward_2_right = sum([-0.3 if transition[0] < 0 else 0 for transition in segment2[0]])

        total_reward_1 = segment1[-1] + total_reward_1_right
        total_reward_2 = segment2[-1] + total_reward_2_right
        #total_reward_1 = total_reward_1_right
        #total_reward_2 = total_reward_2_right
        truth_percentage = truth / 100.0
        fakes_percentage = 1 - truth_percentage
        if total_reward_1 > total_reward_2:
            preference = [1, 0] if fakes_percentage < random.random() else [0, 1]
            if preference[0] == 1:
                point = [critical_points[1][0], critical_points[0][1]]
            else:
                point = [critical_points[0][0], critical_points[1][1]]
        elif total_reward_1 < total_reward_2:
            preference = [0, 1] if fakes_percentage < random.random() else [1, 0]
            if preference[1] == 1:
                point = [critical_points[0][0], critical_points[1][1]]
            else:
                point = [critical_points[1][0], critical_points[0][1]]
        else:
            preference = [0.5, 0.5]
            point = [-1, -1]
        return [segment1, segment2, preference, point]

    def get_query_results_reward_pendulum(self, segment1, segment2, truth):
        """
        Calculates the reward for the two segments and pick the correct one.
        True Reward range for Pendulum 1 step: (0, -16.2736044)
        True Reward range for Pendulum 50 step trajectories: (0, -813.2736044)
        True Reward range for Pendulum full episode(150 step): (0, -2441)

        Max negative reward the agent can give per timestep: -2000
        Max positive extra reward the oracle can give per timestep: 4 + 4 = 8
            - 4 for being on the right side
            - 4 * (1 - close) where close = 0 when at the top
        total_reward_1_count = sum([1 if transition[1] > 0 else 0 for transition in segment1[0]])
        total_reward_2_count = sum([1 if transition[1] > 0 else 0 for transition in segment2[0]])
        total_reward_1_right = 1000 if total_reward_1_right > total_reward_2_right else 0
        total_reward_2_right = 1000 if total_reward_2_right > total_reward_1_right else 0
        Reasoning, we want to severely punish the agent if it's in the wrong area.
        We also want to give a tiny boost if it's on the correct side, but not too much so we ruin the logic of the true reward
        """
        total_reward_1_right = sum([-1000 if 0.15 < transition[1] < 0.35 and transition[0] > 0 else 0 for transition in segment1[0]])
        total_reward_2_right = sum([-1000 if 0.15 < transition[1] < 0.35 and transition[0] > 0 else 0 for transition in segment2[0]])
        total_reward_1_right_other = 0
        total_reward_2_right_other = 0

        #total_reward_1_count = sum([1 if -1 <= transition[1] <= 0 else 0 for transition in segment1[0]])
        #total_reward_2_count = sum([1 if -1 <= transition[1] <= 0 else 0 for transition in segment2[0]])
        #total_reward_1_right += 1000 if total_reward_1_count > total_reward_2_count else 0
        #total_reward_2_right += 1000 if total_reward_2_count > total_reward_1_count else 0

        #total_reward_1_right_other = sum([2 * (2 - abs(transition[1])) if -1 <= transition[1] <= 0 and transition[0] > 0 else 0 for transition in segment1[0]])
        #total_reward_2_right_other = sum([2 * (2 - abs(transition[1])) if -1 <= transition[1] <= 0 and transition[0] > 0 else 0 for transition in segment2[0]])

        #total_reward_1_right_other += sum([abs(transition[1]) if -1 <= transition[1] <= 0 else 0 for transition in segment1[0]])
        #total_reward_2_right_other += sum([abs(transition[1]) if -1 <= transition[1] <= 0 else 0 for transition in segment2[0]])

        total_reward_1 = segment1[-1] + total_reward_1_right + total_reward_1_right_other
        total_reward_2 = segment2[-1] + total_reward_2_right + total_reward_2_right_other

        truth_percentage = truth / 100.0
        fakes_percentage = 1 - truth_percentage
        if total_reward_1 > total_reward_2:
            preference = [1, 0] if fakes_percentage < random.random() else [0, 1]
        elif total_reward_1 < total_reward_2:
            preference = [0, 1] if fakes_percentage < random.random() else [1, 0]
        elif abs(total_reward_1 - total_reward_2) < 1:
            preference = [0.5, 0.5]
        else:
            raise "Error computing preferences"
        return [segment1, segment2, preference]

    def get_query_results_reward_walker(self, segment1, segment2, truth):
        total_reward_1_right = sum([-1 if transition[0] > 1.25 else 0 for transition in segment1[0]])
        total_reward_2_right = sum([-1 if transition[0] > 1.25 else 0 for transition in segment2[0]])
        total_reward_1_right += sum([1 if transition[0] > 0.8 else 0 for transition in segment1[0]])
        total_reward_2_right += sum([1 if transition[0] > 0.8 else 0 for transition in segment2[0]])
        total_reward_1_right += sum([-50 if transition[0] <= 0.8 else 0 for transition in segment1[0]])
        total_reward_2_right += sum([-50 if transition[0] <= 0.8 else 0 for transition in segment2[0]])

        total_reward_1 = segment1[-1] + total_reward_1_right
        total_reward_2 = segment2[-1] + total_reward_2_right
        truth_percentage = truth / 100.0
        fakes_percentage = 1 - truth_percentage
        if total_reward_1 > total_reward_2:
            preference = [1, 0] if fakes_percentage < random.random() else [0, 1]
        elif total_reward_1 < total_reward_2:
            preference = [0, 1] if fakes_percentage < random.random() else [1, 0]
        elif abs(total_reward_1 - total_reward_2) < 1:
            preference = [0.5, 0.5]
        else:
            raise "Error computing preferences"
        return [segment1, segment2, preference]

    def get_query_results_reward_hopper(self, segment1, segment2, truth):
        #total_reward_1_right = sum([transition[0] if transition[0] > 1.0 else 0 for transition in segment1[0]])
        #total_reward_2_right = sum([transition[0] if transition[0] > 1.0 else 0 for transition in segment2[0]])
        total_reward_1_right = sum([transition[0]**2 if transition[0] > 1.0 else 0 for transition in segment1[0]])
        total_reward_2_right = sum([transition[0]**2 if transition[0] > 1.0 else 0 for transition in segment2[0]])

        #total_reward_1_right += sum([-50 if transition[0] <= 0.7 else 0 for transition in segment1[0]])
        #total_reward_2_right += sum([-50 if transition[0] <= 0.7 else 0 for transition in segment2[0]])

        total_reward_1 = segment1[-1] + total_reward_1_right
        total_reward_2 = segment2[-1] + total_reward_2_right

        truth_percentage = truth / 100.0
        fakes_percentage = 1 - truth_percentage
        if total_reward_1 > total_reward_2:
            preference = [1, 0] if fakes_percentage < random.random() else [0, 1]
        elif total_reward_1 < total_reward_2:
            preference = [0, 1] if fakes_percentage < random.random() else [1, 0]
        elif abs(total_reward_1 - total_reward_2) < 1:
            preference = [0.5, 0.5]
        else:
            raise "Error computing preferences"
        return [segment1, segment2, preference]

    def shift_interval(self, low=-1, high=1, min_threshold=0, max_threshold=1, x=None):
        return min_threshold + ((max_threshold - min_threshold) / (high - low)) * (x - low)

    def get_query_results_reward_socialnav(self, segment1, segment2, truth):

        reward_force_1 = sum([self.shift_interval(x=transition[-1]) * 1.0 / 20 if transition[-1] > -0.4 else 0 for transition in segment1[0]])
        reward_force_2 = sum([self.shift_interval(x=transition[-1]) * 1.0 / 20 if transition[-1] > -0.4 else 0 for transition in segment2[0]])
        punish_no_force_1 = sum([-1.0 / 10 if transition[-1] <= -0.4 else 0 for transition in segment1[0]])
        punish_no_force_2 = sum([-1.0 / 10 if transition[-1] <= -0.4 else 0 for transition in segment2[0]])


        #Might need to help it not crash
        punishment_reward_1 = segment1[-1] * 5 if segment1[-1] <= -0.9 else 0
        punishment_reward_2 = segment2[-1] * 5 if segment2[-1] <= -0.9 else 0

        total_reward_1 = segment1[-1] + reward_force_1 + punishment_reward_1 + punish_no_force_1
        total_reward_2 = segment2[-1] + reward_force_2 + punishment_reward_2 + punish_no_force_2

        truth_percentage = truth / 100.0
        fakes_percentage = 1 - truth_percentage
        if total_reward_1 > total_reward_2:
            preference = [1, 0] if fakes_percentage < random.random() else [0, 1]
        elif total_reward_1 < total_reward_2:
            preference = [0, 1] if fakes_percentage < random.random() else [1, 0]
        elif abs(total_reward_1 - total_reward_2) < 1:
            preference = [0.5, 0.5]
        else:
            raise "Error computing preferences"
        return [segment1, segment2, preference]

    def get_query_results_reward_socialnav_v2(self, segment1, segment2, truth, critical_points):
        closeness_reward1 = 0
        closeness_reward2 = 0
        transitions1 = segment1[0]
        transitions2 = segment2[0]

        obs_len = len(transitions1[0])
        ray_size = 5

        """
        start_index = (0 * obs_len)
        max_reward_sensor1_start = 0
        max_reward_sensor2_start = 0
        for ray_index in range(9):
            saw_goal1 = transitions1[start_index + ray_index * ray_size + 2]
            saw_goal2 = transitions2[start_index + ray_index * ray_size + 2]

            distance1 = transitions1[start_index + ray_start_index + ray_index * ray_size + 4]
            distance2 = transitions2[start_index + ray_start_index + ray_index * ray_size + 4]

            max_reward_sensor1_start = max(max_reward_sensor1_start, saw_goal1 * (1 - distance1))
            max_reward_sensor2_start = max(max_reward_sensor2_start, saw_goal2 * (1 - distance2))

        start_index = (len(transitions1) - 1) * obs_len
        max_reward_sensor1_final = 0
        max_reward_sensor2_final = 0
        for ray_index in range(9):
            saw_goal1 = transitions1[start_index + ray_index * ray_size + 2]
            saw_goal2 = transitions2[start_index + ray_index * ray_size + 2]

            distance1 = transitions1[start_index + ray_start_index + ray_index * ray_size + 4]
            distance2 = transitions2[start_index + ray_start_index + ray_index * ray_size + 4]

            max_reward_sensor1_final = max(max_reward_sensor1_final, saw_goal1 * (1 - distance1))
            max_reward_sensor2_final = max(max_reward_sensor2_final, saw_goal2 * (1 - distance2))
        closeness_reward1 = (max_reward_sensor1_start - max_reward_sensor1_final) * 0.2
        closeness_reward2 = (max_reward_sensor2_start - max_reward_sensor2_final) * 0.2

        for i in range(len(transitions1)):
            max_reward_sensor1 = 0
            max_reward_sensor2 = 0
            min_reward_sensor1 = 0
            min_reward_sensor2 = 0
            for ray_index in range(9):
                saw_goal1 = transitions1[i][ray_index * ray_size + 2]
                saw_enemy1 = transitions1[i][ray_index * ray_size + 0] + transitions1[i][ray_index * ray_size + 1]
                saw_goal2 = transitions2[i][ray_index * ray_size + 2]
                saw_enemy2 = transitions2[i][ray_index * ray_size + 0] + transitions2[i][ray_index * ray_size + 1]

                distance1 = transitions1[i][ray_index * ray_size + 4]
                distance2 = transitions2[i][ray_index * ray_size + 4]

                max_reward_sensor1 = max(max_reward_sensor1, (saw_goal1 * (1 - distance1)**2) * 0.01)
                max_reward_sensor2 = max(max_reward_sensor2, (saw_goal2 * (1 - distance2)**2) * 0.01)
                min_reward_sensor1 = max(min_reward_sensor1, (saw_enemy1 * (1 - distance1) ** 2) * 0.002)
                min_reward_sensor2 = max(min_reward_sensor2, (saw_enemy2 * (1 - distance2) ** 2) * 0.002)
            closeness_reward1 += max_reward_sensor1 #- min_reward_sensor1
            closeness_reward2 += max_reward_sensor2 #- min_reward_sensor2

        # When we have no access to position
        #closeness_reward1 += sum([abs(transition[-1]) / 1000 for transition in transitions1])
        #closeness_reward2 += sum([abs(transition[-1]) / 1000 for transition in transitions2])


        # When we have access to position
        closeness_reward1 += sum([transition[-3] / 2000 for transition in transitions1])
        closeness_reward2 += sum([transition[-3] / 2000 for transition in transitions2])

        # Punish if robot just stands still and do nothing
        not_moving1 = abs(transitions1[-1][-3] - transitions1[0][-3]) < 1 \
                      and abs(transitions1[-1][-4] - transitions1[0][-4]) < 1

        not_moving2 = abs(transitions2[-1][-3] - transitions2[0][-3]) < 1 \
                      and abs(transitions2[-1][-4] - transitions2[0][-4]) < 1
        closeness_reward1 -= 0.5 if not_moving1 else 0
        closeness_reward2 -= 0.5 if not_moving2 else 0
        """
        """
        magnitude_x = math.sqrt(transitions1[2] ** 2 + transitions1[3] ** 2)
        magnitude_z = math.sqrt(transitions1[-2] ** 2 + transitions1[-1] ** 2)
        dot_product1 = transitions1[-2] * transitions1[2] + transitions1[3] * transitions1[-1]
        same_direction1 = dot_product1 / (magnitude_x * magnitude_z)

        magnitude_x = math.sqrt(transitions2[2] ** 2 + transitions2[3] ** 2)
        magnitude_z = math.sqrt(transitions2[-2] ** 2 + transitions2[-1] ** 2)
        dot_product2 = transitions2[-2] * transitions2[2] + transitions2[3] * transitions2[-1]
        same_direction2 = dot_product2 / (magnitude_x * magnitude_z)
        """

        def vector_of_segment(start, end):
            a, b = start
            c, d = end
            return (c - a, d - b)

        def scalar_product(u, v):
            a, b = u
            c, d = v
            return a * c + b * d

        def norm(u):
            return math.sqrt(scalar_product(u, u))

        def cosine_similarity(u, v):
            return scalar_product(u, v) / (norm(u) * norm(v))

        def cosine_similarity_of_roads(line1, line2):
            u = vector_of_segment(*line1)
            v = vector_of_segment(*line2)
            return cosine_similarity(u, v)


        #closeness_reward1 = sum([math.sqrt((transition[4] - transition[0]) ** 2 + (transition[5] - transition[1]) ** 2) for transition in transitions1]) / 1000
        #closeness_reward2 = sum([math.sqrt((transition[4] - transition[0]) ** 2 + (transition[5] - transition[1]) ** 2) for transition in transitions2]) / 1000
        closeness_rew1 = (sum([max(0, 1 - transition[4]) ** 2 for transition in transitions1]) / 10) * 0.1
        closeness_rew2 = (sum([max(0, 1 - transition[4]) ** 2 for transition in transitions2]) / 10) * 0.1

        moving_towards_goal_rew1 = (transitions1[0][4] - transitions1[-1][4]) * 0.7
        moving_towards_goal_rew2 = (transitions2[0][4] - transitions2[-1][4]) * 0.7

        wall_closeness_rew1 = -(sum([max(0, transition[1] - 0.8) for transition in transitions1]) / 10) * 0.1
        wall_closeness_rew1 += -(sum([max(0, abs(transition[0]) - 0.9) for transition in transitions1]) / 10) * 0.1
        wall_closeness_rew2 = -(sum([max(0, transition[1] - 0.8) for transition in transitions2]) / 10) * 0.1
        wall_closeness_rew2 += -(sum([max(0, abs(transition[0]) - 0.9) for transition in transitions2]) / 10) * 0.1

        char_closeness_rew1 = -min(0.8, sum([max(0, (1 - transition[11] - 0.8) * 5, (1 - transition[14] - 0.8) * 5) for transition in transitions1]) / 10 * 4)
        char_closeness_rew2 = -min(0.8, sum([max(0, (1 - transition[11] - 0.8) * 5, (1 - transition[14] - 0.8) * 5) for transition in transitions2]) / 10 * 4)

        #closeness_reward1 -= (sum([max(0, 1 - (transition[11] - 0.75)**2, 1 - (transition[16] - 0.75)**2) for transition in transitions1]) / 50) * 0.1
        #closeness_reward2 -= (sum([max(0, 1 - (transition[11] - 0.75)**2, 1 - (transition[16] - 0.75)**2) for transition in transitions2]) / 50) * 0.1

        total_reward_1 = segment1[-1] + closeness_rew1 + moving_towards_goal_rew1 + wall_closeness_rew1 + char_closeness_rew1
        total_reward_2 = segment2[-1] + closeness_rew2 + moving_towards_goal_rew2 + wall_closeness_rew2 + char_closeness_rew2
        truth_percentage = truth / 100.0
        fakes_percentage = 1 - truth_percentage
        epsilon = 0.05
        if segment1[-1] < -0.3 and segment2[-1] < -0.3:
            preference = [0, 0]
            point = [-1, -1]
        elif total_reward_1 > total_reward_2 + epsilon:
            preference = [1, 0] if fakes_percentage < random.random() else [0, 1]
            #point = critical_points[0] if preference[0] == 1 else critical_points[1]
            if preference[0] == 1:
                point = [critical_points[1][0], critical_points[0][1]]
            else:
                point = [critical_points[0][0], critical_points[1][1]]

        elif total_reward_1 + epsilon < total_reward_2:
            preference = [0, 1] if fakes_percentage < random.random() else [1, 0]
            #point = critical_points[1] if preference[1] == 1 else critical_points[0]
            if preference[1] == 1:
                point = [critical_points[0][0], critical_points[1][1]]
            else:
                point = [critical_points[1][0], critical_points[0][1]]
        else:
            preference = [0.5, 0.5]
            point = [-1, -1]

        if wandb.run is not None:
            wandb.log({"oracle/total": (total_reward_1 + total_reward_2) / 2,
                       "oracle/closeness": (closeness_rew1 + closeness_rew2) / 2,
                       "oracle/towards_goal": (moving_towards_goal_rew1 + moving_towards_goal_rew2) / 2,
                       "oracle/end_wall_closeness": (wall_closeness_rew1 + wall_closeness_rew1) / 2,
                       "oracle/char_closeness": (char_closeness_rew1 + char_closeness_rew2) / 2,
                       "oracle/true_rew": (segment1[-1] + segment2[-1]) / 2,
                       "oracle/pairs": self.pairs_size
                       })

        return [segment1, segment2, preference, point]

    def get_query_results_reward(self, segment1, segment2, truth, critical_points):
        total_reward_1 = segment1[-1]
        total_reward_2 = segment2[-1]
        truth_percentage = truth / 100.0
        fakes_percentage = 1 - truth_percentage
        epsilon = 1
        if total_reward_1 > total_reward_2 + epsilon:
            preference = [1, 0] if fakes_percentage < random.random() else [0, 1]
            # point = critical_points[0] if preference[0] == 1 else critical_points[1]
            if preference[0] == 1:
                point = [critical_points[1][0], critical_points[0][1]]
            else:
                point = [critical_points[0][0], critical_points[1][1]]
        elif total_reward_1 + epsilon < total_reward_2:
            preference = [0, 1] if fakes_percentage < random.random() else [1, 0]
            # point = critical_points[1] if preference[1] == 1 else critical_points[0]
            if preference[1] == 1:
                point = [critical_points[0][0], critical_points[1][1]]
            else:
                point = [critical_points[1][0], critical_points[0][1]]
        else:
            preference = [0.5, 0.5]
            point = [-1, -1]
        return [segment1, segment2, preference, point]
