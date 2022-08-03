import collections

import gym
import numpy as np
import torch
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from stable_baselines3.common.callbacks import BaseCallback
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader



def train_reward_model(hc, training_epochs):
    o1, o2, prefs = hc.get_all_preference_pairs()
    # TODO: Change all these places to not convert from numpy to tensor...
    tensor_o1 = torch.Tensor(o1)
    tensor_o2 = torch.Tensor(o2)
    tensor_prefs = torch.Tensor(prefs)
    #tensor_critical_points = torch.Tensor(critical_points)
    my_dataset = TensorDataset(tensor_o1, tensor_o2, tensor_prefs)
    my_dataloader = DataLoader(my_dataset, batch_size=hc.batch_size, shuffle=True)
    hc.train_dataset(my_dataloader, None, training_epochs)


class UpdateRewardFunction(BaseCallback):
    """
    Callback to periodically gather query feedback and updates the reward model.

    :param hc: (HumanCritic) the human critic class to perform logging
    :param env_name: (string) Environment name being used
    :param n_queries: (int) Number of queries per update of the reward model
    :param initial_reward_estimation_epochs: (int) Number of queries to gather at initialization
    :param reward_training_epochs: (int) Number of epochs when training the reward model
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self,
                 hc,
                 env_name='LunarLanderContinuous-v2',
                 n_queries=10,
                 initial_reward_estimation_epochs=200,
                 reward_training_epochs=50,
                 truth=90,
                 traj_length=50,
                 smallest_rew_threshold=0,
                 largest_rew_threshold=0,
                 n_initial_queries=200,
                 max_queries=1400,
                 verbose=0,
                 seed=12345
                 ):
        super(UpdateRewardFunction, self).__init__(verbose)
        self.n_queries = n_queries
        self.hc = hc
        self.initial_reward_estimation_epochs = initial_reward_estimation_epochs
        self.env_name = env_name
        self.reward_training_epochs = reward_training_epochs
        self.truth = truth
        self.traj_length = traj_length
        self.smallest_rew_threshold = smallest_rew_threshold
        self.largest_rew_threshold = largest_rew_threshold
        self.n_initial_queries = n_initial_queries
        self.seed = seed
        self.max_queries = max_queries

    def _on_training_start(self) -> None:
        print("Performing initial training of reward model")
        # Train reward model (How much?)
        self.hc.writer = SummaryWriter(self.logger.get_dir())
        traj_to_collect = self.n_initial_queries * 5
        trajectories, critical_points = self.collect_segments_with_critical_points(self.model, 100000, traj_to_collect)
        trajectories = self.hc.segments[:self.hc.segments_size]

        self.hc.generate_preference_pairs(trajectories, truth=self.truth, number_of_queries=self.n_initial_queries)

        self.train_reward_model(self.initial_reward_estimation_epochs)

        self.hc.save_reward_model(self.env_name + "-loop2")

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:

        self.model.save(self.env_name + "-loop2")
        if self.hc.pairs_size < self.max_queries:  # TODO: remove plz or make into argument

            traj_to_collect = self.n_queries * 5
            trajectories, critical_points = self.collect_segments_with_critical_points(self.model, 100000, traj_to_collect)
            trajectories = self.hc.segments[(self.hc.segments_size - traj_to_collect * 2):self.hc.segments_size]

            self.hc.generate_preference_pairs(trajectories, truth=self.truth, number_of_queries=self.n_queries)

            self.train_reward_model(self.reward_training_epochs)
            self.hc.save_reward_model(self.env_name + "-loop2")

        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass

    def train_reward_model(self, training_epochs):
        o1, o2, prefs = self.hc.get_all_preference_pairs()
        # TODO: Change all these places to not convert from numpy to tensor...
        tensor_o1 = torch.Tensor(o1)
        tensor_o2 = torch.Tensor(o2)
        tensor_prefs = torch.Tensor(prefs)
        # tensor_critical_points = torch.Tensor(critical_points)
        my_dataset = TensorDataset(tensor_o1, tensor_o2, tensor_prefs)
        my_dataloader = DataLoader(my_dataset, batch_size=self.hc.batch_size, shuffle=True)
        self.hc.train_dataset(my_dataloader, None, training_epochs)

    def process_traj_segment(self, traj_segment, segment_reward, done, traj_k_lenght=25):
        if len(traj_segment) < traj_k_lenght and done:
            while len(
                    traj_segment) < traj_k_lenght:  # TODO adding last step until we complete traj... we did this because of tensors
                traj_segment.append(traj_segment[-1])
        self.hc.add_segment(traj_segment, segment_reward)

    def collect_segments(self, model, env, test_episodes=5000, n_collect_segments=0):  # evaluate
        total_segments = []
        for e in range(test_episodes):
            obs = env.reset()
            done = False
            score = 0
            traj_segment = []
            segment_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=False)  # TODO test deterministic and non-deterministic
                obs, reward, done, _ = env.step(action)

                segment_reward += reward
                score += reward
                action_shape = 1 if len(action.shape) == 0 else action.shape[0]
                action = np.resize(action, (action_shape,))
                traj_segment.append(np.concatenate((obs.squeeze(), action)))

                if len(traj_segment) == self.traj_length or done:
                    self.process_traj_segment(traj_segment, segment_reward, done, self.traj_length)
                    total_segments.append([traj_segment, segment_reward])
                    traj_segment, segment_reward = [], 0

                    if len(total_segments) % (n_collect_segments // 10) == 0:
                        print("Collected segments: " + str(len(total_segments)) + "/" + str(n_collect_segments))

                if len(total_segments) >= n_collect_segments:
                    env.close()
                    return total_segments

        env.close()
        return total_segments

    def update_params(self,
                      n_queries=10,
                      initial_reward_estimation_epochs=200,
                      reward_training_epochs=50,
                      traj_length=50,
                      n_initial_queries=200,
                      max_queries=1400,
                      ):
        self.n_queries = n_queries
        self.initial_reward_estimation_epochs = initial_reward_estimation_epochs
        self.reward_training_epochs = reward_training_epochs
        self.traj_length = traj_length
        self.n_initial_queries = n_initial_queries
        self.max_queries = max_queries

    def collect_segments_with_critical_points(self, model, test_episodes=5000, n_collect_segments=0, extra_env=None):  # evaluate
        total_segments = []
        critical_points = []

        if self.env_name == "Social-Nav-v1":
            channel = EngineConfigurationChannel()
            unity_env = UnityEnvironment('./envs/socialnav_supersimple6/socialnav1', side_channels=[channel], worker_id=42, no_graphics=True)
            #unity_env = UnityEnvironment('./envs/fixedsocialnav/socialnav1', side_channels=[channel], worker_id=42, no_graphics=True)
            channel.set_configuration_parameters(time_scale=30.0)
            env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=False)
        else:
            env = gym.make(self.env_name)
        env.seed(self.seed)
        print(self.env_name)

        for e in range(test_episodes):
            largest_rew = self.largest_rew_threshold
            largest_rew_index = -1
            smallest_rew = self.smallest_rew_threshold
            smallest_rew_index = -1

            running_average_rew = collections.deque(maxlen=3)
            latest_indexes = collections.deque(maxlen=3)

            obs = env.reset()
            done = False
            score = 0
            traj_segment = []
            segment_reward = 0

            while not done:
                action, _states = model.predict(obs, deterministic=False)
                obs, reward, done, _ = env.step(action)

                segment_reward += reward
                score += reward
                action = np.resize(action, (action.shape[0], ))
                #action = np.array((action,))

                running_average_rew.append(reward)
                running_average_rew_total = sum(running_average_rew) / len(running_average_rew)
                latest_indexes.append(len(traj_segment))

                # Critical point extension
                if largest_rew < running_average_rew_total:
                    largest_rew_index = latest_indexes[np.argmax(running_average_rew)]
                    largest_rew = running_average_rew_total

                if smallest_rew > running_average_rew_total:
                    smallest_rew_index = latest_indexes[np.argmin(running_average_rew)]
                    smallest_rew = running_average_rew_total

                traj_segment.append(np.concatenate((obs.squeeze(), action)))

                if len(traj_segment) == self.traj_length or done:
                    self.process_traj_segment(traj_segment, segment_reward, done, self.traj_length)
                    total_segments.append([traj_segment, segment_reward])
                    traj_segment, segment_reward = [], 0

                    # Critical point extension
                    if smallest_rew < self.smallest_rew_threshold:
                        self.hc.punishments_given += 1
                    else:
                        smallest_rew_index = -1

                    if largest_rew > self.largest_rew_threshold:
                        self.hc.approvements_given += 1
                    else:
                        largest_rew_index = -1

                    self.hc.add_critical_points(smallest_rew_index, largest_rew_index)
                    critical_points.append([smallest_rew_index, largest_rew_index])

                    largest_rew = self.largest_rew_threshold
                    largest_rew_index = -1
                    smallest_rew = self.smallest_rew_threshold
                    smallest_rew_index = -1

                    if len(total_segments) % (n_collect_segments // 10) == 0:
                        print("Collected segments: " + str(len(total_segments)) + "/" + str(n_collect_segments))

                if len(total_segments) >= n_collect_segments != 0:
                    env.close()
                    return total_segments, critical_points
        env.close()
        return total_segments, critical_points


class UpdateRewardFunctionCriticalPoint(BaseCallback):
    """
    Callback to periodically gather query feedback and updates the reward model.

    :param hc: (HumanCritic) the human critic class to perform logging
    :param env_name: (string) Environment name being used
    :param n_queries: (int) Number of queries per update of the reward model
    :param initial_reward_estimation_epochs: (int) Number of queries to gather at initialization
    :param reward_training_epochs: (int) Number of epochs when training the reward model
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self,
                 hc,
                 env_name='LunarLanderContinuous-v2',
                 n_queries=10,
                 initial_reward_estimation_epochs=200,
                 reward_training_epochs=50,
                 truth=90,
                 traj_length=50,
                 smallest_rew_threshold=0,
                 largest_rew_threshold=0,
                 n_initial_queries=200,
                 max_queries=1400,
                 verbose=0,
                 seed=12345,
                 workerid=-1
                 ):
        super(UpdateRewardFunctionCriticalPoint, self).__init__(verbose)
        self.n_queries = n_queries
        self.hc = hc
        self.initial_reward_estimation_epochs = initial_reward_estimation_epochs
        self.env_name = env_name
        self.reward_training_epochs = reward_training_epochs
        self.truth = truth
        self.traj_length = traj_length
        self.smallest_rew_threshold = smallest_rew_threshold
        self.largest_rew_threshold = largest_rew_threshold
        self.n_initial_queries = n_initial_queries
        self.seed = seed
        self.max_queries = max_queries
        self.workerid = workerid

    def _on_training_start(self) -> None:
        print("Performing initial training of reward model")
        # Train reward model (How much?)
        self.hc.writer = SummaryWriter(self.logger.get_dir())
        traj_to_collect = self.n_initial_queries * 5
        trajectories, critical_points = self.collect_segments_with_critical_points(self.model, 100000, traj_to_collect)
        trajectories = self.hc.segments[:self.hc.segments_size]
        critical_points = self.hc.critical_points[:self.hc.critical_points_size]

        self.hc.generate_preference_pairs_with_critical_points(trajectories, critical_points, truth=self.truth, number_of_queries=self.n_initial_queries)

        self.train_reward_model(self.initial_reward_estimation_epochs)

        self.hc.save_reward_model(self.env_name + "-loop2")

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:

        self.model.save(self.env_name + "-loop2")
        if self.hc.pairs_size < self.max_queries:  # TODO: remove plz or make into argument

            traj_to_collect = self.n_queries * 5
            trajectories, critical_points = self.collect_segments_with_critical_points(self.model, 100000, traj_to_collect)
            #trajectories = self.hc.segments[:self.hc.segments_size]
            trajectories = self.hc.segments[max(0, (self.hc.segments_size - (traj_to_collect * 5))):self.hc.segments_size]
            critical_points = self.hc.critical_points[max(0, (self.hc.segments_size - (traj_to_collect * 5))):self.hc.segments_size]
            #critical_points = self.hc.critical_points[:self.hc.critical_points_size]

            self.hc.generate_preference_pairs_with_critical_points(trajectories, critical_points, truth=self.truth, number_of_queries=self.n_queries)

            self.train_reward_model(self.reward_training_epochs)
            self.hc.save_reward_model(self.env_name + "-loop2")

        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass

    def train_reward_model(self, training_epochs):
        o1, o2, prefs, critical_points = self.hc.get_all_preference_pairs_with_critical_points()
        # TODO: Change all these places to not convert from numpy to tensor...
        tensor_o1 = torch.Tensor(o1)
        tensor_o2 = torch.Tensor(o2)
        tensor_prefs = torch.Tensor(prefs)
        tensor_critical_points = torch.Tensor(critical_points)
        my_dataset = TensorDataset(tensor_o1, tensor_o2, tensor_prefs, tensor_critical_points)
        my_dataloader = DataLoader(my_dataset, batch_size=self.hc.batch_size, shuffle=True)
        self.hc.train_dataset_with_critical_points(my_dataloader, None, training_epochs)

    def process_traj_segment(self, traj_segment, segment_reward, done, traj_k_lenght=25):
        if len(traj_segment) < traj_k_lenght and done:
            while len(
                    traj_segment) < traj_k_lenght:  # TODO adding last step until we complete traj... we did this because of tensors
                traj_segment.append(traj_segment[-1])
        self.hc.add_segment(traj_segment, segment_reward)

    def collect_segments(self, model, env, test_episodes=5000, n_collect_segments=0):  # evaluate
        total_segments = []
        for e in range(test_episodes):
            obs = env.reset()
            done = False
            score = 0
            traj_segment = []
            segment_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=False)  # TODO test deterministic and non-deterministic
                obs, reward, done, _ = env.step(action)

                segment_reward += reward
                score += reward
                action_shape = 1 if len(action.shape) == 0 else action.shape[0]
                action = np.resize(action, (action_shape,))
                traj_segment.append(np.concatenate((obs.squeeze(), action)))

                if len(traj_segment) == self.traj_length or done:
                    self.process_traj_segment(traj_segment, segment_reward, done, self.traj_length)
                    total_segments.append([traj_segment, segment_reward])
                    traj_segment, segment_reward = [], 0

                    if len(total_segments) % (n_collect_segments // 10) == 0:
                        print("Collected segments: " + str(len(total_segments)) + "/" + str(n_collect_segments))

                if len(total_segments) >= n_collect_segments:
                    env.close()
                    return total_segments

        env.close()
        return total_segments

    def update_params(self,
                      n_queries=10,
                      initial_reward_estimation_epochs=200,
                      reward_training_epochs=50,
                      traj_length=50,
                      n_initial_queries=200,
                      max_queries=1400,
                      ):
        self.n_queries = n_queries
        self.initial_reward_estimation_epochs = initial_reward_estimation_epochs
        self.reward_training_epochs = reward_training_epochs
        self.traj_length = traj_length
        self.n_initial_queries = n_initial_queries
        self.max_queries = max_queries

    def collect_segments_with_critical_points(self, model, test_episodes=5000, n_collect_segments=0, extra_env=None):  # evaluate
        total_segments = []
        critical_points = []

        if self.env_name == "Social-Nav-v1":
            channel = EngineConfigurationChannel()
            #unity_env = UnityEnvironment('./envs/fixedsocialnav/socialnav1', side_channels=[channel], worker_id=42, no_graphics=True)
            workerid = 42
            if self.workerid != -1:
                workerid = 43
            unity_env = UnityEnvironment('./envs/socialnav_supersimple6/socialnav1', side_channels=[channel], worker_id=workerid, no_graphics=True)
            channel.set_configuration_parameters(time_scale=30.0)
            env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=False)
            obs = env.reset()
        else:
            env = gym.make(self.env_name)
        env.seed(self.seed)

        for e in range(test_episodes):
            largest_rew = self.largest_rew_threshold
            largest_rew_index = -1
            smallest_rew = self.smallest_rew_threshold
            smallest_rew_index = -1

            running_average_rew = collections.deque(maxlen=3)
            latest_indexes = collections.deque(maxlen=3)

            obs = env.reset()
            done = False
            score = 0
            traj_segment = []
            segment_reward = 0

            while not done:
                action, _states = model.predict(obs, deterministic=False)
                obs, reward, done, _ = env.step(action)

                segment_reward += reward
                score += reward
                action = np.resize(action, (action.shape[0], ))

                running_average_rew.append(reward)
                running_average_rew_total = sum(running_average_rew) / len(running_average_rew)
                latest_indexes.append(len(traj_segment))

                # Critical point extension
                if largest_rew < running_average_rew_total:
                    largest_rew_index = latest_indexes[np.argmax(running_average_rew)]
                    largest_rew = running_average_rew_total

                if smallest_rew > running_average_rew_total:
                    smallest_rew_index = latest_indexes[np.argmin(running_average_rew)]
                    smallest_rew = running_average_rew_total

                traj_segment.append(np.concatenate((obs.squeeze(), action)))

                if len(traj_segment) == self.traj_length or done:
                    self.process_traj_segment(traj_segment, segment_reward, done, self.traj_length)
                    total_segments.append([traj_segment, segment_reward])
                    traj_segment, segment_reward = [], 0

                    # Critical point extension
                    if smallest_rew < self.smallest_rew_threshold:
                        self.hc.punishments_given += 1
                    else:
                        smallest_rew_index = -1

                    if largest_rew > self.largest_rew_threshold:
                        self.hc.approvements_given += 1
                    else:
                        largest_rew_index = -1

                    self.hc.add_critical_points(smallest_rew_index, largest_rew_index)
                    critical_points.append([smallest_rew_index, largest_rew_index])

                    largest_rew = self.largest_rew_threshold
                    largest_rew_index = -1
                    smallest_rew = self.smallest_rew_threshold
                    smallest_rew_index = -1

                    if len(total_segments) % (n_collect_segments // 10) == 0:
                        print("Collected segments: " + str(len(total_segments)) + "/" + str(n_collect_segments))

                if len(total_segments) >= n_collect_segments != 0:
                    env.close()
                    return total_segments, critical_points
        env.close()
        return total_segments, critical_points