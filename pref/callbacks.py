import collections
import os
import random
import time

import gym
import numpy as np
import torch
from gym.wrappers import RecordVideo
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from stable_baselines3.common.callbacks import BaseCallback
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader

from pref.query_database import QueryDatabase


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
        self.no_improvements_count = 0
        self.collect_queries = True
        self.workerid = workerid
        print("truth:")
        print(self.truth)

    def _on_training_start(self) -> None:
        print("Performing initial training of reward model")
        # Train reward model (How much?)
        self.hc.writer = SummaryWriter(self.logger.get_dir())
        traj_to_collect = self.n_initial_queries * 5
        trajectories, critical_points = self.collect_segments_with_critical_points(self.model, 100000, traj_to_collect)
        trajectories = self.hc.segments[:self.hc.segments_size]
        critical_points = self.hc.critical_points[:self.hc.critical_points_size]

        self.hc.generate_preference_pairs_with_critical_points(trajectories, critical_points, truth=self.truth, number_of_queries=self.n_initial_queries)
        #self.hc.generate_preference_pairs_information_based(trajectories, critical_points, truth=self.truth, number_of_queries=self.n_initial_queries, uncertain_ratio=0.5)

        self.train_reward_model(self.initial_reward_estimation_epochs)

        self.hc.save_reward_model(self.env_name + "-loop2")

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:

        self.model.save(self.env_name + "-loop2")
        if self.hc.pairs_size < self.max_queries and (self.collect_queries or True):  # TODO: remove plz or make into argument

            traj_to_collect = self.n_queries * 5
            trajectories, critical_points = self.collect_segments_with_critical_points(self.model, 100000, traj_to_collect)
            #trajectories = self.hc.segments[:self.hc.segments_size]
            trajectories = self.hc.segments[max(0, (self.hc.segments_size - (traj_to_collect * 2))):self.hc.segments_size]
            critical_points = self.hc.critical_points[max(0, (self.hc.segments_size - (traj_to_collect * 2))):self.hc.segments_size]
            #critical_points = self.hc.critical_points[:self.hc.critical_points_size]

            self.hc.generate_preference_pairs_with_critical_points(trajectories, critical_points, truth=self.truth, number_of_queries=self.n_queries)
            #self.hc.generate_preference_pairs_information_based(trajectories, critical_points, truth=self.truth, number_of_queries=self.n_queries, uncertain_ratio=0.1, type="max")

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
        meta_data = self.hc.train_dataset_with_critical_points(my_dataloader, None, training_epochs)

        if meta_data["improved"]:
            self.no_improvements_count = max(0, self.no_improvements_count - 1)
        else:
            self.no_improvements_count += 1
        if self.no_improvements_count >= 3:
            self.collect_queries = False


    def process_traj_segment(self, traj_segment, segment_reward, done, traj_k_lenght=25):
        if len(traj_segment) < traj_k_lenght and done:
            while len(
                    traj_segment) < traj_k_lenght:  # TODO adding last step until we complete traj... we did this because of tensors
                traj_segment.append(traj_segment[-1])
        self.hc.add_segment(traj_segment, segment_reward)

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
            workerid = 42
            if self.workerid != -1:
                workerid = 43
            unity_env = UnityEnvironment('./envs/socialnav_supersimple6/socialnav1', side_channels=[channel], worker_id=workerid, no_graphics=True)
            channel.set_configuration_parameters(time_scale=30.0)
            env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=False)
            env.reset()
        else:
            env = gym.make(self.env_name)
        env.seed(self.seed)

        score = 0
        traj_segment = []
        segment_reward = 0
        for e in range(test_episodes):
            largest_rew = self.largest_rew_threshold
            largest_rew_index = -1
            smallest_rew = self.smallest_rew_threshold
            smallest_rew_index = -1

            running_average_rew = collections.deque(maxlen=3)
            latest_indexes = collections.deque(maxlen=3)

            obs = env.reset()
            done = False


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


class UpdateRewardFunctionRealHuman(BaseCallback):
    """
    A callback that helps gather queries from the user and updates the reward model periodically.
    :param hc: (HumanCritic) the human critic class to perform logging
    :param agent: (BaseAlgorithm) Stablebaseline agent
    :param videos_col: (Collection) Mongodb collection
    :param env_name: (string) Environment name being used
    :param video_location: (string) Path to where the video clips should be saved (Frontend media folder)
    :param n_initial_queries: (string) Initial amount of queries to give the user
    :param initial_reward_estimation_epochs: (int) Number of queries to gather at initialization
    :param n_queries: (int) Number of queries per update of the reward model
    :param traj_lenght: (int) Length of trajectories
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self,
                 hc,
                 agent,
                 env_name='LunarLanderContinuous-v2',
                 video_location='./UI/preflearn/public/media/',
                 n_initial_queries=200,
                 initial_reward_estimaton_epochs=200,
                 n_queries=40,
                 traj_length=50,
                 verbose=0,
                 workerid=-1,
                 recording_channel=None):
        super(UpdateRewardFunctionRealHuman, self).__init__(verbose)
        self.n_queries = n_queries
        self.n_init_queries = n_initial_queries
        self.agent = agent
        self.hc = hc
        self.env_name = env_name
        self.video_location = video_location
        self.initial_reward_estimaton_epochs = initial_reward_estimaton_epochs
        self.videos_col = QueryDatabase(video_location=video_location)
        self.traj_length = traj_length
        self.workerid = workerid
        self.recording_channel = recording_channel

    def _on_training_start(self) -> None:
        self.hc.writer = SummaryWriter(self.agent.logger.get_dir())
        print("Performing initial training of reward model")

        self.make_reward_update(self.n_init_queries, self.initial_reward_estimaton_epochs)

    def delete_video_folder(self):
        for file in os.listdir(self.video_location):
            if file.endswith(".mp4") or file.endswith(".json"):
                os.remove(self.video_location + file)

    def make_reward_update(self, n_queries=10, reward_training_epochs=50):
        # TODO, how to store things?
        self.videos_col.clear_database_folder()
        self.delete_video_folder()

        # Generate segments
        segments, video_names = self.collect_segments(self.agent, n_queries * 4)

        # Create queries
        for i in range(n_queries):  # self.n_queries * 4):
            segment_pair, video_pair, indexes = self.hc.random_sample_batch_segments_and_videos(segments, video_names, number_of_sampled_segments=2)
            self.videos_col.insert_one_query(video_pair, indexes)

        rated = 0
        while rated < n_queries:  # self.n_queries:
            rated = self.videos_col.get_number_of_rated_queries()
            time.sleep(5)

        self.add_pairs_to_buffer()
        self.train_reward_model(reward_training_epochs)
        self.hc.save_reward_model(self.env_name + "-loop")

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        self.agent.save(self.env_name + "-loop")
        if self.hc.pairs_size < 2100:
            self.make_reward_update(self.n_queries, self.hc.training_epochs)
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass

    def add_pairs_to_buffer(self):
        allQueries = self.videos_col.get_all_queries()
        for q in allQueries:
            index1 = q["index1"]
            index2 = q["index2"]
            segment1 = self.hc.segments[index1]
            segment2 = self.hc.segments[index2]

            preftmp = ["preference"]
            pref = [1, 0] if preftmp == 1 else [0, 1]  # TODO handle neither
            self.hc.add_pairs(segment1, segment2, pref)

    def collect_segments(self, model, n_queries=5, recording=False):  # evaluate
        def start_recording(agent_timestep):
            if (agent_timestep == trigger) or (agent_timestep == trigger):
                print("Started recording: " + str(agent_timestep))
                return True
            else:
                if agent_timestep == trigger + self.traj_length + 1 or agent_timestep == trigger + self.traj_length + 1:
                    print("Finished recording: " + str(agent_timestep - 1))
                return False

        timestep = 0
        total_segments, triggers, traj_segment, names = [], [], [], []
        video_prefix = "collect"

        trigger = random.randint(0, 1000)
        triggers.append(trigger)

        if self.env_name == "Social-Nav-v1":
            channel = EngineConfigurationChannel()
            workerid = 42
            if self.workerid != -1:
                workerid = 43
            unity_env = UnityEnvironment('./envs/socialnav_supersimple6/socialnav1', side_channels=[channel],
                                         worker_id=workerid, no_graphics=True)
            channel.set_configuration_parameters(time_scale=30.0)
            collecting_env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=False)
            collecting_env.reset()
        else:
            collecting_env = gym.make(self.env_name)

        if recording:
            collecting_env = RecordVideo(collecting_env, self.video_location, step_trigger=start_recording, video_length=self.traj_length, name_prefix=video_prefix)

        while True:
            obs = collecting_env.reset()
            done = False

            while not done:
                action, _states = model.predict(obs)
                obs, reward, done, _ = collecting_env.step(action)

                if trigger <= timestep < trigger + self.traj_length:
                    if timestep == trigger:
                        print("Starting to collect: " + str(timestep))
                        if not recording:
                            self.recording_channel.send_string("record")

                    action = np.resize(action, (action.shape[0], ))
                    traj_segment.append(np.concatenate((obs.squeeze(), action)))

                if len(traj_segment) == self.traj_length:
                    total_segments.append(traj_segment)
                    self.hc.add_segment_human(traj_segment)
                    traj_segment, segment_reward = [], 0

                    if not recording:
                        self.recording_channel.send_string(video_prefix + "-" + str(trigger))

                    if len(total_segments) >= n_queries:
                        if recording:
                            collecting_env.close_video_recorder()
                        collecting_env.close()

                        if recording:
                            names = [video_prefix + "-step-" + str(tr) + ".mp4" for tr in triggers]
                        else:
                            names = [video_prefix + "-" + str(tr) for tr in triggers]

                        return total_segments, names

                    start = trigger + self.traj_length
                    end = trigger + self.traj_length + 1000
                    trigger = random.randint(start, end)
                    triggers.append(trigger)
                timestep += 1

    def train_reward_model(self, training_epochs):
        o1, o2, prefs = self.hc.get_all_preference_pairs_human()
        # TODO: Change all these places to not convert from numpy to tensor...
        tensor_o1 = torch.Tensor(o1)
        tensor_o1 = tensor_o1.squeeze()
        tensor_o2 = torch.Tensor(o2)
        tensor_o2 = tensor_o2.squeeze()
        tensor_prefs = torch.Tensor(prefs)

        my_dataset = TensorDataset(tensor_o1, tensor_o2, tensor_prefs)
        my_dataloader = DataLoader(my_dataset, batch_size=self.hc.batch_size, shuffle=True)
        self.hc.train_dataset(my_dataloader, None, training_epochs)