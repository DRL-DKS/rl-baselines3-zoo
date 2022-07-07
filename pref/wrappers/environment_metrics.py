import collections
from abc import abstractmethod

import gym


def get_metric(env_name, agent):
    if env_name == 'LunarLanderContinuous-v2':
        return KeepRightMetric(agent)
    elif env_name == 'Pendulum-v1':
        return PendulumMetric(agent)
    elif env_name == 'Walker2d-v3':
        print("Walker2d metric")
        return WalkerMetric(agent)
    elif env_name == 'Hopper-v3':
        print("Hopper-v3 metric")
        return WalkerMetric(agent)
    elif env_name == 'window-open-v2':
        return PressButtonMetric(agent)
    elif env_name == 'unity-env':
        return SocialNavigationMetric(agent)
    else:
        return None


class MetricWrapper(gym.Wrapper):
    """
    Wrapper to log environment specific metrics.
    """
    def __init__(self, env, env_name, agent):
        super().__init__(env)
        self.env = env
        self.t = 0
        self.metric = get_metric(env_name, agent)
        self.running_average_reward = collections.deque(maxlen=10)
        self.episode_reward = 0

    def step(self, action):
        self.t += 1
        next_state, reward, done, info = self.env.step(action)

        self.episode_reward += reward

        if self.metric is not None: self.metric.update(next_state, done, info, action, reward)

        if done:
            self.running_average_reward.append(self.episode_reward)
            self.episode_reward = 0
            self.t = 0
        return next_state, reward, done, info

    def reset(self):
        state = self.env.reset()
        return state

    def observation(self, obs):
        return obs


class Metric:

    @abstractmethod
    def update(self, state, done, info, action, reward):
        pass

    def get_dict(self):
        pass

    def get_raw_data(self):
        pass


class SocialNavigationMetric(Metric):
    def __init__(self, agent):
        self.correct_count = 0
        self.wrong_count = 0
        self.running_average_ratios = collections.deque(maxlen=10)
        self.avoided_wrong_area_whole_ep_count = 0
        self.entered_wrong_area_ep_count = 0
        self.action_force = []
        self.running_average_force = collections.deque(maxlen=10)
        self.above_threshold = []
        self.above_threshold_average = collections.deque(maxlen=10)
        self.agent = agent


    def shift_interval(self, low=-1, high=1, min_threshold=0, max_threshold=1, x=None):
        return min_threshold + ((max_threshold - min_threshold) / (high - low)) * (x - low)


    def update(self, state, done, info=None, action=None, reward=0):
        norm_action = self.shift_interval(-1, 1, 0, 1, action[2])
        self.action_force.append(norm_action)

        if done:

            result = []
            average_force = sum(self.action_force) / len(self.action_force)
            self.running_average_force.append(average_force)
            result.append(average_force)
            self.agent.logger.record("rollout/ep_force_applied", average_force)
            average_force = sum(self.running_average_force) / len(self.running_average_force)
            self.agent.logger.record("rollout/ep_force_applied_mean", average_force)

            self.action_force = []
            self.correct_count = 0
            self.wrong_count = 0
            return result
        return []

    def get_dict(self):
        average_force = sum(self.running_average_force) / len(self.running_average_force)
        return [{'avg_force': str(average_force)}]

    def get_raw_data(self):
        average_force = sum(self.running_average_force) / len(self.running_average_force)
        return average_force, 0


class PressButtonMetric(Metric):
    def __init__(self, agent):
        self.running_average_ratios = collections.deque(maxlen=10)
        self.episode_ratios = []
        self.success = 0
        self.fail = 0
        self.agent = agent

    def update(self, state, done, info=None, action=None, reward=None):
        if done:
            result = []
            if info['success'] == 1.0:
                self.success += 1
            else:
                self.fail += 1

            ratio = self.success / (self.success + self.fail)
            self.running_average_ratios.append(ratio)
            ratio = sum(self.running_average_ratios) / len(self.running_average_ratios)
            result.append(ratio)
            self.agent.logger.record("rollout/ep_successful_rate", ratio)
            return result
        return []

    def get_dict(self):
        ratio = sum(self.running_average_ratios) / len(self.running_average_ratios)
        return [{'right_side_ratio': str(ratio)}]

    def get_raw_data(self):
        ratio = sum(self.running_average_ratios) / len(self.running_average_ratios)
        return ratio, 0


class WalkerMetric(Metric):
    def __init__(self, agent):
        self.correct_count = 0
        self.wrong_count = 0
        self.running_average_ratios = collections.deque(maxlen=10)
        self.avoided_wrong_area_whole_ep_count = 0
        self.entered_wrong_area_ep_count = 0
        self.z_coordinates = collections.deque(maxlen=10)
        self.running_average_z = collections.deque(maxlen=10)
        self.agent = agent

    def update(self, state, done, info=None, action=None, reward=None):
        if state[0] < 1.3:
            self.wrong_count += 1
        else:
            self.correct_count += 1

        self.z_coordinates.append(state[0])
        if done:
            result = []
            if self.wrong_count == 0:
                self.avoided_wrong_area_whole_ep_count += 1
            else:
                self.entered_wrong_area_ep_count += 1

            correct_ratio = self.correct_count / (self.correct_count + self.wrong_count)
            result.append(correct_ratio)  # complete avoidance
            self.agent.logger.record_mean("rollout/ep_correct_ratio_mean", correct_ratio)

            self.running_average_ratios.append(correct_ratio)
            correct_ratio = sum(self.running_average_ratios) / len(self.running_average_ratios)
            result.append(correct_ratio)
            self.agent.logger.record("rollout/ep_correct_ratio", correct_ratio)

            z_coord_average = sum(self.z_coordinates) / len(self.z_coordinates)
            self.running_average_z.append(z_coord_average)
            self.agent.logger.record("rollout/ep_avg_z", z_coord_average)
            z_coord_average = sum(self.running_average_z) / len(self.running_average_z)
            self.agent.logger.record("rollout/ep_avg_z_mean", z_coord_average)

            self.correct_count = 0
            self.wrong_count = 0
            return result
        return []

    def get_dict(self):
        correct_ratio = sum(self.running_average_ratios) / len(self.running_average_ratios)
        z_coord_average = sum(self.running_average_z) / len(self.running_average_z)
        return [{'correct_area_fraction_timesteps': str(correct_ratio)},
                {'avg_height': str(z_coord_average)}]

    def get_raw_data(self):
        correct_ratio = sum(self.running_average_ratios) / len(self.running_average_ratios)
        z_coord_average = sum(self.running_average_z) / len(self.running_average_z)
        return correct_ratio, z_coord_average


class PendulumMetric(Metric):
    def __init__(self, agent):
        self.correct_count = 0
        self.wrong_count = 0
        self.running_average_ratios = collections.deque(maxlen=10)
        self.avoided_wrong_area_whole_ep_count = 0
        self.entered_wrong_area_ep_count = 0
        self.agent = agent

    def update(self, state, done, info=None, action=None, reward=None):
        if 0.2 < state[1] < 0.3 and state[0] > 0:
            self.wrong_count += 1
        else:
            self.correct_count += 1

        if done:
            result = []
            if self.wrong_count == 0:
                self.avoided_wrong_area_whole_ep_count += 1
            else:
                self.entered_wrong_area_ep_count += 1
            correct_ratio = self.correct_count / (self.correct_count + self.wrong_count)
            result.append(correct_ratio)  # complete avoidance
            self.agent.logger.record_mean("rollout/ep_correct_ratio_nonmean", correct_ratio)
            self.running_average_ratios.append(correct_ratio)

            correct_ratio = sum(self.running_average_ratios) / len(self.running_average_ratios)
            result.append(correct_ratio)  # All timesteps
            self.agent.logger.record("rollout/ep_correct_ratio", correct_ratio)
            self.agent.logger.record("rollout/ep_successful", self.avoided_wrong_area_whole_ep_count)

            correct_area_whole_ep_ratio = self.avoided_wrong_area_whole_ep_count / (self.entered_wrong_area_ep_count + self.avoided_wrong_area_whole_ep_count)
            self.agent.logger.record("rollout/ep_successful_rate", correct_area_whole_ep_ratio)

            self.correct_count = 0
            self.wrong_count = 0
            return result
        return []

    def get_dict(self):
        correct_area_whole_ep_ratio = self.avoided_wrong_area_whole_ep_count / (
                    self.entered_wrong_area_ep_count + self.avoided_wrong_area_whole_ep_count)
        correct_ratio = sum(self.running_average_ratios) / len(self.running_average_ratios)
        return [{'correct_area_whole_ep_ratio': str(correct_area_whole_ep_ratio)},
                {'correct_area_fraction_timesteps': str(correct_ratio)}]

    def get_raw_data(self):
        correct_ratio = sum(self.running_average_ratios) / len(self.running_average_ratios)
        correct_area_whole_ep_ratio = self.avoided_wrong_area_whole_ep_count / (self.entered_wrong_area_ep_count + self.avoided_wrong_area_whole_ep_count)
        return correct_ratio, correct_area_whole_ep_ratio


class KeepRightMetric(Metric):
    def __init__(self, agent):
        self.right_side_count = 0
        self.left_side_count = 0
        self.running_average_ratios = collections.deque(maxlen=10)
        self.episode_ratios = []
        self.agent = agent

    def update(self, state, done, info=None, action=None, reward=None):
        if state[0] < 0:
            self.left_side_count += 1
        else:
            self.right_side_count += 1

        right_side_ratio = self.right_side_count / (self.right_side_count + self.left_side_count)

        if done:
            result = []

            right_side_ratio = sum(self.running_average_ratios) / len(self.running_average_ratios)
            result.append(right_side_ratio)

            self.agent.logger.record("rollout/ep_right_side_ratio", right_side_ratio)

            self.right_side_count = 0
            self.left_side_count = 0

            return result

    def get_dict(self):
        right_side_ratio = sum(self.running_average_ratios) / len(self.running_average_ratios)
        return [{'right_side_ratio': str(right_side_ratio)}]

    def get_raw_data(self):
        right_side_ratio = sum(self.running_average_ratios) / len(self.running_average_ratios)
        return right_side_ratio, 0