from abc import abstractmethod

import gym
import torch


class HumanReward(gym.Wrapper):
    """
    Environment wrapper to replace the env reward function with a human based reward function.
    In addition, it logs both the env and human reward per episode.
    """

    def __init__(self, env, human_model, logger):
        super().__init__(env)
        self.env = env
        self.current_state = []
        self.episode_reward_human = 0
        self.episode_true_reward = 0
        self.t = 0
        self.human_model = human_model
        self.logger = logger

    def step(self, action):
        observation = self.get_human_reward_observation(action)

        next_state, reward, done, info = self.env.step(action)
        self.episode_true_reward += reward

        reward_human = self.human_model.reward_model(observation)[0].detach().numpy().item()
        self.episode_reward_human += reward_human

        reward = reward_human
        self.current_state = next_state

        if done:
            self.logger.record("rollout/ep_human_rew", self.episode_reward_human)
            self.logger.record("rollout/ep_true_rew_mean", self.episode_true_reward)

            self.episode_reward_human = 0
            self.episode_true_reward = 0

        return next_state, reward, done, info

    def get_human_reward_observation(self, action):
        """
        Creates an observation by concatenating env observations with actions.
        The if handles edge cases that can show up depending on env/vectorization.
        :param action: action that the agent will take given current observation
        :return: observation (obs, act) to be used in a human reward model
        """
        action_tensor = torch.tensor(action)
        if len(action_tensor.shape) != 0:
            observation = torch.cat([torch.tensor(self.current_state), action_tensor])
        else:
            action_tensor = torch.tensor(action)
            action_tensor = torch.unsqueeze(action_tensor, 0)
            observation = torch.cat([torch.tensor(self.current_state), action_tensor])
        observation = torch.tensor(observation)
        observation = observation.type(torch.float32).unsqueeze(0)
        return observation

    def reset(self):
        state = self.env.reset()
        self.current_state = state

        return state

    def observation(self, obs):
        self.current_state = obs
        return obs


def get_metric(env_name, agent):
    if env_name == 'unity-env':
        return SocialNavigationMetric(agent)
    else:
        raise NotImplementedError(f"No metric implemented for environment {env_name}")


class MetricWrapper(gym.Wrapper):
    """
    Wrapper to log environment specific metrics.
    """
    def __init__(self, env, env_name, agent):
        super().__init__(env)
        self.env = env
        self.metric = get_metric(env_name, agent)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self.metric.update(next_state, done, info, action, reward)
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


class SocialNavigationMetric(Metric):
    """
    Metric for the social navigation unity environment.
    Logs the amount of force app    lied on average per episode.
    """
    def __init__(self, agent):
        self.action_force = []
        self.agent = agent

    def update(self, state, done, info=None, action=None, reward=0):
        norm_action = self.shift_interval(-1, 1, 0, 1, action[2])
        self.action_force.append(norm_action)

        if done:
            result = []
            average_force = sum(self.action_force) / len(self.action_force)
            self.agent.logger.record("rollout/ep_force_applied", average_force)

            self.action_force = []
            return result
        return []

    def shift_interval(self, low=-1, high=1, min_threshold=0, max_threshold=1, x=None):
        return min_threshold + ((max_threshold - min_threshold) / (high - low)) * (x - low)
