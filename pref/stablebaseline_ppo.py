import collections
import os.path
import random

import numpy.random
import yaml

import gym
from gym.wrappers import NormalizeReward
import torch
import numpy as np

from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_linear_fn

from pref.utils import utils
from pref.utils.parse_arguments import parse_args_ppo
from pref.callbacks.HelperCallbacks import UpdateRewardFunctionRealHuman, InitializeRewardModel, UpdateRewardFunction, \
    UpdateRewardFunction2
from pref.wrappers.environment_metrics import MetricWrapper, get_metric
from pref.wrappers.human_reward import HumanReward
from stablebaseline_oracle import HumanCritic
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps, EvalCallback

torch.cuda.empty_cache()


def collect_segments(model, env, test_episodes=5000, n_collect_segments=0, deterministic=False):  # evaluate
    total_segments = []
    for e in range(test_episodes):
        obs = env.reset()
        done = False
        score = 0
        traj_segment = []
        segment_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)  # TODO test deterministic and non-deterministic
            obs, reward, done, _ = env.step(action)

            segment_reward += reward
            score += reward
            action_shape = 1 if len(action.shape) == 0 else action.shape[0]
            action = np.resize(action, (action_shape, ))
            traj_segment.append(np.concatenate((obs.squeeze(), action)))

            if len(traj_segment) == traj_k_lenght or done:
                process_traj_segment(traj_segment, segment_reward, done, traj_k_lenght)
                total_segments.append([traj_segment, segment_reward])
                traj_segment, segment_reward = [], 0

                if len(total_segments) % (n_collect_segments // 10) == 0:
                    print("Collected segments: " + str(len(total_segments)) + "/" + str(n_collect_segments))

            if len(total_segments) >= n_collect_segments:
                env.close()
                return total_segments

    env.close()
    return total_segments


def test_agent(agent, test_episodes=10, render=False):  # evaluate

    def start_recording(agent_timestep):
        if (agent_timestep == 1):
            print("Started recording: " + str(agent_timestep))
            return True
        else:
            return False

    scores = []
    episodes = []
    average = []

    if env_name == 'unity-env':
        channel = EngineConfigurationChannel()
        unity_env = UnityEnvironment('envs/snappy_rays', side_channels=[channel], worker_id=1, seed=101)
        channel.set_configuration_parameters(time_scale=1.0)
        collecting_env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=False, action_space_seed=101)
    else:
        collecting_env = gym.make(env_name, **env_hyperparameters)
        collecting_env.seed(42)
    #collecting_env = RecordVideo(collecting_env, "video/", episode_trigger=start_recording, video_length=1000)
    #metric = get_metric(env_name, agent)

    for e in range(20):
        obs = collecting_env.reset()

        done = False
        #metric.update(obs, done)
        score = 0
        segment_reward = 0
        agent.policy.set_training_mode(False)
        t = 0
        while not done:
            if True:
                collecting_env.render()
            action, _states = agent.predict(obs, deterministic=True)
            t += 1

            obs, reward, done, _ = collecting_env.step(action)
            segment_reward += reward
            score += reward
            # result = metric.update(obs, done, action=action, reward=reward)
            if done:
                scores.append(score)
                episodes.append(e)
                average.append(sum(scores[-50:]) / len(scores[-50:]))
                print("episode: {}/{}, score: {}, average {}".format(e, test_episodes, score, average[-1]))
                #print(result)
                break
    collecting_env.close()


def evaluate_agent(model, test_episodes=10):  # evaluate
    if env_name == 'unity-env':
        channel = EngineConfigurationChannel()
        unity_env = UnityEnvironment('envs/snappy_rays', side_channels=[channel], worker_id=5)
        channel.set_configuration_parameters(time_scale=20.0)
        collecting_env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=False)
    else:
        collecting_env = gym.make(env_name, **env_hyperparameters)
    scores = []
    episodes = []
    average = []
    metric = get_metric(env_name, model)

    for e in range(test_episodes):
        obs = collecting_env.reset()
        done = False
        score = 0
        segment_reward = 0

        while not done:

            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _ = collecting_env.step(action)

            result = metric.update(obs, done, action=action)

            segment_reward += reward
            score += reward

            if done:
                scores.append(score)
                episodes.append(e)
                average.append(sum(scores) / len(scores))
                print("episode: {}/{}, score: {}, average {}".format(e, test_episodes, score, average[-1]))
                print(result)
                break

    metric_dict = metric.get_dict()
    dict_file = [{
        'mean_true_reward': str(average[-1])
    }, {
        'torch_seed': str(torch.seed())
    }, {
        'numpy_seed': str(numpy.random.get_state())
    }, {
        'python_seed': str(random.getstate())
    }, metric_dict]
    with open(working_path + "ratio" + str(human_reward_ratio) + '_' + str(args.index) + '.yaml', 'w') as file:
        documents = yaml.dump(dict_file, file)
    env.close()


def process_traj_segment(traj_segment, segment_reward, done, traj_k_lenght=25):
    if len(traj_segment) < traj_k_lenght and done:
        while len(traj_segment) < traj_k_lenght:  # TODO adding last step until we complete traj... we did this because of tensors
            traj_segment.append(traj_segment[-1])
    hc.add_segment(traj_segment, segment_reward)


def load_agent_vec_env(env_name, tensorboard_log, model_name, env_hyperparameters, hyperparameters, wrapper_class=None):
    env = make_vec_env(env_name, wrapper_class=wrapper_class, **env_hyperparameters)
    agent = PPO.load(model_name, env=env, tensorboard_log=tensorboard_log, verbose=1, **hyperparameters)
    agent.load(model_name)
    return agent, env


def create_agent_vec_env(env_name, tensorboard_log, env_hyperparameters, hyperparameters, wrapper_class=None):
    env = make_vec_env(env_name, wrapper_class=wrapper_class, **env_hyperparameters)
    agent = PPO(env=env, tensorboard_log=tensorboard_log, verbose=1, **hyperparameters)
    return agent, env


def expert_trajectory_preference_learning():
    global buffer_path, expert_model_name, hc, agent, env, checkpoint_callback
    global policy_name_load, policy_name_save, reward_model_load, reward_model_save

    policy_name_load_file_path = utils.find_model_name(args.policy_name_load, env_path)

    expert_model_name = env_name + "-Expert"
    log_name = working_path + "prefs=" + str(init_prefs_n) + "_epochs=" + str(initial_reward_estimaton_epochs) \
               + "_ratio=" + str(args.human_reward_ratio)

    learn_hyperparameters['total_timesteps'] = int(args.final_timesteps)

    print("Collecting segments and queries")
    hc.load_buffers(path=buffer_path)

    agent, env = load_agent_vec_env(env_name, tensorboard_log, policy_name_load_file_path, env_hyperparameters, hyperparameters, HumanReward)

    initialization_callback = InitializeRewardModel(hc,
                                                    agent,
                                                    meta_data,
                                                    init_prefs_n=init_prefs_n,
                                                    reward_model_save=working_path + reward_model_save,
                                                    initial_training_epochs=initial_reward_estimaton_epochs,
                                                    truth=truth)

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=checkpoints_path, name_prefix=policy_name_save)

    eval_env = gym.make(env_name)
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=20000, n_eval_episodes=5, deterministic=True, render=False)

    callbacks = [initialization_callback, checkpoint_callback, eval_callback]

    agent.set_env(env)
    agent.learn(tb_log_name=log_name, callback=callbacks, **learn_hyperparameters)
    agent.save(working_path + policy_name_save)

    np.savetxt(working_path + 'scores' + str(args.human_reward_ratio) + '.csv', [p for p in zip(total_reward, total_correct_ratio, total_success_ratio)], delimiter=',', fmt='%s')
    evaluate_agent(agent, test_episodes=100)


def train_with_oracle_preferences():
    global env, collecting_env, y_roof, agent, event_callback

    hc = HumanCritic(working_path=working_path,
                     obs_size=state_size,
                     action_size=action_size,
                     batch_size=128,
                     hiden_sizes=(256, 256, 256),
                     maximum_segment_buffer=1000000,
                     training_epochs=reward_training_epochs,
                     maximum_preference_buffer=10000,
                     traj_k_lenght=traj_k_lenght,
                     regularize=args.regularize,
                     custom_oracle=False,
                     env_name=env_name,
                     seed=seed)

    env = HumanReward(env)
    agent = PPO(env=env, tensorboard_log=tensorboard_log, verbose=1, device="cpu", seed=torch_seed, **hyperparameters)
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=checkpoints_path, name_prefix=policy_name_save)

    update_reward_callback = UpdateRewardFunction(hc=hc,
                                                  agent=agent,
                                                  env_name=env_name,
                                                  n_queries=n_queries,
                                                  initial_reward_estimation_epochs=initial_reward_estimaton_epochs,
                                                  reward_training_epochs=reward_training_epochs,
                                                  truth=truth,
                                                  verbose=0,
                                                  n_initial_queries=init_prefs_n,
                                                  seed=seed,
                                                  max_queries=max_queries
                                                  )

    event_callback = EveryNTimesteps(n_steps=timesteps_per_update, callback=update_reward_callback)

    agent.learn(callback=[event_callback, checkpoint_callback], tb_log_name=tb_log_name, **learn_hyperparameters)
    hc.save_reward_model(working_path + 'Hopper-v3-rew')
    agent.save(working_path + 'Hopper-v3-model')
    hc.save_buffers(working_path)
    np.savetxt(working_path + 'scores000.csv',
               [p for p in zip(total_reward, total_correct_ratio, total_success_ratio)], delimiter=',', fmt='%s')


def train_with_human_preferences():
    global timesteps_per_update, init_prefs_n, n_queries, env, agent, event_callback
    video_location = './UI/preflearn/public/media/'
    # Train in cycles instead of full batch
    timesteps_per_update = 20000
    init_prefs_n = 40
    n_queries = 40
    env = gym.make(env_name)
    env = HumanReward(env)
    env.reset()

    agent = PPO(env=env, tensorboard_log=tensorboard_log, verbose=1, **hyperparameters)
    update_reward_human_callback = UpdateRewardFunctionRealHuman(hc,
                                                                 agent,
                                                                 env_name,
                                                                 video_location=video_location,
                                                                 n_initial_queries=init_prefs_n,
                                                                 initial_reward_estimaton_epochs=initial_reward_estimaton_epochs,
                                                                 n_queries=n_queries,
                                                                 traj_lenght=traj_k_lenght,
                                                                 verbose=0)
    event_callback = EveryNTimesteps(n_steps=timesteps_per_update, callback=update_reward_human_callback)
    agent.learn(callback=[event_callback], tb_log_name=tb_log_name, **learn_hyperparameters)
    hc.save_reward_model(env_path + env_name + "-loop" + extra_name)
    agent.save(env_name + "-loop" + extra_name)


def train_expert_trajectory_with_loop():
    global buffer_path, expert_model_name, hc, agent, env, checkpoint_callback
    global policy_name_load, policy_name_save, reward_model_load, reward_model_save

    hc = HumanCritic(working_path=working_path,
                     obs_size=state_size,
                     action_size=action_size,
                     batch_size=128,
                     hiden_sizes=(256, 256, 256),
                     maximum_segment_buffer=1000000,
                     training_epochs=reward_training_epochs,
                     maximum_preference_buffer=5000,
                     traj_k_lenght=traj_k_lenght,
                     regularize=args.regularize,
                     custom_oracle=True,  # Should be true when not training master oracle
                     env_name=env_name,
                     seed=seed)

    hc_expert = HumanCritic(working_path=working_path,
                            obs_size=state_size,
                            action_size=action_size,
                            batch_size=128,
                            hiden_sizes=(256, 256, 256),
                            maximum_segment_buffer=1000000,
                            training_epochs=reward_training_epochs,
                            traj_k_lenght=traj_k_lenght,
                            regularize=args.regularize,
                            env_name=env_name,
                            custom_oracle=False,
                            seed=seed)
    if env_name == "LunarLanderContinuous-v2" or env_name == "Hopper-v3":
        hc_expert.load_reward_model("LunarLanderContinuous-v2-final-model-rew")
        hc.load_reward_model("LunarLanderContinuous-v2-final-model-rew")

    policy_name_load_file_path = utils.find_model_name(args.policy_name_load, env_path)
    print("policy:")
    print(policy_name_load_file_path)

    expert_model_name = env_name + "-Expert"
    log_name = working_path + "prefs=" + str(init_prefs_n) + "_epochs=" + str(initial_reward_estimaton_epochs) \
               + "_ratio=" + str(args.human_reward_ratio)

    learn_hyperparameters['total_timesteps'] = int(args.final_timesteps)

    env.seed(seed)

    env = HumanReward(env)
    env = NormalizeReward(env)  # TODO see if it's worth or if we should cancel
    agent = PPO.load(policy_name_load_file_path, env, tensorboard_log=tensorboard_log, verbose=1)

    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=checkpoints_path, name_prefix=policy_name_save)

    update_reward_callback = UpdateRewardFunction2(hc=hc,
                                                   hc_expert=hc_expert,
                                                   agent=agent,
                                                   env_name=env_name,
                                                   n_queries=n_queries,
                                                   initial_reward_estimation_epochs=initial_reward_estimaton_epochs,
                                                   reward_training_epochs=reward_training_epochs,
                                                   truth=truth,
                                                   verbose=0,
                                                   traj_length=traj_k_lenght,
                                                   n_initial_queries=init_prefs_n,
                                                   seed=seed,
                                                   max_queries=max_queries
                                                   )
    update_reward_callback_N_ts = EveryNTimesteps(n_steps=timesteps_per_update, callback=update_reward_callback)

    callbacks = [checkpoint_callback, update_reward_callback_N_ts]

    agent.learn(tb_log_name=log_name, callback=callbacks, **learn_hyperparameters)
    agent.save(working_path + policy_name_save)

    np.savetxt(working_path + 'scores' + str(args.human_reward_ratio) + '.csv',
               [p for p in zip(total_reward, total_correct_ratio, total_success_ratio)], delimiter=',', fmt='%s')


if __name__ == "__main__":
    env_seed = 12345
    torch_seed = 12345
    numpy_seed = 12345
    python_seed = 12345
    total_reward = []
    total_correct_ratio = []
    total_success_ratio = []
    timestep = 0

    meta_data = {'loss': [], 'accuracy': [], 'episode_accuracy_mean': [], 'episode_loss_mean': []}

    env_name = "LunarLanderContinuous-v2"
    index = 0
    traj_k_lenght = 50
    all_segments = False
    all_segments_text = "all" if all_segments else ""

    args = parse_args_ppo()
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    #check for current folder
    current_path = os.getcwd()
    if not os.path.isdir(current_path + "/data/"): os.mkdir(current_path + "/data/")
    env_path = current_path + "/data/" + args.env
    if not os.path.isdir(env_path): os.mkdir(env_path)
    mode_path = env_path + "/mode_" + str(args.mode) + "/"
    if not os.path.isdir(mode_path): os.mkdir(mode_path)
    buffer_path = env_path + "/mode_4/"

    policy_name_load = args.policy_name_load
    policy_name_save = args.policy_name_save
    reward_model_load = args.reward_model_load
    reward_model_save = args.reward_model_save

    folder_name = args.folder_data_name + "_"
    folder_name += utils.return_date()
    working_path = mode_path + folder_name + "/"
    os.mkdir(working_path)

    extra_name = args.name + args.index
    env_name = args.env
    max_queries = args.max_queries
    smallest_rew_threshold = args.lowth
    largest_rew_threshold = args.highth
    mode = args.mode
    init_prefs_n = args.init_prefs_n
    n_queries = args.prefs_per_update
    truth = args.truth
    reward_training_epochs = args.epochs_per_update
    initial_reward_estimaton_epochs = args.init_epochs
    timesteps_per_update = 20000
    human_reward_ratio = args.human_reward_ratio


    #TODO: remove some of this
    expert_model_name = "PPO-optimal"
    checkpoints_path = working_path + "checkpoints/"
    tensorboard_log = working_path

    hyperparameters, env_hyperparameters, learn_hyperparameters, misc = utils.get_hyperparameters(args.env)

    if args.n_timesteps is not None:
        learn_hyperparameters['total_timesteps'] = args.n_timesteps

    if env_name == 'unity-env':
        channel = EngineConfigurationChannel()
        unity_env = UnityEnvironment('envs/snappy_rays', side_channels=[channel])
        channel.set_configuration_parameters(time_scale=30.0)
        env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=False)
    else:
        env = gym.make(env_name, **env_hyperparameters)
    action_size = env.action_space.shape[0] if len(env.action_space.shape) > 0 else 1
    state_size = env.observation_space.shape

    hc = HumanCritic(working_path=working_path,
                     obs_size=state_size,
                     action_size=action_size,
                     batch_size=128,
                     hiden_sizes=(256, 256, 256),
                     maximum_segment_buffer=1000000,
                     training_epochs=reward_training_epochs,
                     maximum_preference_buffer=5000,
                     traj_k_lenght=traj_k_lenght,
                     regularize=args.regularize,
                     custom_oracle=False,  # Should be true when not training master oracle
                     env_name=env_name)

    tb_log_name = env_name + "_" + extra_name

    # 0 = Train ppo with environmental reward function
    # 1 = Collect dataset
    # 2 = Train ppo with human reward model
    # 3 = Test ppo agent
    # 4 = Train agent iteratively with an oracle
    # 5 = Train agent with expert trajectories preferences
    # 6 = Train agent iteratively with a human

    print("Using environment: " + env_name)
    if mode == 0:
        # TODO fix that annealing is done when we load parameters
        lr_annealing = get_linear_fn(hyperparameters["learning_rate"], 1e-8, 1.0)
        del hyperparameters["learning_rate"]
        hyperparameters["learning_rate"] = lr_annealing

        clip_annealing = get_linear_fn(hyperparameters["clip_range"], 0.1, 1.0)
        del hyperparameters["clip_range"]
        hyperparameters["clip_range"] = clip_annealing

        agent = PPO(env=env, tensorboard_log=tensorboard_log, verbose=0, **hyperparameters)
        checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=checkpoints_path,
                                                 name_prefix=expert_model_name)
        callbacks = [checkpoint_callback]
        env = MetricWrapper(env, env_name, agent)
        env.seed(seed)

        agent.learn(callback=callbacks, tb_log_name="expert", **learn_hyperparameters)

        agent.save(working_path + "/" + args.policy_name_save)
        np.savetxt(working_path + 'scores0.csv',
                   [p for p in zip(total_reward, total_correct_ratio, total_success_ratio)], delimiter=',', fmt='%s')
    elif mode == 1:
        env = gym.make(env_name)
        obs = env.reset()
        policy_name_load = utils.find_model_name(args.policy_name_load, env_path)
        model = PPO.load(policy_name_load)
        model.load(policy_name_load)
        collect_segments(model, env, test_episodes=5000, n_collect_segments=init_prefs_n * 5)
        hc.save_buffers(path=mode_path)
    elif mode == 2:
        hc.load_reward_model(env_name)
        env = make_vec_env(env_name, n_envs=10, wrapper_class=HumanReward)
        model = PPO(env=env, tensorboard_log=tensorboard_log, verbose=1 **hyperparameters)
        model.learn(callback=checkpoint_callback, tb_log_name="HR", **learn_hyperparameters)
        model.save(expert_model_name + "-HR")
    elif mode == 3:
        policy_name_load = utils.find_model_name(args.policy_name_load, current_path)
        obs = env.reset()
        agent = PPO.load(policy_name_load + ".zip")
        test_agent(agent, test_episodes=10, render=args.render)
    elif mode == 4:
        train_with_oracle_preferences()
    elif mode == 5:
        expert_trajectory_preference_learning()
    elif mode == 6:  # Real humans
        train_with_human_preferences()
    elif mode == 7:
        train_with_oracle_preferences()
        agent, env = create_agent_vec_env(env_name, tensorboard_log, env_hyperparameters, hyperparameters, HumanReward)
        hc.load_reward_model(env_name + "-loop" + extra_name)
        checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=checkpoints_path, name_prefix=expert_model_name + "-Final")

        eval_env = gym.make(env_name)
        eval_env = HumanReward(eval_env)
        callbacks = [checkpoint_callback]

        agent.learn(tb_log_name="scratchHR", callback=callbacks, **learn_hyperparameters)
        agent.save(expert_model_name + "-Final-scratchHR")
    elif mode == 8:
        hc.load_reward_model("lunarlolrew")
        train_expert_trajectory_with_loop()
    elif mode == 9:
        train_expert_trajectory_with_loop()
    elif mode == 10:
        agent = PPO(env=env, tensorboard_log=tensorboard_log, verbose=0, **hyperparameters)
        checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=checkpoints_path,
                                                 name_prefix=expert_model_name)

        update_reward_callback = UpdateRewardFunction(hc=hc,
                                                      agent=agent,
                                                      env_name=env_name,
                                                      n_queries=n_queries,
                                                      initial_reward_estimation_epochs=initial_reward_estimaton_epochs,
                                                      reward_training_epochs=reward_training_epochs,
                                                      truth=truth,
                                                      verbose=0,
                                                      n_initial_queries=init_prefs_n
                                                      )
        update_reward_callback_N_ts = EveryNTimesteps(n_steps=timesteps_per_update, callback=update_reward_callback)
        env.seed(seed)
        callbacks = [checkpoint_callback, update_reward_callback_N_ts]
        agent.learn(callback=callbacks, tb_log_name="expert", **learn_hyperparameters)
        agent.save(working_path + "/" + args.policy_name_save)
        hc.save_reward_model(env_name + "-loop-reward")
        hc.save_buffers(working_path)
        np.savetxt(working_path + 'scores00.csv',
                   [p for p in zip(total_reward, total_correct_ratio, total_success_ratio)], delimiter=',', fmt='%s')