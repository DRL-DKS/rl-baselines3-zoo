# Preference-learning Branch
Changes made and how to run/install things specfically for preference learning go here

1) Included new callback which collects trajectories and preferences over those trajectories and updates the oracle reward function.
2) Included a wrapper which uses the learned reward function from the oracle and returns the human reward instead of the environment reward.
3) Made it possible to do hyperparameter tuning using Optuna for preference learning hyperparameters

## Running preference learning
To run preference learning all you have to do is to add the preference learning hyperparameters to the hyperparameters/algo.yml file similar to this:
~~~
  pref_learning:
    active: true
    human_critic:
      maximum_segment_buffer: 1000000
      maximum_preference_buffer: 3500
      training_epochs: 10
      batch_size: 32
      hidden_sizes: [64, 64]
      traj_k_lenght: 50
      regularize: false
      custom_oracle: true
    wrapper:
      - utils.wrappers.HumanReward:
          human_critic: true
    callback:
      - pref.callbacks.UpdateRewardFunction:
          human_critic: true
          use_env_name: true
          n_queries: 10
          initial_reward_estimation_epochs: 200
          reward_training_epochs: 50
          truth: 90
          traj_length: 50
          smallest_rew_threshold: 0
          largest_rew_threshold: 0
          verbose: 0
          n_initial_queries: 200
          seed: 12345
          max_queries: 1400
          every_n_timestep: 20000
~~~

| Hyperparameter                            | Description                                                                                                                              |
|-------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| active                                    | Tells train.py if preference learning should be used                                                                                     |
| human_critic.maximum_segment_buffer       | How many segments we can store in our buffer used to collect preference pairs                                                            |
| human_critic.maximum_preference_buffer    | How many preferences we can store in our buffer used to update reward model                                                              |
| human_critic.training_epochs              | How many training epochs to run for the reward model (currently overriden by callback.reward_training_epochs)                            |
| human_critic.batch_size                   | Batch sizes to be used per reward model update                                                                                           |
| human_critic.hidden_sizes                 | The FC network architecture to use for the reward model                                                                                  |
| human_critic.traj_k_lenght                | The trajectory length for each segment collected for preferences (overrides callback.traj_length)(misspelled)                            |
| human_critic.regularize                   | Decide if we should use the critical point loss function                                                                                 |
| human_critic.custom_oracle                | If the oracle should calculate loss according to true reward function or if we want to apply some reward shaping (e.g. additional tasks) |
| wrapper                                   | Takes a list of wrappers that should be used in preference learning (Currently only HumanReward is needed)                               |
| callback                                  | Takes a list of callbacks that should be used in preference learning (Currently only uses UpdateRewardFunction)                          |
| callback.n_queries                        | How many queries / preferences to be collected per reward model update                                                                   |
| callback.initial_reward_estimation_epochs | How many epochs to run for the initial training of the reward model                                                                      |
| callback.reward_training_epochs           | How many epochs to run for the update of the reward model                                                                                |
| callback.truth                            | Error rate (to simulate faulty human) for how often the oracle human will give the wrong preference                                      |
| callback.traj_length                      | Overriden by human_critic                                                                                                                |
| callback.n_initial_queries                | The initial amount of queries / preferences to collect for the initial reward model update                                               |
| callback.max_queries                      | The max amount of queries that will be collected before the agent stop asking for more and stop updating the reward model                |
| callback.every_n_timestep                 | The amount of timesteps between each update (currently does nothing as it's always set to 20k)                                           |

### Run training

To run, simply use the same commands as usual.

No docker:
~~~
python train.py --algo ppo --env LunarLanderContinuous-v2
~~~
Docker:
~~~
./scripts/run_docker_cpu.sh python train.py --algo ppo --env LunarLanderContinuous-v2
~~~

If you don't want to use preference learning, simply change the hyperparameter file so that "active: false". Currently no command line argument has been added to do this.

### Hyperparameter tuning

You can also run hyperparameter tuning using the regular commands. For example:
~~~
python train.py --algo ppo --env LunarLanderContinuous-v2 -n 500000 -optimize --n-trials 300 --n-jobs 2 --sampler tpe --pruner median
~~~


## Wrappers

### HumanReward
Replaces the environment reward with the reward received from the preference learning reward model.

#### Code usage
To create the wrapper in code you need to create it as
~~~
hc = HumanCritic(**args)
env = HumanReward(env, hc)
~~~
Important to note is that if you want to update the reward model of the HumanCritic and it's done via a callback, then it's important that they share the same HumanCritic instance.

## Callbacks

### UpdateRewardFunction
Callback to periodically gather query feedback and updates the reward model.

#### Code usage
To create the callback in code you need to create it as
~~~
hc = HumanCritic(**args)
env = UpdateRewardFunction(hc, env_name, **callbackArgs)
~~~
Important to note is that if you want to use it with the HumanCritic wrapper, then they should share the same HumanCritic instance.