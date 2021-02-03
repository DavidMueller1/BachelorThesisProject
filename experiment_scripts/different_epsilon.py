from data_util.experiment_data_classes import DeepQParameters
from init_experiment_assistant import ExperimentAssistant
from logger import Logger

experiment_name = "2021_01_26_different_epsilon_3"
best_of_n = 3

experiment_assistant = ExperimentAssistant(save_folder=experiment_name, save_name="epsilon_normal", terrain_file="test_2",
                    plot_training_progress=True, plot_interval=5, plot_moving_average_period=100, show_path_interval=10)

best = -99999
for n in range(0, best_of_n):
    Logger.status("\n\n===================\nExperiment %d/%d" % (n + 1, best_of_n))

    params = DeepQParameters(
        num_episodes=1000,
        max_steps_per_episode=90,
        replay_buffer_size=20000,
        batch_size=32,

        learning_rate=0.001,
        discount_rate=0.999,
        target_update=25,

        start_exploration_rate=1,
        max_exploration_rate=1,
        min_exploration_rate=0.005,
        exploration_decay_rate=0.01,

        rewards_all_episodes=[],
        max_rewards_all_episodes=[],
        max_reward_average=0
    )

    experiment_assistant.init_experiment(params=params)

    Logger.status("Max Reward Average:", experiment_assistant.params.max_reward_average)

    if experiment_assistant.params.max_reward_average > best:
        best = experiment_assistant.params.max_reward_average
        Logger.status("Better average reward. Overwriting save.")
        experiment_assistant.save_experiment_data(experiment_name, "epsilon_normal")


for i in range(0, 3):
    experiment_assistant.save_name = "epsilon_" + str(int(i * 33))
    best = -99999
    for n in range(0, best_of_n):
        Logger.status("\n\n===================\nExperiment %d/%d" % (n + 1, best_of_n))

        params = DeepQParameters(
            num_episodes=1000,
            max_steps_per_episode=90,
            replay_buffer_size=20000,
            batch_size=32,

            learning_rate=0.001,
            discount_rate=0.999,
            target_update=25,

            start_exploration_rate=i * 0.33,
            max_exploration_rate=1,
            min_exploration_rate=i * 0.33,
            exploration_decay_rate=0,

            rewards_all_episodes=[],
            max_rewards_all_episodes=[],
            max_reward_average=0
        )

        experiment_assistant.init_experiment(params=params)

        Logger.status("Max Reward Average:", experiment_assistant.params.max_reward_average)

        if experiment_assistant.params.max_reward_average > best:
            best = experiment_assistant.params.max_reward_average
            Logger.status("Better average reward. Overwriting save.")
            experiment_assistant.save_experiment_data(experiment_name, "epsilon_" + str(int(i * 33)))
