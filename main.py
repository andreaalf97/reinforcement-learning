from game.game import GameSimulator, generate_max_moves_list
from game.brain import FFN, ConvBrain
from game.data import StateDataset
from game.memory import ReplayMemory
from game.analytics import mean_num_steps, mean_reward

from argparse import ArgumentParser
import random
from loguru import logger
import torch
from torch.utils.data import DataLoader
from datetime import datetime
import json
from typing import Tuple
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
import math
from typing import Dict


TIMERS: Dict[str, list] = {
    "train": [],
    "create_train_tensors": [],
    "sample_memory": [],
    "calculate_qs": [],
    "episodes": [],
}

def mean_random_moves(max_moves_allowed: int):
    return 50 * (1 - math.pow(math.e, -(max_moves_allowed-110)/50)) + 110 if max_moves_allowed > 110 else max_moves_allowed

def init_networks(args):
    # Initialize the MAIN and TARGET networks
    if not args.conv:
        main_network = FFN(16, 4, hidden_size=args.hidden_size).to("cuda" if args.cuda else "cpu")
        target_network = FFN(16, 4, hidden_size=args.hidden_size).to("cuda" if args.cuda else "cpu")
    else:
        main_network = ConvBrain((4, 4), 4, hidden_size=args.hidden_size).to("cuda" if args.cuda else "cpu")
        target_network = ConvBrain((4, 4), 4, hidden_size=args.hidden_size).to("cuda" if args.cuda else "cpu")

    if args.resume_from != "":
        assert os.path.exists(f"{args.store_run_at}/{args.resume_from}"), f"Folder `{args.store_run_at}/{args.resume_from}` does not exist"
        for _, _, file_names in os.walk(f"{args.store_run_at}/{args.resume_from}"):
            all_models = [n for n in sorted(file_names) if '.pt' in n]
            if 'model.pt' in all_models:
                main_network.load_state_dict(torch.load(f"{args.store_run_at}/{args.resume_from}/model.pt"))
            else:
                main_network.load_state_dict(torch.load(f"{args.store_run_at}/{args.resume_from}/{all_models[-1]}"))
    # Initialize both netoworks with the same weights
    target_network.load_state_dict(
        main_network.state_dict()
    )
    return main_network, target_network


def train_model(main_network: torch.nn.Module, replay_memory: ReplayMemory, target_network: torch.nn.Module, losses:list, args, action_order=["right", "left", "up", "down"]) -> Tuple[torch.nn.Module, list]:
    if len(replay_memory) < args.min_replay_size:
        if args.log_training_events:
            logger.warning(f"SKIPPING TRAINING - Memory size {len(replay_memory)}")
        return losses

    start_sample = datetime.now()
    training_sample = random.sample(replay_memory.get_memory(), args.n_samples_to_train_on if args.n_samples_to_train_on <= len(replay_memory) else len(replay_memory))
    TIMERS["sample_memory"].append(str(datetime.now() - start_sample))

    start_create_tensors = datetime.now()
    # Each sample in memory is: [current_state, action_name, new_state, reward, done]
    with torch.no_grad():
        current_states = torch.tensor([i[0].tolist() for i in training_sample], dtype=torch.float32).to("cuda" if args.cuda else "cpu")
        current_qs = main_network(current_states)
        new_states = torch.tensor([i[2].tolist() for i in training_sample], dtype=torch.float32).to("cuda" if args.cuda else "cpu")
        target_qs = target_network(new_states)
    TIMERS["create_train_tensors"].append(str(datetime.now() - start_create_tensors))

    start_calc_q = datetime.now()
    X, Y = [], []
    for index, (current_state, action_name, new_state, reward, done) in enumerate(training_sample):
        if done:
            target_q = reward
        else:
            target_q = reward + args.discount_factor * torch.max(target_qs[index])
        action_index = action_order.index(action_name)
        current_qs[index][action_index] = (1 - args.learning_rate) * current_qs[index][action_index] + args.learning_rate * target_q

        X.append(torch.tensor(current_state, dtype=torch.float32).to("cuda" if args.cuda else "cpu"))
        Y.append(current_qs[index])
    TIMERS["calculate_qs"].append(str(datetime.now() - start_calc_q))

    start_train = datetime.now()
    dataloader = DataLoader(
        StateDataset(X, Y),
        batch_size=args.mini_batch_size,
        shuffle=True,
    )

    for epoch in range(args.epochs):
        ls = []
        for data in dataloader:
            X_batch, Y_batch = data
            criterion = torch.nn.MSELoss()
            main_network.zero_grad()
            out = main_network(X_batch)
            loss = criterion(out, Y_batch)
            ls.append(loss.item())
            loss.backward()

    mean_loss = sum(ls)/len(ls)
    losses.append(mean_loss)

    TIMERS["train"].append(str(datetime.now() - start_train))
    return losses


def main(args):

    now = datetime.now()
    base_folder_name = f"{now.year:04d}_{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}"
    if not args.no_store:
        if not os.path.exists(f"{args.store_run_at}"):
            os.makedirs(f"{args.store_run_at}")
        if not os.path.exists(f"{args.store_run_at}/{base_folder_name}"):
            os.makedirs(f"{args.store_run_at}/{base_folder_name}")

    logger.add(f"{args.store_run_at}/{base_folder_name}/run_info.log", encoding="utf8")

    main_network, target_network = init_networks(args)

    # The amount of steps played in the game
    steps = 0

    # Replay memory: each step contains
    # - current state
    # - action
    # - reward
    # - new state
    # - done
    replay_memory = ReplayMemory(args)

    max_moves_per_episode = generate_max_moves_list(args)

    rewards = []
    losses = []
    episodes_total_steps = []
    target_resets = []

    logger.info(f"Running for {args.episodes} episodes..")
    for episode_number in range(args.episodes):

        start_episode = datetime.now()
        total_episode_reward = 0
        current_episode_steps = 0
        # current_epsilon = args.epsilon / (1 + (args.decay_factor*episode_number))
        current_epsilon = args.epsilon * pow(1 - args.decay_factor, episode_number)
        done = False
        game = GameSimulator(int(max_moves_per_episode[0]), is_conv=args.conv)
        current_state = game.board_to_state()

        max_moves = int(max_moves_per_episode[episode_number])

        n_random_actions = 0
        n_best_actions = 0


        while not done:

            steps += 1
            current_episode_steps += 1
            
            available_actions = game.get_available_actions()
            if random.random() < current_epsilon:
                n_random_actions += 1
                action_name = random.choice(list([k for k in game.actions.keys() if k in available_actions]))
            else:
                n_best_actions += 1
                with torch.no_grad():
                    # Use the network to extract the Q-values for this state
                    q_values = main_network(torch.tensor(current_state, dtype=torch.float32).to("cuda" if args.cuda else "cpu").unsqueeze(dim=0))[0]

                    # Set indices for unavailable moves to -inf
                    missing_moves = [k for k in game.actions.keys()  if k not in available_actions]
                    missing_moves_indices = [i for i, k in enumerate(game.actions.keys() ) if k in missing_moves]
                    for ind in missing_moves_indices:
                        q_values[ind] = -float('inf')
                    
                    # Extract the action with the maximum Q-value
                    action_name = list(game.actions.keys() )[int(torch.argmax(q_values))]

            assert current_state.tolist() == game.board_to_state().tolist()
            reward, done = game.move(action_name)
            new_state = game.board_to_state()
            total_episode_reward += reward

            replay_memory.append((
                current_state, action_name, new_state, reward, done
            ))

            if steps % args.update_main_network_every == 0:
                if args.log_training_events:
                    logger.warning("[M] Updating MAIN network")
                losses = train_model(
                    main_network,
                    replay_memory,
                    target_network,
                    losses,
                    args
                )
                losses[-1] = (losses[-1], False)

            if (episode_number % int(args.episodes / 5) == 0) and (episode_number != 0):
                if not args.no_store:
                    if not os.path.exists(f"{args.store_run_at}/{base_folder_name}/checkpoint_{episode_number:04d}.pt"):
                        logger.warning(f"Storing checkpoint model at {args.store_run_at}/{base_folder_name}/checkpoint_{episode_number:04d}.pt")
                        torch.save(target_network.state_dict(), f"{args.store_run_at}/{base_folder_name}/checkpoint_{episode_number:04d}.pt")

            # training_started = len(replay_memory) >= args.min_replay_size
            if current_episode_steps > max_moves:
                done = True

            if steps > args.update_target_network_every:
                losses[-1] = (losses[-1][0], True)
                target_resets.append(episode_number)
                if args.log_training_events:
                    logger.error("[T] Updating TARGET network")
                target_network.load_state_dict(main_network.state_dict())
                steps = 0

            if done:
                episodes_total_steps.append(current_episode_steps)

            current_state = new_state

        rewards.append(str(total_episode_reward))

        TIMERS["episodes"].append(str(datetime.now() - start_episode))

        if episode_number % int(args.episodes / 10) == 0:
            logger.info(f"[{episode_number}] Episode completed with epsilon {current_epsilon:.3f}")
            logger.info(f"[{episode_number}] Avg reward: --[{sum([int(i) for i in rewards[-10:]])/10:.2f}]--")
            logger.info(f"[{episode_number}] Memory size: {len(replay_memory)}")
            logger.info(f"[{episode_number}] Max moves: {max_moves}")
            logger.info(f"[{episode_number}] Avg moves: --[{sum([int(i) for i in episodes_total_steps[-10:]])/10:.2f}]--")
            logger.info(f"[{episode_number}] Taken the best action {n_best_actions/(n_best_actions+n_random_actions)*100:.2f}% of the time")
            logger.info(f"---------")
    
    if args.log_training_events:
        logger.error("[T] Updating TARGET network")
    target_network.load_state_dict(main_network.state_dict())

    run_info = {}
    run_info["params"] = vars(args)
    run_info["rewards"] = rewards
    run_info["max_moves"] = max_moves_per_episode
    run_info["losses"] = losses
    run_info["timers"] = TIMERS
    run_info["episodes_total_steps"] = episodes_total_steps
    run_info["target_reset_episodes"] = target_resets
    if not args.no_store:
        logger.info(f"Storing run parameters at {args.store_run_at}/{base_folder_name}/run_info.json")
        with open(f"{args.store_run_at}/{base_folder_name}/run_info.json", 'w') as fp:
            json.dump(run_info, fp)
        logger.info(f"Storing trained model at {args.store_run_at}/{base_folder_name}/model.pt")
        torch.save(target_network.state_dict(), f"{args.store_run_at}/{base_folder_name}/model.pt")
        logger.info(f"Storing run last memory at {args.store_run_at}/{base_folder_name}/memory.pkl")
        with open(f"{args.store_run_at}/{base_folder_name}/memory.pkl", 'wb') as file:
                pickle.dump(replay_memory.get_memory(), file)

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.plot(pd.Series([int(i) for i in run_info["rewards"]]).rolling(5).mean().dropna())
    # ax.plot(
    #     range(len(run_info["max_moves"])),
    #     [mean_reward(m) + 1000 for m in run_info["max_moves"]]
    # )
    ax.set_title(f"Reward over episodes. [MAX {max([int(i) for i in run_info['rewards']])}]")

    target_train_indices = [i for i, (loss, target_train) in enumerate(run_info["losses"]) if target_train]
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.plot(
        [i for i, (_, _) in enumerate(run_info["losses"])],
        [l for _, (l, _) in enumerate(run_info["losses"])]
        # pd.Series([l for l, t in run_info["losses"]]).rolling(10).mean().dropna()
    )
    ax.scatter(
        target_train_indices,
        [loss for i, (loss, _) in enumerate(run_info["losses"]) if i in target_train_indices],
        c='red'
    )
    ax.set_title("Loss")
    plt.show()


if __name__ == "__main__":


    parser = ArgumentParser(
        prog="deep2048",
        description="Reinforcement learning Deep Q Netork to play the game '2048'.",
    )

    parser.add_argument("--resume-from", default="", help="Resume from the latest model in the given folder")

    parser.add_argument("-t", "--update-target-network-every", default=1000, type=int, help="The amount of steps after which the target network is updated")
    parser.add_argument("-m", "--update-main-network-every", default=32, type=int, help="The amount of steps after which the main network is updated")
    parser.add_argument("--epsilon", default=1., type=float, help="The starting epsilon parameter for the epsilon-greedy policy")
    parser.add_argument("-d", "--decay-factor", default=.1, type=float, help="The speed at which the epsilon factor decreases")
    
    parser.add_argument("--episodes", default=350, type=int, help="How many games to play during training")
    # parser.add_argument("--max-moves-per-episode", default=-1, type=int, help="How many moves are allowed per episode")
    parser.add_argument("--max-moves-start", default=100, type=int)
    parser.add_argument("--max-moves-end", default=400, type=int)
    
    parser.add_argument("--hidden-size", default=32, type=int, help="The hidden size of the neural network")
    parser.add_argument("--conv", default=False, action="store_true", help="Uses a convolutional NN as brain")
    parser.add_argument("--random-seed", default=0, type=int, help="The random seed to initialize all random number generators")

    parser.add_argument("--mini-batch-size", default=32, type=int, help="The size of the mini-batches to train the main network on")
    parser.add_argument("-e", "--epochs", default=10, type=int, help="How many times the model will go through the same sample of states in a single training session")
    parser.add_argument("--max-memory", default=10_000, type=int, help="The maximum size of the replay memory")

    parser.add_argument("-l", "--learning-rate", default=.7, type=float, help="The learning rate for the Bellman equation")
    parser.add_argument("--discount-factor", default=.618, type=float, help="The discount factor for the Bellman equation")
    parser.add_argument("--min-replay-size", default=1000, type=int, help="Minimum amount of samples to trigger training")
    parser.add_argument("--n-samples-to-train-on", default=10_000, type=int, help="Samples used for every training step")

    parser.add_argument("--log-training-events", default=False, action="store_true", help="Prints a message every time a training event happens")
    parser.add_argument("--store-run-at", default="model_checkpoints", help="Where to store the training runs")
    parser.add_argument("--no-store", default=False, action="store_true", help="Don't store training parameters and trained model")
    parser.add_argument("--checkpoint-every", default=100, type=int, help="How often (# episodes) to store a checkpoint model")
    parser.add_argument("--cuda", default=torch.cuda.is_available(), action="store_true", help="Train using GPU")

    args = parser.parse_args()

    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    logger.info("RUN PARAMETERS:")
    logger.info(vars(args))

    main(args)