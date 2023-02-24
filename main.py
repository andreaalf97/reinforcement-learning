from game.game import start, get_available_actions, right, left, up, down
from game.brain import FFN
from game.data import StateDataset

from argparse import ArgumentParser
import random
from collections import deque
from loguru import logger
import torch
from torch.utils.data import DataLoader
from datetime import datetime

ACTIONS = {
    "right": right,
    "left": left,
    "up": up,
    "down": down,
}

def board_to_state(board):
    return board.flatten().tolist()

def train_model(main_network: torch.nn.Module, replay_memory: deque, target_network: torch.nn.Module) -> torch.nn.Module:
    if len(replay_memory) < args.min_replay_size:
        if not args.suppress_warnings:
            logger.warning(f"SKIPPING TRAINING - Memory size {len(replay_memory)}")
        return main_network
    
    training_sample = random.sample(replay_memory, args.n_samples_to_train_on)
    # Each sample in memory is: [current_state, action_name, new_state, reward, done]

    with torch.no_grad():
        current_states = torch.tensor([i[0] for i in training_sample], dtype=torch.float32)
        current_qs = main_network(current_states)
        new_states = torch.tensor([i[2] for i in training_sample], dtype=torch.float32)
        target_qs = target_network(new_states)

    X, Y = [], []
    for index, (current_state, action_name, new_state, reward, done) in enumerate(training_sample):
        if done:
            target_q = reward
        else:
            target_q = reward + args.discount_factor * torch.max(target_qs[index])
        action_index = list(ACTIONS.keys()).index(action_name)
        current_qs[index][action_index] = (1 - args.learning_rate) * current_qs[index][action_index] + args.learning_rate * target_q

        X.append(torch.tensor(current_state, dtype=torch.float32))
        Y.append(current_qs[index])

    dataloader = DataLoader(
        StateDataset(X, Y),
        batch_size=args.mini_batch_size,
        shuffle=True,
    )

    for epoch in range(args.epochs):
        for data in dataloader:
            X_batch, Y_batch = data
            criterion = torch.nn.MSELoss()
            main_network.zero_grad()
            out = main_network(X_batch)
            loss = criterion(out, Y_batch)
            loss.backward()


    return main_network


def main(args):

    now = datetime.now()
    base_file_name = f"{now.year:04d}_{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}"

    # Initialize the MAIN and TARGET networks
    main_network = FFN(
        16, 4, hidden_size=args.hidden_size
    )
    target_network = FFN(
        16, 4, hidden_size=args.hidden_size
    )
    target_network.load_state_dict(
        main_network.state_dict()
    )

    # The amount of steps played in the game
    steps = 0

    # Replay memory: each step contains
    # - current state
    # - action
    # - reward
    # - new state
    # - done
    replay_memory = deque(maxlen=50_000)

    logger.info(f"Running for {args.episodes} episodes..")
    for episode_number in range(args.episodes):

        total_episode_reward = 0
        current_episode_steps = 0
        current_epsilon = args.epsilon / (1 + (args.decay_factor*episode_number))
        done = False
        board = start()
        current_state = board_to_state(board)

        n_random_actions = 0
        n_best_actions = 0


        while not done:

            steps += 1
            current_episode_steps += 1
            
            if random.random() < current_epsilon:
                n_random_actions += 1
                action_name = random.choice(list(ACTIONS.keys()))
            else:
                n_best_actions += 1
                with torch.no_grad():
                    # Use the network to extract the Q-values for this state
                    q_values = main_network(torch.tensor(current_state, dtype=torch.float32).unsqueeze(dim=0))[0]
                    # Extract the action with the maximum Q-value
                    action_name = list(ACTIONS.keys())[int(torch.argmax(q_values))]

            assert current_state == board_to_state(board)
            action = ACTIONS[action_name]
            board, reward, done = action(board)
            new_state = board_to_state(board)
            total_episode_reward += reward

            replay_memory.append([
                current_state, action_name, new_state, reward, done
            ])

            if steps % args.update_main_network_every == 0:
                if not args.suppress_warnings:
                    logger.warning("[M] Updating MAIN network")
                main_network = train_model(
                    main_network,
                    replay_memory,
                    target_network
                )

            if steps > args.update_target_network_every:
                logger.error("[T] Updating TARGET network")
                target_network.load_state_dict(main_network.state_dict())
                steps = 0
            
            if current_episode_steps > args.max_moves_per_episode:
                done = True

            current_state = new_state

        
        logger.info(f"[{episode_number}] Episode completed with epsilon {current_epsilon:.3f}")
        logger.info(f"[{episode_number}] Reward achieved: --[{total_episode_reward}]--")
        logger.info(f"[{episode_number}] Memory size: {len(replay_memory)}")
        logger.info(f"[{episode_number}] Taken the best action {n_best_actions/(n_best_actions+n_random_actions)*100:.2f}% of the time")

if __name__ == "__main__":


    parser = ArgumentParser(
        prog="Deep2048",
        description="Reinforcement learning Deep Q Netork to play the game '2048'.",
    )
    parser.add_argument("-t", "--update-target-network-every", default=500, type=int, help="The amount of steps after which the target network is updated")
    parser.add_argument("-m", "--update-main-network-every", default=10, type=int, help="The amount of steps after which the main network is updated")
    parser.add_argument("-e", "--epsilon", default=.9, type=float, help="The starting epsilon parameter for the epsilon-greedy policy")
    parser.add_argument("-d", "--decay-factor", default=.06, type=float, help="The speed at which the epsilon factor decreases")
    
    parser.add_argument("--episodes", default=200, type=int, help="How many games to play during training")
    parser.add_argument("--max-moves-per-episode", default=400, type=int, help="How many moves are allowed per episode")
    
    parser.add_argument("--hidden-size", default=100, type=int, help="The hidden size of the neural network")
    parser.add_argument("--random-seed", default=0, type=int, help="The random seed to initialize all random number generators")

    parser.add_argument("--mini-batch-size", default=32, type=int, help="The size of the mini-batches to train the main network on")
    parser.add_argument("--epochs", default=1, type=int, help="How many times the model will go through the same sample of states in a single training session")

    parser.add_argument("--learning-rate", default=.7, type=float, help="The learning rate for the Bellman equation")
    parser.add_argument("--discount-factor", default=.600, type=float, help="The discount factor for the Bellman equation")
    parser.add_argument("--min-replay-size", default=1000, type=int, help="Minimum amount of samples to trigger training")
    parser.add_argument("--n-samples-to-train-on", default=1000, type=int, help="Samples used for every training step")

    parser.add_argument("--suppress-warnings", default=False, action="store_true", help="Suppress all training messages")
    parser.add_argument("--store-run-at", default="model_checkpoints", help="Where to store the training runs")

    args = parser.parse_args()

    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    main(args)