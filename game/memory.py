from collections import deque
import random
from loguru import logger
from typing import Tuple
import pandas as pd
from game.game import GameSimulator


ACTIONS = [
    "right", "left", "up", "down"
]

class ReplayMemory:
    def __init__(self, args) -> None:
        self.args = args

        logger.info(f"[RM] Init replay memory..")
        self.__replay_memory, self.__info = self.__init_replay_memory()

        m, s = self.__mean_std_steps(self.__info["steps_to_complete_game"])
        logger.info(f"[RM] Filled initial memory in {self.__info['episodes_simulated']} episodes [{m:.0f} ± {s:.0f} s/e].")
        logger.success("[RM] Done.")

    def append(self, el):
        self.__replay_memory.append(el)

    def get_memory(self):
        return self.__replay_memory.copy()

    def __len__(self):
        return len(self.__replay_memory)
        
    def __init_replay_memory(self) -> Tuple[deque, dict]:
        replay_memory = deque(maxlen=self.args.max_memory)
        info = {
            "steps_to_complete_game": [],
            "episodes_simulated": 0,
        }

        while len(replay_memory) < self.args.max_memory:
            if len(replay_memory) % int(self.args.max_memory / 10) < 100:
                m, s = self.__mean_std_steps(info["steps_to_complete_game"])
                logger.info(f"[RM] Mem size {len(replay_memory)}, mean steps {m:.2f} ± {s:.2f}")
            game = GameSimulator(self.args.max_moves_start, is_conv=self.args.conv)
            board = game.board_to_state()

            done = False
            n_steps = 0
            while not done:
                n_steps += 1
                action_name = random.choice(ACTIONS)

                reward, done = game.move(action_name)
                new_board = game.board_to_state()
                replay_memory.append((board, action_name, new_board, reward, done))
                board = new_board


            info["episodes_simulated"] += 1
            info["steps_to_complete_game"].append(n_steps)

        info["avg_std_steps"] = self.__mean_std_steps(info["steps_to_complete_game"])
        return replay_memory, info

    def __mean_std_steps(self, steps: list) -> Tuple[float, float]:
        s = pd.Series(steps, dtype="float32")
        return s.mean(), s.std()





