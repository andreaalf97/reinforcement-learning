from collections import deque
import random
from loguru import logger
from typing import Tuple
import pandas as pd
from game.game import start, right, left, up, down, board_to_state


ACTIONS = {
    "right": right,
    "left": left,
    "up": up,
    "down": down,
}

class ReplayMemory:
    def __init__(self, max_memory: int, is_conv: bool=False) -> None:
        assert isinstance(max_memory, int), f"`max_memory` expected type int, received {type(max_memory)}"
        assert isinstance(is_conv, bool), f"`is_conv` expected type int, received {type(is_conv)}"

        self.__is_conv = is_conv

        logger.info(f"[RM] Init replay memory..")
        self.__replay_memory, self.__info = self.__init_replay_memory(max_memory)

        m, s = self.__mean_std_steps(self.__info["steps_to_complete_game"])
        logger.info(f"[RM] Filled initial memory in {self.__info['episodes_simulated']} episodes [{m:.0f} ± {s:.0f} s/e].")
        logger.success("[RM] Done.")

    def append(self, el):
        self.__replay_memory.append(el)

    def get_memory(self):
        return self.__replay_memory.copy()

    def __len__(self):
        return len(self.__replay_memory)
        
    def __init_replay_memory(self, max_memory: int) -> Tuple[deque, dict]:
        replay_memory = deque(maxlen=max_memory)
        info = {
            "steps_to_complete_game": [],
            "episodes_simulated": 0,
        }

        while len(replay_memory) < max_memory:
            if len(replay_memory) % int(max_memory / 10) < 100:
                m, s = self.__mean_std_steps(info["steps_to_complete_game"])
                logger.info(f"[RM] Mem size {len(replay_memory)}, mean steps {m:.2f} ± {s:.2f}")
            board = start()

            last_board = board.copy()
            done = False
            n_steps = 0
            while not done:
                n_steps += 1
                action_name = random.choice(list(ACTIONS.keys()))

                board, reward, done = ACTIONS[action_name](last_board)
                replay_memory.append((
                    board_to_state(last_board, is_conv=self.__is_conv), action_name, board_to_state(board, is_conv=self.__is_conv), reward, done
                ))

                last_board = board

            info["episodes_simulated"] += 1
            info["steps_to_complete_game"].append(n_steps)

        info["avg_std_steps"] = self.__mean_std_steps(info["steps_to_complete_game"])
        return replay_memory, info

    def __mean_std_steps(self, steps: list) -> Tuple[float, float]:
        s = pd.Series(steps, dtype="float32")
        return s.mean(), s.std()





