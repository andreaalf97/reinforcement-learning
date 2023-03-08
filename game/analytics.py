import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
from math import pow, e


def mean_reward(max_moves):
    if 0 <= max_moves < 25:
        return (200/50) * max_moves - 1000
    elif 25 <= max_moves < 100:
        return (380/50) * max_moves - 1100
    elif 100 <= max_moves < 150:
        return (350/50) * max_moves - 1050
    elif 150 <= max_moves < 200:
        return (100/50) * max_moves - 300
    elif 200 <= max_moves:
        return (50/50) * max_moves - 100

def mean_num_steps(max_moves):
    return 50 * (1 - pow(e, -(max_moves-110)/50)) + 110 if max_moves >= 110 else max_moves

def str_to_sec(delta: str):
    splits = delta.split(':')
    return (float(splits[0]) * 60 * 60) + (float(splits[1]) * 60) + float(splits[2])

def plot_timers(timers: dict) -> None:
    for key in timers.keys():
        if len(timers[key]) == 0:
            print(f"Timer {key.upper()} is empty")
            continue
        fig, ax = plt.subplots(1, 1, figsize=(20, 8))
        ax.scatter(
            pd.Series([str_to_sec(t) for t in timers[key]], dtype="float32").index,
            pd.Series([str_to_sec(t) for t in timers[key]], dtype="float32"),
        )
        ax.set_title(f"Timers for {key.upper()} [s]")
        fig.show()

def plot_loss(losses: list) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    ax.plot(
        # [i for i, _ in enumerate(log["losses"])],
        pd.Series(losses).rolling(20, center=True).mean().dropna(),
    )
    ax.set_title("Distribution on MAIN NETWORK loss over time")
    fig.show();

def plot_total_steps(episodes_total_steps: list, epsilon: float, decay_factor: float, target_reset_episodes: list, window=20) -> None:
    from math import pow
    fig, (ax, ax_decay) = plt.subplots(2, 1, figsize=(20, 8), sharex=True)
    rolling_mean = pd.Series(episodes_total_steps).rolling(window, center=True).mean().dropna().tolist()
    ax.plot(rolling_mean)
    ax.scatter(
        [i for i in target_reset_episodes if i < len(rolling_mean)],
        [rolling_mean[i] for i in target_reset_episodes if i < len(rolling_mean)],
        c="red"
    )
    ax_decay.plot(pd.Series([epsilon * pow(1 - decay_factor, i) for i in range(len(episodes_total_steps))]))
    ax.set_title("Distribution on TOTAL STEPS over time")
    ax_decay.set_title("Decay of the EPSILON parameter")
    fig.show();

def plot_reward(rewards: list, window=5) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    ax.plot(
        # [i for i, _ in enumerate(log["losses"])],
        pd.Series(rewards).rolling(window, center=True).mean().dropna(),
    )
    ax.set_title("Distribution of total reward over multiple episodes")
    fig.show();