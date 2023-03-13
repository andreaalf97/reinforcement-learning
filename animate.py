from game.game import GameSimulator

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import matplotlib
import numpy as np
import random
from typing import Tuple

DEFAUL_BACKGROUND_COLORS = {
    "0": "white",
    "2": "#dad9bb",
    "4": "#e6e38e",
    "8": "#f49236",
    "16": "#f44d36",
    "32": "#ff1f00",
    "64": "#fff800",
    "128": "#afffff",
    "256": "#619aff",
    "512": "#e485f1",
    "1024": "#b5ffc0",
    "2048": "#00ff26"
}

SQUARE_WIDTH = 10.

ANCHOR = (20., 20.)

def plot_state(transition):
    start_state, action, end_state, reward, done = transition

    for row in range(len(start_state)):
        for col in range(len(start_state[row])):
            cell_value = str(start_state[row, col])
            x_offset = (4 - col - 1) * SQUARE_WIDTH
            y_offset = (4 - row - 1) * SQUARE_WIDTH
            rect_xy = (ANCHOR[0] + x_offset, ANCHOR[1] + y_offset)
            # rect = Rectangle(
            rect = Rectangle(
                rect_xy,
                SQUARE_WIDTH, SQUARE_WIDTH,
                angle=0,
                facecolor=DEFAUL_BACKGROUND_COLORS[cell_value],
                edgecolor="black"
            )
            ax.add_patch(rect)

            if cell_value != '0':
                ax.annotate(
                    cell_value,
                    (rect_xy[0] + (.85*SQUARE_WIDTH)/2, rect_xy[1] + (.90*SQUARE_WIDTH)/2),
                    weight='bold',
                    color="black"
                )


def init_fig() -> Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
    fig, ax = plt.subplots()

    # Setting limits for x and y axis
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 80)

    rectangles = [[], [], [], []]
    annotations = [[], [], [], []]
    for row in range(4):
        for col in range(4):
            cell_value = "0"
            x_offset = (4 - col - 1) * SQUARE_WIDTH
            y_offset = (4 - row - 1) * SQUARE_WIDTH
            rect_xy = (ANCHOR[0] + x_offset, ANCHOR[1] + y_offset)
            # rect = Rectangle(
            rect = Rectangle(
                rect_xy,
                SQUARE_WIDTH, SQUARE_WIDTH,
                angle=0,
                facecolor=DEFAUL_BACKGROUND_COLORS[cell_value],
                edgecolor="black"
            )
            rectangles[row].append(ax.add_patch(rect))
            annotations[row].append(
                ax.annotate(
                    "0",
                    (rect_xy[0] + (.85*SQUARE_WIDTH)/2, rect_xy[1] + (.90*SQUARE_WIDTH)/2),
                    weight='bold',
                    color="black"
                )
            )
            
    print(rectangles)
    print(annotations)
    return fig, ax

def main():
    game = GameSimulator(100, is_conv=True)
    current_board = game.board_to_state()
    done = False

    transitions = []

    while not done:
        action = random.choice(['right', 'left', 'up', 'down'])
        reward, done = game.move(action)
        new_board = game.board_to_state()

        transitions.append((
            current_board[0],
            action,
            new_board[0],
            reward,
            done
        ))

        current_board = new_board


    fig, ax = init_fig()

    plt.show()
    return
    x = []
    y = []


    # Since plotting a single graph
    line, = ax.plot(0, 0)

    def animation_function(i):
        x.append(i * 10)
        y.append(i * 10)

        line.set_xdata(x)
        line.set_ydata(y)
        return line,

    animation = FuncAnimation(
        figure,
        func = animation_function,
        frames = np.arange(0, 10, 0.1),
        interval = 100
    )
    plt.show()


if __name__ == "__main__":
	main()
