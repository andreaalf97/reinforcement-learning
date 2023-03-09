# Reinforcement Learning

**This repo contains a set of games and my corresponding attempts at creating Reinforcement Learning Agents capable of beating these games.**

## Current games
* [2048](https://en.wikipedia.org/wiki/2048_(video_game)) - "a single-player sliding tile puzzle video game written by Italian web developer Gabriele Cirulli and published on GitHub."

### 2048

For now, this is the only game present in the repo. It is present as a playable online game under folders `api/` and `frontend/` and as a set of functions to train an agent on under `game/`.

#### How to play the game locally
- To play locally, first clone the repo and create a local environment (e.g. using [conda](https://docs.conda.io/en/latest/miniconda.html)) with `Python 3.9.11` installed.
```
git clone https://github.com/andreaalf97/reinforcement-learning
conda create -n <env-name> python==3.9.11
conda activate <env-name>
```
- Install the required packages.
```
pip install -r playing_requirements.txt
```
- Start the API where the game is hosted, which will live on **port 5000**.
```
python api/app.py
```
- Start the frontend server from the `frontend/` folder.
```
cd frontend
python -m http.server 1234
```
- Navigate to the game URL at [http://localhost:1234/](http://localhost:1234/)

#### How to train the RL Agent

The training strategy is based on [Deep Q Networks](https://en.wikipedia.org/wiki/Q-learning#Deep_Q-learning), an adaptation of Q Learning for high state-action dimensionalities.
In particular, this is an adaptation of [this](https://www.tensorflow.org/agents/tutorials/0_intro_rl) and [this](https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc) articles.

##### Installation
- First, clone the repo and create a local environment (e.g. using [conda](https://docs.conda.io/en/latest/miniconda.html)) with `Python 3.9.11` installed.
```
git clone https://github.com/andreaalf97/reinforcement-learning
conda create -n <env-name> python==3.9.11
conda activate <env-name>
```
- Install the requirements.
```
pip install -r training_requirements.txt
```
- Start training
```
python main.py
```
- At the end of training, you will be prompted with two images showing the training loss convergence and the average total reward achieved over the episodes. The model parameters, loss, total loss and a copy of the final trained model will be stored under `model_checkpoints/` or wherever indicated by parameter `--store-run-at`.

##### State representation

The game state is represented by a one dimension array of size 16. E.g. the following game state:
|||||
|----|----|---|---|
| 2  | 4  | 4 | 0 |
| 0  | 0  | 0 | 0 |
| 4  | 4  | 0 | 0 |
| 16 | 16 | 8 | 0 |

is represented by the following array:
`[2, 4, 4, 0, 0, 0, 0, 0, 4, 4, 0, 0, 16, 16, 8, 0]`.

##### Possible actions

The available actions are `right`, `left`, `up`, `down`.

##### Reward scheme

Merging two tiles with the same number has as reward the amount of the number generated. If multiple tiles are merged in one move, the total reward is the sum of the single rewards. E.g. the reward for performing action`left` on the state above is `48 = 32 + 8 + 8`.

##### Brain

The architecture used to learn the game is composed by two linear layers with a ReLu activation function in between and hidden dimension given by argument `--hidden-size`.

##### Default parameters
For a full view on the parameters, run `python main.py --help`.

| Parameter | Default value | Description |
|-----------|---------------|-------------|
|`--update-target-network-every`|1000|After how many game steps the main model weights are copied onto the target model|
|`--update-main-network-every`|16|After how many game steps the main target is trained|
|`--episodes`|350|How many game episodes will be simulated to collect training samples|
|`--max-moves-per-episode`|400|How many moves are allowed per episode|
|`--hidden-size`|32|The size of model's hidden dimension|
|`--random-seed`|0|The seed for all random number generators, for reproducibility purposes|
|`--n-samples-to-train-on`|1000|How many (randomly picked from the replay memory) samples the model is trained on at each training step|
|`--mini-batch-size`|32|The mini-batch size used for training|
|`--min-replay-size`|1000|The minimum size of the replay-memory before training can start|
|`--epochs`|1|How many times training will go over the same randomly picked sample of replay-memory|
|`--learning-rate`|0.7|The learning rate for the Bellman equation: `new_q = (1-lr)*(current_q) + (lr)*(new_q)`|
|`--discount-factor`|0.618|The discount factor for the Bellman equation: `new_q = reward + DF * max(Q | a)`|
