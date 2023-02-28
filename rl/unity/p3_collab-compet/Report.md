# Code Report
## Learning Algorithm

Our `Agent` uses the actor-critic model with the Deep Deterministic Policy Gradients (DDPG) algorithm. We describe the hyperparameters chosen below:

This implementation uses some of the following techniques, which have been shown to boost agent reward:
- Ornstein Uhlenbeck noise
- Replay buffer
- Target networks
- Soft updating

The hyperparameters/structure chosen for our `Actor` are as follows:
- Input is the size of the state space
- First layer has `400` linear units with `ReLU` activation
- Second layer has `400` linear units with the same activation
- Third (output) layer has size of the action space with `tanh` activation

The hyperparameters/structure chosen for our `Critic` are as follows:
- Input is the size of the state space
- First layer has `400 + action_size` linear units with `ReLU` activation
- Second layer has `400` linear units with the same activation
- Third (output) layer has size of the action space with no (`Linear`) activation

For our `ActorCriticAgent`'s learning process (with experience reply using `ReplayBuffer` and `OUNoise`), our hyperparameters are:
- Learning Rate: `1e-4` (in both DNN)
- Update the network every `4` steps
- Have a discount factor (gamma) of `0.97`
- Use soft update of target parameters (tau) with `1e-3`
- Minibatch size of `64`
- Replay buffer size of `1e5`
- Critic weight decay of `0`
- Ornstein-Uhlenbeck noise parameters of `0.15` (theta) and `0.2` (sigma)

## Plot of Rewards

We achieved the required reward of about `0.6` (`0.1` above required threshold) after around `4000` episodes. This is quite a lot of episodes. These rewards, as observable in `main.py:27` are averaged over a sliding window (deque) of size `100`.

### Training Graph
![](res/training_score_graph.PNG)

## Ideas for Future Work

Here are some ideas I'm interested in applying to improve the performance of the agent:

- **D4PG, PPO, and/or AC3**: Implementing new methods which are known to achieve state-of-the-art performance will likely improve our agent's average return in this continuous state space. Training multiple agents concurrently would be quite cool.
- **Soccer**: Applying these models to a new environment may lead to even better hyperparameter choices that will work in the general case, for other environments. Even without applying new models to the "Soccer" environment, a new understanding of hyperparameter tuning would transfer to tackling that new problem.