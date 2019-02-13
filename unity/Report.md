# Code Report
## Learning Algorithm

An average reward of about `+15` was achieved using a double Deep Q-Learning with a standard experience replay. Our "local" and "target" networks have the same architecture.

The hyperparameters chosen for our `QNetwork` are as follows:
- Input layer is the size of the state space (no preprocessing)
- First layer has `64` linear units with ReLU activation
- Second layer has `128` linear units with the same activation
- Third (output) layer has size of the action space with no activation

For our `DqnAgent`'s learning process (with experience reply using `ReplayBuffer`), our hyperparameters are:
- Update the network every `4` steps
- Have a discount factor (gamma) of `0.99`
- Have a learning rate of `5e-4`
- Use soft update of target parameters (tau) with `1e-3`
- Minibatch size of `64`
- Replay buffer size of `1e5`

## Plot of Rewards

We achieved the required reward of `+13` quite quickly, shown below. It appears we broke the required reward threshold between episode `200` and `400`. These rewards, as observable in `main.py:35` are averaged over a sliding window (deque) of size `100`.

Note that the training was not stopped after achieving the required reward in order to test the maximum capacity of this network.

### Training Terminal Output

```
Episode 100     Average Score: 2.960
Episode 200     Average Score: 8.04
Episode 300     Average Score: 12.07
Episode 400     Average Score: 13.73
Episode 500     Average Score: 13.38
Episode 600     Average Score: 14.72
Episode 700     Average Score: 15.29
Episode 800     Average Score: 15.63
Episode 900     Average Score: 16.24
Episode 1000    Average Score: 16.19
Episode 1100    Average Score: 16.42
Episode 1200    Average Score: 15.78
Episode 1300    Average Score: 16.59
Episode 1400    Average Score: 15.81
Episode 1500    Average Score: 15.61
Episode 1600    Average Score: 15.73
Episode 1700    Average Score: 15.35
Episode 1800    Average Score: 15.74
Episode 1900    Average Score: 15.26
Episode 2000    Average Score: 15.22
```

### Training Graph
![](res/training_score_graph.PNG)

## Ideas for Future Work

Here are some ideas I'm interested in applying to improve the performance of the agent:

- **More network layers**: Through experimentation, it was noticed that increasing the number of nodes in a single layer from `64` to `128` led to an improvement of about `+4` in average reward. Additionally, since it is known that deeper networks tend to improve accuracy by creating more complex heirarchies of concepts, going deeper is likely a good idea.
- **Prioritized experience reply**: In this environment, there are states in which the landscape appears to be quite empty (which I posit are less important states) and those which are very dense with both blue and yellow bananas, which are the most ripe (pun intended) experiences for discerning positive and negative reward.
- **Dueling Deep Q-Networks**: Having an advantage and state-value function would seem to be beneficial since most states in the "banana" environment do not vary across actions.