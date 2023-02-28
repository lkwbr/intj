
import gym


if __name__ == "__main__":
    env = gym.make("SpaceInvaders-v0")
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())

exit()

# custom utilies for displaying animation, collecting rollouts and more
import pong_utils

# check which device is being used.
# I recommend disabling gpu until you've made sure that the code runs
device = pong_utils.device
print("using device: ",device)

# render ai gym environment
import gym
import time

# PongDeterministic does not contain random frameskip
# so is faster to train than the vanilla Pong-v4 environment
env = gym.make('PongDeterministic-v4')

print("List of available actions: ", env.unwrapped.get_action_meanings())

# we will only use the actions 'RIGHTFIRE' = 4 and 'LEFTFIRE" = 5
# the 'FIRE' part ensures that the game starts again after losing a life
# the actions are hard-coded in pong_utils.py