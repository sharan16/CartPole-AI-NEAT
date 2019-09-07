import neat
import pickle
import gym

def test(net):
    env = gym.make('CartPole-v1')
    observation = env.reset()
    for t in range(1000):
        env.render()
        if net.activate(tuple(observation))[0] > 0:
            action = 0
        else:
            action = 1       
        observation, reward, done, info = env.step(action)

        
        if done:
            env.reset()
    env.close()

with open("winner.pkl","rb") as file:
    net = pickle.load(file)
test(net)