import gym
import os
import neat
import pickle

def eval_genomes(genomes, config):
    env = gym.make('CartPole-v1')
    action = -5
    for _,g in genomes:
        observation = env.reset()
        g.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(g,config)
        reward_sum = 0
        averager = 20
        for _ in range(averager):
            observation = env.reset()
            g.fitness = 0
            for t in range(1000):
 
                if net.activate(tuple(observation))[0] > 0:
                    action = 0
                else:
                    action = 1
                observation, reward, done, info = env.step(action)
                g.fitness += reward
                if done:
                    break
            reward_sum += g.fitness
            print("finished reward sum: {}    g.fitness:{}".format(reward_sum,g.fitness))
        g.fitness = round(reward_sum/averager)
        print("Genome finished with a fitness of {}".format(g.fitness))
    print("Finished Generation")
    env.close()
def run():
    local_dir = os.path.dirname(__file__)
    config_file_path = os.path.join(local_dir,'neatconfig.txt')
    config = neat.config.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,config_file_path)
    p = neat.Population(config)
    # p.add_reporter(neat.StdOutReporter(True))
    winner = p.run(eval_genomes,50)
    with open('winner.pkl','wb') as file:
        pickle.dump(neat.nn.FeedForwardNetwork.create(winner,config),file)

run()
