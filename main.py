import gym

def eval_genomes(genomes, config):
    env = gym.make('CartPole-v0')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print(reward)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
def run():
    local_dir = os.path.dirname(__file__)
    config_file_path = os.path.join(local_dir,'neatconfig.txt')
    config = neat.config.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,config_file_path)
    p = neat.Population(config)
    # p.add_reporter(neat.StdOutReporter(True))
    winner = p.run(eval_genomes,100)
eval_genomes(None,None)