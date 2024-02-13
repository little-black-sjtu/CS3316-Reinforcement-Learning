import argparse
import gymnasium as gym

def init_parser():
    parser = argparse.ArgumentParser(description="CS3316 PROJECT")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--memory_size', type=int, default=100000)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--target_update_interval', type=int, default=5000)
    
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.001)
    
    parser.add_argument('--max_num_steps', type=int, default=10000)
    parser.add_argument('--saver_steps', type=int, default=10000)

    parser.add_argument('--episode_start', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=5000)
    parser.add_argument('--num_eval', type=int, default=100)
    parser.add_argument('--keep_train', type=bool, default=False)

    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.1)
    parser.add_argument('--explore_steps', type=int, default=800000)

    parser.add_argument('--save_dir', type=str, default='save_ddpg')
    parser.add_argument('--log_file', type=str, default='log/log_ddpg.csv')
    parser.add_argument('--model', type=str, default=None)

    parser.add_argument('--test_dqn', type=bool, default=False)
    parser.add_argument('--train_dqn', type=bool, default=False)
    parser.add_argument('--test_ddpg', type=bool, default=False)
    parser.add_argument('--train_ddpg', type=bool, default=False)

    return parser.parse_args()

def run():
    args = init_parser()

    if args.test_dqn:
        env = gym.make("BreakoutNoFrameskip-v4") #, render_mode='human')
        env = gym.wrappers.AtariPreprocessing(env, frame_skip=1, terminal_on_life_loss=True)
        env = gym.wrappers.FrameStack(env, 4)

        from dqn_agent import DQN_Agent
        agent = DQN_Agent(env, args)
        agent.test()
    elif args.train_dqn:
        env = gym.make("BreakoutNoFrameskip-v4")
        env = gym.wrappers.AtariPreprocessing(env, frame_skip=1, terminal_on_life_loss=True)
        env = gym.wrappers.FrameStack(env, 4)

        from dqn_agent import DQN_Agent
        agent = DQN_Agent(env, args)
        agent.train()

    elif args.train_ddpg:
        env = gym.make("Hopper-v2")
        
        from ddpg_agent import DDPG_Agent
        agent = DDPG_Agent(env, args)
        agent.train()
    
    elif args.test_ddpg:
        env = gym.make("Hopper-v2")
        
        from ddpg_agent import DDPG_Agent
        agent = DDPG_Agent(env, args)
        agent.test()
    
if __name__ == "__main__":
    run()
