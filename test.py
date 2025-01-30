import os
import time
import argparse
import torch
import pickle
import mujoco
import numpy as np
import transforms3d as tf3


from envs.robots.jvrc_walk import JvrcWalkEnv
from envs.robots.talos_walk import TalosWalkEnv



def print_reward(ep_rewards):
    mean_rewards = {k:[] for k in ep_rewards[-1].keys()}
    print('*********************************')
    for key in mean_rewards.keys():
        l = [step[key] for step in ep_rewards]
        mean_rewards[key] = sum(l)/len(l)
        print(key, ': ', mean_rewards[key])
        #total_rewards = [r for step in ep_rewards for r in step.values()]
    print('*********************************')
    print("mean per step reward: ", sum(mean_rewards.values()))




def run(env, policy, args):
    observation = env.reset()

    env.render()
    
    
    viewer = env.viewer
    viewer._paused = False
    done = False
    ts, end_ts = 0, 2000
    ep_rewards = []

    while (ts < end_ts):
        print("ts: ", ts)
        if hasattr(env, 'frame_skip'):
            start = time.time()

        with torch.no_grad():
            action = policy.forward(torch.Tensor(observation), deterministic=True).detach().numpy()

        print("action: ", action)
        observation, _, done, info = env.step(action.copy())


        print("info: ", info)


        ep_rewards.append(info)


        if args.sync and hasattr(env, 'frame_skip'):
            end = time.time()
            sim_dt = env.robot.client.sim_dt()
            delaytime = max(0, env.frame_skip / (1/sim_dt) - (end-start))
            time.sleep(delaytime)

        if args.quit_on_done and done:
            break
        

        env.render()

    
        ts+=1

    print("Episode finished after {} timesteps".format(ts))
    print_reward(ep_rewards)
    env.close()

def main():
    # get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",
                        required=True,
                        type=str,
                        help="Select the environment: 'talos' or 'jvrc'.")

    parser.add_argument("--path",
                        required=True,
                        type=str,
                        help="path to trained model dir",
    )
    parser.add_argument("--sync",
                        required=False,
                        action="store_true",
                        help="sync the simulation speed with real time",
    )
    parser.add_argument("--quit-on-done",
                        required=False,
                        action="store_true",
                        help="Exit when done condition is reached",
    )


    args = parser.parse_args()

    path_to_actor = ""
    path_to_pkl = ""
    if os.path.isfile(args.path) and args.path.endswith(".pt"):
        path_to_actor = args.path
        path_to_pkl = os.path.join(os.path.dirname(args.path), "experiment.pkl")
    if os.path.isdir(args.path):
        path_to_actor = os.path.join(args.path, "actor.pt")
        path_to_pkl = os.path.join(args.path, "experiment.pkl")

    # load experiment args
    run_args = pickle.load(open(path_to_pkl, "rb"))
    # load trained policy
    policy = torch.load(path_to_actor)
    policy.eval()
    
    if args.env.lower() == "talos":
        env = TalosWalkEnv()  
    elif args.env.lower() == "jvrc":
        env = JvrcWalkEnv()  
    run(env, policy, args)



if __name__=='__main__':
    main()
