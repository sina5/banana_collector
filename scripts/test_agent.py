import sys

sys.path.insert(0, "../mlagents")
sys.path.insert(1, "../dqn")

import os
from argparse import ArgumentParser

import numpy as np
import torch
from dqn_agent import Agent
from unityagents import UnityEnvironment


def parse_args():
    parser = ArgumentParser(
        prog="Banana Collector Test",
        description="Tests a trained RL agent to collect bananas",
    )
    parser.add_argument(
        "-c",
        "--checkpoint-file",
        type=str,
        help="Path to a trained Pytorch model checkpoint",
        default="../checkpoints/checkpoint_532.pth",
    )
    parser.add_argument(
        "-u",
        "--unity-app",
        type=str,
        help="Path to a banana collector unity app",
        default="../Banana.app",
    )
    return vars(parser.parse_args())


def start_unity_env(file_name, worker_id=10):
    env = UnityEnvironment(file_name=file_name, worker_id=worker_id)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    print(env.brain_names)
    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]

    # number of agents in the environment
    print("Number of agents:", len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print("Number of actions:", action_size)
    # examine the state space
    state = env_info.vector_observations[0]
    state_size = len(state)
    print("States have length:", state_size)
    return env, brain_name, state_size, action_size


def run_untrained_agent(env, brain_name):
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state
    score = 0  # initialize the score
    while True:
        action = np.random.randint(action_size)  # select an action
        env_info = env.step(action)[
            brain_name
        ]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        if done:  # exit loop if episode finished
            break

    print("[-] Score: {}".format(score))


def test_trained_agent(
    env, brain_name, check_point_path, state_size, action_size, seed=0
):

    agent = Agent(
        state_size=state_size,
        action_size=action_size,
        seed=seed,
    )

    if os.path.exists(check_point_path):
        agent.qnetwork_local.load_state_dict(torch.load(check_point_path))

    score = 0  # initialize the score
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    for i in range(3000):
        action = agent.act(state)
        env.step(action)
        env_info = env.step(action)[brain_name]
        state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        score += reward  # update the score
        done = env_info.local_done[0]
        if done:
            break

    print("[-] Score: {}".format(score))


if __name__ == "__main__":
    args = parse_args()

    ua = args.get("unity_app", "")
    if os.path.exists(ua):
        env, brain_name, state_size, action_size = start_unity_env(ua)
    else:
        raise FileNotFoundError

    print("[>] Try an untrained agent to collect bananas.")
    run_untrained_agent(env, brain_name)

    cf = args.get("checkpoint_file", "")
    if os.path.exists(cf):
        print("[>] Try a trained DQN agent to collect bananas.")
        test_trained_agent(env, brain_name, cf, state_size, action_size)
    else:
        raise FileNotFoundError

    env.close()
