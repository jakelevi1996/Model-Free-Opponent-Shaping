import torch
from ppo import PPO, Memory
from environments import MetaGames, SymmetricMetaGames
import os
import argparse
import json
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--game", type=str, required=True)
parser.add_argument("--entropy", type=float, default=0.01)
parser.add_argument("--exp-name", type=str, default="runs/mfos_ppo_ppo_alternating")
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--alternate-every", type=int, default=10)
parser.add_argument("--offset-action", action="store_true")
args = parser.parse_args()


if __name__ == "__main__":
    ############################################
    K_epochs = 4  # update policy for K epochs

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr = 0.0002  # parameters for Adam optimizer
    betas = (0.9, 0.999)

    max_episodes = 1024
    batch_size = 4096
    random_seed = None
    num_steps = 100

    save_freq = 1000

    name = "%s_%s_alternate_every_%i" % (args.exp_name, args.offset_action, args.alternate_every)

    print(f"RUNNING NAME: {name}")
    if not os.path.isdir(name):
        os.makedirs(name)
        with open(os.path.join(name, "commandline_args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    #############################################

    # creating environment
    env = SymmetricMetaGames(batch_size, game=args.game, offset_action=args.offset_action)

    action_dim = env.d
    state_dim = env.d * 2

    memory_0 = Memory()
    memory_1 = Memory()

    ppo_0 = PPO(state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip, entropy=args.entropy)
    ppo_1 = PPO(state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip, entropy=args.entropy)

    nl_env = MetaGames(batch_size, game=args.game, offset_action=args.offset_action)

    print(lr, betas)
    # training loop
    rew_means = []

    for i_episode in range(1, max_episodes + 1):
        print("=" * 100)
        print(i_episode)
        state_0, state_1 = env.reset()

        running_reward_0 = torch.zeros(batch_size).cuda()
        running_reward_1 = torch.zeros(batch_size).cuda()

        for t in range(num_steps):

            # Running policy_old:
            action_0 = ppo_0.policy_old.act(state_0, memory_0)
            action_1 = ppo_1.policy_old.act(state_1, memory_1)
            states, rewards, M = env.step(action_0, action_1)
            state_0, state_1 = states
            reward_0, reward_1 = rewards

            memory_0.rewards.append(reward_0)
            memory_1.rewards.append(reward_1)

            running_reward_0 += reward_0.squeeze(-1)
            running_reward_1 += reward_1.squeeze(-1)

        if ((i_episode // args.alternate_every) % 2) == 0:
            print("Agent 0 update")
            ppo_0.update(memory_0)
        else:
            print("Agent 1 update")
            ppo_1.update(memory_1)

        memory_0.clear_memory()
        memory_1.clear_memory()

        l0 = -running_reward_0.mean() / num_steps
        l1 = -running_reward_1.mean() / num_steps
        print(f"loss 0: {l0}")
        print(f"loss 1: {l1}")
        print(f"sum: {l0 + l1}")

        rew_means.append(
            {
                "ep": i_episode,
                "other": True,
                "rew 0": -l0.item(),
                "rew 1": -l1.item(),
            }
        )


        if i_episode % save_freq == 0:
            ppo_0.save(os.path.join(name, f"{i_episode}_0.pth"))
            ppo_1.save(os.path.join(name, f"{i_episode}_1.pth"))
            with open(os.path.join(name, f"out_{i_episode}.json"), "w") as f:
                json.dump(rew_means, f)
            print(f"SAVING! {i_episode}")

    ppo_0.save(os.path.join(name, f"{i_episode}_0.pth"))
    ppo_1.save(os.path.join(name, f"{i_episode}_1.pth"))
    with open(os.path.join(name, f"out_{i_episode}.json"), "w") as f:
        json.dump(rew_means, f)
    print(f"SAVING! {i_episode}")
