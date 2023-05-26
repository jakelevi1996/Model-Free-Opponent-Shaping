import torch
from ppo import PPO, PpoNaiveInit, Memory
from environments import MetaGames
import os
import argparse
import json
import sys
sys.path.append(".")
from jutility import plotting, util

parser = argparse.ArgumentParser()
parser.add_argument("--game",           type=str,   required=True)
parser.add_argument("--opponent",       type=str,   required=True)
parser.add_argument("--entropy",        type=float, default=0.01)
parser.add_argument("--exp-name",       type=str,   default=None)
parser.add_argument("--checkpoint",     type=str,   default="")
parser.add_argument("--mamaml-id",      type=int,   default=0)
parser.add_argument("--max_episodes",   type=int,   default=100)
parser.add_argument("--naivify",        action="store_true")
parser.add_argument("--ppo_update",     action="store_true")
args = parser.parse_args()

def main():
    ############################################
    K_epochs = 4  # update policy for K epochs

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr = 0.002  # parameters for Adam optimizer
    betas = (0.9, 0.999)

    # max_episodes = 1024
    batch_size = 4096
    random_seed = None
    num_steps = 100

    save_freq = 250
    name = args.exp_name
    if name is None:
        name = (
            "runs/mfos_%s_naivify_%s_ppo_update_%s_max_episodes_%i"
            % (args.opponent, args.naivify, args.ppo_update, args.max_episodes)
        )

    print(f"RUNNING NAME: {name}")
    if not os.path.isdir(name):
        os.mkdir(name)
        with open(os.path.join(name, "commandline_args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    #############################################

    # creating environment
    env = MetaGames(batch_size, opponent=args.opponent, game=args.game, mmapg_id=args.mamaml_id)

    action_dim = env.d
    state_dim = env.d * 2

    memory = Memory()
    ppo = PpoNaiveInit(state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip, args.entropy)

    if args.naivify:
        ppo.naivify(batch_size)

    if args.checkpoint:
        ppo.load(args.checkpoint)

    print(lr, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    # training loop
    rew_means = []

    for i_episode in range(1, args.max_episodes + 1):
        state = env.reset()

        running_reward = torch.zeros(batch_size).cuda()
        running_opp_reward = torch.zeros(batch_size).cuda()

        last_reward = 0

        for t in range(num_steps):

            # Running policy_old:
            action = ppo.policy_old.act(state, memory)

            if args.naivify:
                action += ppo.action_offset

            state, reward, info, M = env.step(action, detach_rewards=False, detach_actions=False)

            if args.naivify:
                ppo.update_action_offset(reward)

            memory.rewards.append(reward.detach())
            running_reward += reward.detach().squeeze(-1)
            running_opp_reward += info.squeeze(-1)
            last_reward = reward.squeeze(-1)

        # If ppo_update is False, and naivify is True, we should see behaviour
        # equivalent to naive vs naive, IE both agents converge to mutual
        # defection

        if args.ppo_update:
            ppo.update(memory)
        memory.clear_memory()

        print("=" * 100, flush=True)

        print(f"episode: {i_episode}", flush=True)

        print(f"loss: {-running_reward.mean() / num_steps}", flush=True)

        rew_means.append(
            {
                "rew": (running_reward.mean() / num_steps).item(),
                "opp_rew": (running_opp_reward.mean() / num_steps).item(),
            }
        )

        print(f"opponent loss: {-running_opp_reward.mean() / num_steps}", flush=True)

        if i_episode % save_freq == 0:
            ppo.save(os.path.join(name, f"{i_episode}.pth"))
            with open(os.path.join(name, f"out_{i_episode}.json"), "w") as f:
                json.dump(rew_means, f)
            print(f"SAVING! {i_episode}")

        min_max = lambda t: "[%f, %f]" % (torch.min(t).item(), torch.max(t).item())
        print(
            "Weight range = %s, bias mean range = %s, bias cov range = %s"
            % (
                min_max(ppo.policy.actor[-1].weight),
                min_max(ppo.policy.actor[-1].bias[:5]),
                min_max(ppo.policy.actor[-1].bias[5:]),
            )
        )

    ppo.save(os.path.join(name, f"{i_episode}.pth"))
    with open(os.path.join(name, f"out_{i_episode}.json"), "w") as f:
        json.dump(rew_means, f)
    print(f"SAVING! {i_episode}")

    episodes = list(range(len(rew_means)))
    r0 = [row["rew"] for row in rew_means]
    r1 = [row["opp_rew"] for row in rew_means]

    line_kwargs = {"alpha": 0.2, "marker": "o", "ls": ""}
    plotting.plot(
        plotting.Line(episodes, r0, c="r", label="Rewards for MFOS", **line_kwargs),
        plotting.Line(episodes, r1, c="b", label="Rewards for NL", **line_kwargs),
        legend_properties=plotting.LegendProperties(),
        plot_name=(
            "MFOS vs NL, naivify = %s, PPO updates = %s, max_episodes = %i"
            % (args.naivify, args.ppo_update, args.max_episodes)
        ),
    )

if __name__ == "__main__":
    with util.Timer("Main function"):
        main()
