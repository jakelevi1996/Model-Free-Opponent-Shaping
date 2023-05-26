import json
from jutility import plotting, util

with open("runs/mfos_ppo_ppo_tracing/out_1024.json") as f:
    data = json.load(f)

episodes = [row["ep"] for row in data]
r0 = [row["rew 0"] for row in data]
r1 = [row["rew 1"] for row in data]

line_kwargs = {"alpha": 0.2, "marker": "o", "ls": ""}
plotting.plot(
    plotting.Line(episodes, r0, c="r", label="Rewards for agent 0", **line_kwargs),
    plotting.Line(episodes, r1, c="b", label="Rewards for agent 1", **line_kwargs),
    legend_properties=plotting.LegendProperties(),
    plot_name="MFOS self play with tracing procedure",
)

with open("runs/mfos_ppo_ppo_alternating/out_1024.json") as f:
    data = json.load(f)

episodes = [row["ep"] for row in data]
r0 = [row["rew 0"] for row in data]
r1 = [row["rew 1"] for row in data]

line_kwargs = {"alpha": 0.2, "marker": "o", "ls": ""}
plotting.plot(
    plotting.Line(episodes, r0, c="r", label="Rewards for agent 0", **line_kwargs),
    plotting.Line(episodes, r1, c="b", label="Rewards for agent 1", **line_kwargs),
    legend_properties=plotting.LegendProperties(),
    plot_name="MFOS self play with alternating gradient descent",
)

with open("runs/mfos_ppo_ppo_vanilla/out_1024.json") as f:
    data = json.load(f)

episodes = [row["ep"] for row in data]
r0 = [row["rew 0"] for row in data]
r1 = [row["rew 1"] for row in data]

line_kwargs = {"alpha": 0.2, "marker": "o", "ls": ""}
plotting.plot(
    plotting.Line(episodes, r0, c="r", label="Rewards for agent 0", **line_kwargs),
    plotting.Line(episodes, r1, c="b", label="Rewards for agent 1", **line_kwargs),
    legend_properties=plotting.LegendProperties(),
    plot_name="MFOS vanilla self play",
)

with open("runs/mfos_ppo_ppo_tracing_step/out_200.json") as f:
    data = json.load(f)

episodes = [row["ep"] for row in data]
r0 = [row["rew 0"] for row in data]
r1 = [row["rew 1"] for row in data]

line_kwargs = {"alpha": 0.2, "marker": "o", "ls": ""}
plotting.plot(
    plotting.Line(episodes, r0, c="r", label="Rewards for agent 0", **line_kwargs),
    plotting.Line(episodes, r1, c="b", label="Rewards for agent 1", **line_kwargs),
    legend_properties=plotting.LegendProperties(),
    plot_name="MFOS vs NL for 100 episodes + MFOS vs MFOS for 100 episodes",
)

for alternate_every in [5, 10, 20, 50]:
    with open("server_runs/mfos_ppo_ppo_alternating_alternate_every_%i/out_1024.json" % alternate_every) as f:
        data = json.load(f)

    episodes = [row["ep"] for row in data]
    r0 = [row["rew 0"] for row in data]
    r1 = [row["rew 1"] for row in data]

    line_kwargs = {"alpha": 0.2, "marker": "o", "ls": ""}
    plotting.plot(
        plotting.Line(episodes, r0, c="r", label="Rewards for agent 0", **line_kwargs),
        plotting.Line(episodes, r1, c="b", label="Rewards for agent 1", **line_kwargs),
        legend_properties=plotting.LegendProperties(),
        plot_name="MFOS self play alternating every %i steps" % alternate_every,
    )

for alternate_every in [5, 10, 50, 100]:
    with open("server_runs/mfos_ppo_ppo_alternating_True_alternate_every_%i/out_1024.json" % alternate_every) as f:
        data = json.load(f)

    episodes = [row["ep"] for row in data]
    r0 = [row["rew 0"] for row in data]
    r1 = [row["rew 1"] for row in data]

    line_kwargs = {"alpha": 0.2, "marker": "o", "ls": ""}
    plotting.plot(
        plotting.Line(episodes, r0, c="r", label="Rewards for agent 0", **line_kwargs),
        plotting.Line(episodes, r1, c="b", label="Rewards for agent 1", **line_kwargs),
        legend_properties=plotting.LegendProperties(),
        plot_name="MFOS self play with naive action offsets, alternating every %i steps" % alternate_every,
    )
