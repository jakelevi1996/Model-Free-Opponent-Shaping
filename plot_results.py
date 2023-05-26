from jutility import plotting, util

table = util.Table().load_json("out_300.json", "server_runs/mfos_ppo_ppo_tracing_step_freeze_100")
plotting.plot(
    plotting.Scatter(*table.get_data("ep", "rew 0"), c="r", label="Rewards for agent 0"),
    plotting.Scatter(*table.get_data("ep", "rew 1"), c="b", label="Rewards for agent 1"),
    plotting.HVSpan(xlims=[0, 100],   color="g", alpha=0.2, label="Vs naive learner"),
    plotting.HVSpan(xlims=[100, 200], color="c", alpha=0.2, label="Self play"),
    plotting.HVSpan(xlims=[200, 300], color="m", alpha=0.2, label="Agent 0 frozen"),
    legend=True,
    plot_name="MFOS self play, 100*NL, 100*SP, 100*agent 0 frozen",
)

table = util.Table().load_json("out_300.json", "server_runs/mfos_ppo_ppo_tracing_step_freeze_reset_100")
plotting.plot(
    plotting.Scatter(*table.get_data("ep", "rew 0"), c="r", label="Rewards for agent 0"),
    plotting.Scatter(*table.get_data("ep", "rew 1"), c="b", label="Rewards for agent 1"),
    plotting.HVSpan(xlims=[0, 100],   color="g", alpha=0.2, label="Vs naive learner"),
    plotting.HVSpan(xlims=[100, 200], color="c", alpha=0.2, label="Self play"),
    plotting.HVSpan(xlims=[200, 300], color="m", alpha=0.2, label="Agent 0 frozen and agent 1 reset"),
    legend=True,
    plot_name="MFOS self play, 100*NL, 100*SP, 100*agent 0 frozen and agent 1 reset",
)

table = util.Table().load_json("out_1200.json", "server_runs/mfos_ppo_ppo_tracing_step_freeze_reset")
plotting.plot(
    plotting.Scatter(*table.get_data("ep", "rew 0"), c="r", label="Rewards for agent 0"),
    plotting.Scatter(*table.get_data("ep", "rew 1"), c="b", label="Rewards for agent 1"),
    plotting.HVSpan(xlims=[0, 100],   color="g", alpha=0.2, label="Vs naive learner"),
    plotting.HVSpan(xlims=[100, 200], color="c", alpha=0.2, label="Self play"),
    plotting.HVSpan(xlims=[200, 1200], color="m", alpha=0.2, label="Agent 0 frozen and agent 1 reset"),
    legend=True,
    plot_name="MFOS self play, 100*NL, 100*SP, 1000*agent 0 frozen and agent 1 reset",
)

table = util.Table().load_json("out_1200.json", "server_runs/mfos_ppo_ppo_tracing_step_freeze")
plotting.plot(
    plotting.Scatter(*table.get_data("ep", "rew 0"), c="r", label="Rewards for agent 0"),
    plotting.Scatter(*table.get_data("ep", "rew 1"), c="b", label="Rewards for agent 1"),
    plotting.HVSpan(xlims=[0, 100],   color="g", alpha=0.2, label="Vs naive learner"),
    plotting.HVSpan(xlims=[100, 200], color="c", alpha=0.2, label="Self play"),
    plotting.HVSpan(xlims=[200, 1200], color="m", alpha=0.2, label="Agent 0 frozen"),
    legend=True,
    plot_name="MFOS self play, 100*NL, 100*SP, 1000*agent 0 frozen",
)

table = util.Table().load_json("out_1200.json", "server_runs/mfos_ppo_ppo_tracing_step_freeze_reset_repeat")
plotting.plot(
    plotting.Scatter(*table.get_data("ep", "rew 0"), c="r", label="Rewards for agent 0"),
    plotting.Scatter(*table.get_data("ep", "rew 1"), c="b", label="Rewards for agent 1"),
    plotting.HVSpan(xlims=[0, 100],   color="g", alpha=0.2, label="Vs naive learner"),
    plotting.HVSpan(xlims=[100, 200], color="c", alpha=0.2, label="Self play"),
    plotting.HVSpan(xlims=[200, 1200], color="m", alpha=0.2, label="Agent 0 frozen and agent 1 reset"),
    legend=True,
    plot_name="MFOS self play, 100*NL, 100*SP, 1000*agent 0 frozen and agent 1 reset (repeat)",
)

table = util.Table().load_json("out_1200.json", "server_runs/mfos_ppo_ppo_tracing_step_freeze_repeat")
plotting.plot(
    plotting.Scatter(*table.get_data("ep", "rew 0"), c="r", label="Rewards for agent 0"),
    plotting.Scatter(*table.get_data("ep", "rew 1"), c="b", label="Rewards for agent 1"),
    plotting.HVSpan(xlims=[0, 100],   color="g", alpha=0.2, label="Vs naive learner"),
    plotting.HVSpan(xlims=[100, 200], color="c", alpha=0.2, label="Self play"),
    plotting.HVSpan(xlims=[200, 1200], color="m", alpha=0.2, label="Agent 0 frozen"),
    legend=True,
    plot_name="MFOS self play, 100*NL, 100*SP, 1000*agent 0 frozen (repeat)",
)

table = util.Table().load_json("out_1200.json", "server_runs/mfos_ppo_tracing_step_freeze_reset_repeat_varpar")
plotting.plot(
    plotting.Scatter(*table.get_data("ep", "rew 0"), c="r", label="Rewards for agent 0"),
    plotting.Scatter(*table.get_data("ep", "rew 1"), c="b", label="Rewards for agent 1"),
    plotting.HVSpan(xlims=[0, 100],   color="g", alpha=0.2, label="Vs naive learner"),
    plotting.HVSpan(xlims=[100, 200], color="c", alpha=0.2, label="Self play"),
    plotting.HVSpan(xlims=[200, 1200], color="m", alpha=0.2, label="Agent 0 frozen and agent 1 reset"),
    legend=True,
    plot_name="MFOS self play, 100*NL, 100*SP, 1000*agent 0 frozen and agent 1 reset with parameterised variance",
)

for experiment_name in [
    "mfos_ppo_tracing_step_freeze_recurrent_seed_0",
    "mfos_ppo_tracing_step_freeze_recurrent_seed_1",
    "mfos_ppo_tracing_step_freeze_recurrent_seed_2",
    "mfos_ppo_tracing_step_freeze_recurrent_seed_0_batch_size_2000",
    "mfos_ppo_tracing_step_freeze_recurrent_seed_1_batch_size_2000",
]:
    table = util.Table().load_json("out_1200.json", "server_runs/%s" % experiment_name)
    plotting.plot(
        plotting.Scatter(*table.get_data("ep", "rew 0"), c="r", label="Rewards for agent 0"),
        plotting.Scatter(*table.get_data("ep", "rew 1"), c="b", label="Rewards for agent 1"),
        plotting.HVSpan(xlims=[0, 100],   color="g", alpha=0.2, label="Vs naive learner"),
        plotting.HVSpan(xlims=[100, 200], color="c", alpha=0.2, label="Self play"),
        plotting.HVSpan(xlims=[200, 1200], color="m", alpha=0.2, label="Agent 0 frozen and agent 1 reset"),
        legend=True,
        plot_name="MFOS self play, 100*NL, 100*SP, 1000*agent 0 frozen and agent 1 reset %s" % experiment_name,
    )

for game in ["pennies", "awkward", "IPD"]:
    for seed in [1, 2, 3]:
        experiment_name = "mfos_self_tracing_exploit_%s_seed_%i" % (game, seed)
        table = util.Table().load_json("out_2048.json", "server_runs/%s" % experiment_name)
        plotting.plot(
            plotting.Scatter(*table.get_data("ep", "rew 0"), c="r", zorder=20, label="Rewards for agent 0"),
            plotting.Scatter(*table.get_data("ep", "rew 1"), c="b", zorder=20, label="Rewards for agent 1"),
            plotting.HVSpan(xlims=[0, 1024], color="c", alpha=0.2, label="Self play (tracing)"),
            plotting.HVSpan(xlims=[1024, 2048], color="m", alpha=0.2, label="Agent 0 frozen and agent 1 reset"),
            axis_properties=plotting.AxisProperties("Episode", "Reward", xlim=[0, 2048]),
            legend=True,
            plot_name=experiment_name.replace("_", " ").title(),
        )

for seed in [1, 2, 3]:
    experiment_name = "mfos_self_tracing_exploit_matching_pennies_seed_%i_num_sp_3000" % seed
    table = util.Table().load_json("out_4000.json", "server_runs/%s" % experiment_name)
    plotting.plot(
        plotting.Scatter(*table.get_data("ep", "rew 0"), c="r", zorder=20, label="Rewards for agent 0"),
        plotting.Scatter(*table.get_data("ep", "rew 1"), c="b", zorder=20, label="Rewards for agent 1"),
        plotting.HVSpan(xlims=[0, 3000], color="c", alpha=0.2, label="Self play (tracing)"),
        plotting.HVSpan(xlims=[3000, 4000], color="m", alpha=0.2, label="Agent 0 frozen and agent 1 reset"),
        axis_properties=plotting.AxisProperties("Episode", "Reward", xlim=[0, 4000]),
        legend=True,
        plot_name=experiment_name.replace("_", " ").title(),
    )

    for strategy in ["ALLC", "ALLD", "TFT", "LOLA", "random_static"]:
        experiment_name = "mfos_self_tracing_exploit_IPD_seed_%i_opponent_%s" % (
            seed if (strategy in ["ALLC", "LOLA", "random_static"]) else (seed - 1),
            strategy,
        )
        table = util.Table().load_json("out_2048.json", "server_runs/%s" % experiment_name)
        plotting.plot(
            plotting.Scatter(*table.get_data("ep", "rew 0"), c="r", zorder=20, label="Rewards for agent 0"),
            plotting.Scatter(*table.get_data("ep", "rew 1"), c="b", zorder=20, label="Rewards for agent 1"),
            plotting.HVSpan(xlims=[0, 1024], color="c", alpha=0.2, label="Self play (tracing)"),
            plotting.HVSpan(xlims=[1024, 2048], color="m", alpha=0.2, label="Agent 0 frozen and agent 1 reset"),
            axis_properties=plotting.AxisProperties("Episode", "Reward", xlim=[0, 2048]),
            legend=True,
            plot_name=experiment_name.title().replace(strategy.title(), strategy.upper()).replace("Ipd", "IPD").replace("_", " "),
        )
