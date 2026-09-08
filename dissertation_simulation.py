"""
Dissertation simulation: decentralised reputation mechanism.

Architectures:
1. No reputation baseline
2. Reputation without Sybil resistance
3. Full mechanism with gradual reputation access

The seven-variable empirical score is abstracted at mechanism level as:
S approximately theta + epsilon

Run:
    python dissertation_simulation.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass
class Config:
    n_agents: int = 5000
    periods: int = 60
    strategic_share: float = 0.20
    alpha: float = 0.50
    min_history: int = 6
    default_penalty: float = 0.20
    collateral_floor: float = 1.00
    collateral_ceiling: float = 2.50
    fresh_identity_prior: float = 0.50
    signal_noise: float = 0.12
    n_seeds: int = 30
    seed0: int = 1000


def collateral_from_reputation(r, cfg):
    c = cfg.collateral_floor + (1.0 - np.asarray(r)) * (
        cfg.collateral_ceiling - cfg.collateral_floor
    )
    return np.clip(c, cfg.collateral_floor, cfg.collateral_ceiling)


def access_factor(history, cfg):
    if cfg.min_history <= 0:
        return np.ones_like(np.asarray(history, dtype=float))
    return np.minimum(np.asarray(history, dtype=float) / cfg.min_history, 1.0)


def effective_reputation(r, history, cfg):
    a = access_factor(history, cfg)
    return cfg.fresh_identity_prior + a * (np.asarray(r) - cfg.fresh_identity_prior)


def run_scenario(name, cfg, seed):
    rng = np.random.default_rng(seed)

    theta = rng.beta(4.0, 3.0, cfg.n_agents)
    strategic = rng.random(cfg.n_agents) < cfg.strategic_share
    reputation = np.full(cfg.n_agents, cfg.fresh_identity_prior)
    history = np.zeros(cfg.n_agents, dtype=int)

    resets = np.zeros(cfg.n_agents, dtype=int)
    defaults = np.zeros(cfg.n_agents, dtype=int)
    collateral_paid = np.zeros(cfg.n_agents)
    loans = np.zeros(cfg.n_agents, dtype=int)

    period_rows = []

    for t in range(1, cfg.periods + 1):
        if name == "No reputation":
            rep_for_collateral = np.full(cfg.n_agents, cfg.fresh_identity_prior)
        elif name == "Reputation without Sybil resistance":
            rep_for_collateral = reputation.copy()
        elif name == "Full mechanism":
            rep_for_collateral = effective_reputation(reputation, history, cfg)
        else:
            raise ValueError("Unknown scenario")

        collateral = collateral_from_reputation(rep_for_collateral, cfg)

        default_prob = np.clip(0.35 * (1.0 - theta), 0.01, 0.60)
        default_event = rng.random(cfg.n_agents) < default_prob

        signal = np.clip(
            theta + rng.normal(0.0, cfg.signal_noise, cfg.n_agents),
            0.0,
            1.0,
        )
        signal = np.where(
            default_event,
            np.clip(signal - cfg.default_penalty, 0.0, 1.0),
            signal,
        )

        if name != "No reputation":
            reputation = cfg.alpha * signal + (1.0 - cfg.alpha) * reputation

            if name == "Reputation without Sybil resistance":
                current_future = collateral_from_reputation(reputation, cfg)
                fresh_future = collateral_from_reputation(
                    np.full(cfg.n_agents, cfg.fresh_identity_prior), cfg
                )
            else:
                current_future = collateral_from_reputation(
                    effective_reputation(reputation, history + 1, cfg), cfg
                )
                fresh_future = collateral_from_reputation(
                    effective_reputation(
                        np.full(cfg.n_agents, cfg.fresh_identity_prior),
                        np.zeros(cfg.n_agents, dtype=int),
                        cfg,
                    ),
                    cfg,
                )

            reset = strategic & (fresh_future < current_future)
            reputation[reset] = cfg.fresh_identity_prior
            history[reset] = 0
            resets[reset] += 1

        history += 1
        defaults += default_event.astype(int)
        collateral_paid += collateral
        loans += 1

        period_rows.append({
            "scenario": name,
            "seed": seed,
            "period": t,
            "mean_reputation": float(reputation.mean()),
            "mean_effective_reputation": float(rep_for_collateral.mean()),
            "mean_collateral": float(collateral.mean()),
            "defaults_this_period": int(default_event.sum()),
            "strategic_resets_this_period": int((strategic & (resets > 0)).sum()),
        })

    agent_df = pd.DataFrame({
        "scenario": name,
        "seed": seed,
        "agent_id": np.arange(cfg.n_agents),
        "strategic": strategic.astype(int),
        "latent_quality": theta,
        "final_reputation": reputation,
        "final_history": history,
        "resets": resets,
        "defaults": defaults,
        "mean_collateral": collateral_paid / loans,
    })

    return pd.DataFrame(period_rows), agent_df


def run_experiment(cfg, scenarios):
    period_frames = []
    agent_frames = []
    for scenario in scenarios:
        for i in range(cfg.n_seeds):
            p, a = run_scenario(scenario, cfg, cfg.seed0 + i)
            period_frames.append(p)
            agent_frames.append(a)
    return (
        pd.concat(period_frames, ignore_index=True),
        pd.concat(agent_frames, ignore_index=True),
    )


def summarise_agents(agent_df):
    result = (
        agent_df.groupby(["scenario", "strategic"])
        .agg(
            mean_collateral=("mean_collateral", "mean"),
            mean_reputation=("final_reputation", "mean"),
            mean_resets=("resets", "mean"),
            mean_defaults=("defaults", "mean"),
            agents=("agent_id", "count"),
        )
        .reset_index()
    )
    result["borrower_type"] = np.where(
        result["strategic"] == 1, "Strategic", "Non-strategic"
    )
    return result.drop(columns="strategic")


def sensitivity_runs(base_cfg):
    grids = {
        "min_history": [3, 6, 12],
        "alpha": [0.25, 0.50, 0.75],
        "default_penalty": [0.10, 0.20, 0.30],
        "collateral_floor": [0.75, 1.00, 1.25],
    }

    rows = []
    for parameter, values in grids.items():
        for value in values:
            cfg = replace(base_cfg, **{parameter: value})
            _, agents = run_experiment(cfg, ["Full mechanism"])
            strategic_agents = agents[agents["strategic"] == 1]
            rows.append({
                "parameter": parameter,
                "value": value,
                "mean_collateral": agents["mean_collateral"].mean(),
                "mean_reputation": agents["final_reputation"].mean(),
                "mean_resets_strategic": strategic_agents["resets"].mean(),
                "mean_defaults": agents["defaults"].mean(),
            })
    return pd.DataFrame(rows)


def make_figures(period_df, agent_df, outdir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for scenario, sub in period_df.groupby("scenario"):
        series = sub.groupby("period")["mean_collateral"].mean()
        ax.plot(series.index, series.values, label=scenario)
    ax.set_xlabel("Simulation period")
    ax.set_ylabel("Mean collateral requirement")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "mean_collateral_by_period.png", dpi=300)
    plt.close(fig)

    strategic = agent_df[agent_df["strategic"] == 1]
    resets = strategic.groupby("scenario")["resets"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(resets.index, resets.values)
    ax.set_ylabel("Mean identity resets per strategic borrower")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(outdir / "mean_resets_by_scenario.png", dpi=300)
    plt.close(fig)


def main():
    cfg = Config()
    outdir = Path("simulation_outputs")
    outdir.mkdir(exist_ok=True)

    scenarios = [
        "No reputation",
        "Reputation without Sybil resistance",
        "Full mechanism",
    ]

    period_df, agent_df = run_experiment(cfg, scenarios)
    summary_df = summarise_agents(agent_df)
    sensitivity_df = sensitivity_runs(cfg)

    period_df.to_csv(outdir / "period_level_results.csv", index=False)
    agent_df.to_csv(outdir / "agent_level_results.csv", index=False)
    summary_df.to_csv(outdir / "scenario_summary.csv", index=False)
    sensitivity_df.to_csv(outdir / "sensitivity_results.csv", index=False)

    make_figures(period_df, agent_df, outdir)

    print("\nScenario summary")
    print(summary_df.to_string(index=False))
    print("\nSensitivity results")
    print(sensitivity_df.to_string(index=False))
    print("\nSaved all outputs to:", outdir.resolve())


if __name__ == "__main__":
    main()
