"""Turn cached experiment runs into results/REPORT.md plus charts."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lcm.data import T_CTX, HORIZON, normalize
from lcm.model import LCM
from lcm.simulators import SIMULATORS

# reference palette (categorical slots 1-3 + neutral baseline)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, INK, INK2, GRID = "#9a9893", "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False,
})


def skill(m):
    """Forecast skill vs 'nothing changes': 0 = no better, 1 = perfect, <0 = worse."""
    return 1 - m["crps"] / m["persist_crps"]


def _bars(ax, labels, series, title):
    n = len(series)
    h = 0.8 / n
    y = np.arange(len(labels))
    for i, (name, vals, color) in enumerate(series):
        vals = np.array([np.nan if v is None else v for v in vals], float)
        ax.barh(y + (i - (n - 1) / 2) * h, vals, height=h * 0.9, color=color, label=name, edgecolor="#fcfcfb", linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.grid(axis="y", visible=False)


def write_report(out, domains, held_out, cfg, test_sets, dev):
    runs = {f[:-5]: json.load(open(os.path.join(out, "runs", f))) for f in os.listdir(os.path.join(out, "runs"))}
    J = runs["joint"]["eval"]
    S = {n: runs[f"specialist_{n}"]["eval"][n] for n in domains}
    Z = {n: runs[f"loo_{n}"]["eval"][n] for n in held_out}
    FT = {n: runs[f"finetune_{n}"]["eval"][n] for n in held_out}
    SC = {n: runs[f"scratch_{n}"]["eval"][n] for n in held_out}
    lines = []
    w = lines.append

    # ---------- charts
    fig, ax = plt.subplots(figsize=(8, 5.8))
    labels = [f"{n}\n({SIMULATORS[n].field})" for n in domains]
    _bars(ax, labels, [
        ("Specialist (one system only)", [skill(S[n]) for n in domains], ORANGE),
        ("Joint LCM (all 8 systems)", [skill(J[n]) for n in domains], BLUE),
        ("LCM that never saw this system", [skill(Z[n]) if n in Z else None for n in domains], AQUA),
    ], "Forecast skill over the 'nothing changes' baseline (higher is better)")
    ax.set_xlabel("skill = 1 − CRPS / baseline CRPS")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_skill.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    _bars(ax, held_out, [
        (f"Fresh model, {cfg['few_shot_eps']} examples", [skill(SC[n]) for n in held_out], GRAY),
        ("LCM, zero examples", [skill(Z[n]) for n in held_out], AQUA),
        (f"LCM + {cfg['few_shot_eps']} examples", [skill(FT[n]) for n in held_out], BLUE),
        (f"Specialist, {cfg['n_train']} examples", [skill(S[n]) for n in held_out], ORANGE),
    ], "Learning a new system: transfer vs starting from scratch")
    ax.set_xlabel("forecast skill (higher is better)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.22), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_transfer.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(11, 5), sharey=False)
    for ax, n in zip(axes.flat, domains):
        pt, pm = np.array(J[n]["p_true"]), np.array(J[n]["p_model"])
        ax.plot([0, 1], [0, 1], color=GRAY, linewidth=1, linestyle="--")
        ax.scatter(pt, pm, s=18, color=BLUE, edgecolor="#fcfcfb", linewidth=0.8)
        ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"{n}  (err {J[n]['prob_error']:.2f})", fontsize=9, loc="left", color=INK)
    for ax in axes[1]:
        ax.set_xlabel("true odds", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("LCM's odds", fontsize=9)
    fig.suptitle("Does the LCM give the right odds? (\"will this quantity rise?\", one dot per situation)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_odds.png"), dpi=150)
    plt.close(fig)

    model = LCM().to(dev)
    model.load_state_dict(torch.load(os.path.join(out, "checkpoints", "joint.pt"), map_location=dev))
    model.eval()
    torch.manual_seed(0)
    fig, axes = plt.subplots(2, 4, figsize=(11, 5))
    for ax, n in zip(axes.flat, domains):
        d = {k: v[3:4] for k, v in test_sets[n].items()}
        xn, st = normalize(d["x"], d["ent_mask"], d["feat_mask"])
        fut = model.sample_future(xn[:, :T_CTX], d["ent_mask"], d["feat_mask"], d["adj"], st, HORIZON, 200)
        f = SIMULATORS[n].event_f
        series = xn[0, :, 0, f].cpu().numpy()
        samp = fut[0, :, :, 0, f].cpu().numpy()
        tf = np.arange(T_CTX, T_CTX + HORIZON)
        lo, mid, hi = np.percentile(samp, [5, 50, 95], 0)
        ax.fill_between(tf, lo, hi, color=BLUE, alpha=0.2, linewidth=0, label="LCM 90% range")
        ax.plot(tf, mid, color=BLUE, linewidth=2, label="LCM median")
        ax.plot(np.arange(T_CTX), series[:T_CTX], color=INK2, linewidth=2, label="history shown")
        ax.plot(np.arange(T_CTX - 1, T_CTX + HORIZON), series[T_CTX - 1:], color=ORANGE, linewidth=2, label="what happened")
        ax.axvline(T_CTX - 0.5, color=GRAY, linewidth=1, linestyle=":")
        ax.set_title(f"{n}: {SIMULATORS[n].feature_names[f]}", fontsize=9, loc="left", color=INK)
        ax.set_xticks([])
    axes.flat[0].legend(fontsize=7, loc="best")
    fig.suptitle("Sample forecasts from the joint LCM (one entity per system, unseen test episodes)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_forecasts.png"), dpi=150)
    plt.close(fig)

    # ---------- verdicts
    wins = [n for n in domains if J[n]["crps"] < S[n]["crps"]]
    z_beats = [n for n in held_out if Z[n]["crps"] < Z[n]["persist_crps"]]
    ft_beats = [n for n in held_out if FT[n]["crps"] < SC[n]["crps"]]
    cover = np.mean([J[n]["cover90"] for n in domains])
    odds_err = np.mean([J[n]["prob_error"] for n in domains])
    odds_base = np.mean([J[n]["prob_error_baserate"] for n in domains])

    w("# LCM Phase 0 results\n")
    w(f"Config: {cfg['n_train']} training episodes per system, {cfg['steps']} training steps per model, "
      f"{cfg['eval_limit']} test episodes per system. Numbers are in per-episode normalised units.\n")
    w("## The three questions\n")
    w(f"1. **Does one model trained on all systems beat specialists?** The joint model beats the specialist on "
      f"**{len(wins)} of {len(domains)}** systems ({', '.join(wins) or 'none'}), with the same training budget.")
    w(f"2. **Can it forecast a system it has never seen?** Zero-shot it beats the 'nothing changes' baseline on "
      f"**{len(z_beats)} of {len(held_out)}** held-out systems ({', '.join(z_beats) or 'none'}).")
    w(f"3. **Does prior experience make new systems faster to learn?** With only {cfg['few_shot_eps']} examples, "
      f"the pretrained LCM beats a fresh model on **{len(ft_beats)} of {len(held_out)}** ({', '.join(ft_beats) or 'none'}).\n")
    w(f"Odds: across situations, the joint model's odds are off by **{odds_err:.2f}** on average, against "
      f"{odds_base:.2f} for an informed guess that always says the system's average odds. The truth lands inside its 90% range "
      f"**{cover:.0%}** of the time (ideal: 90%).\n")
    w("![skill](fig_skill.png)\n\n![transfer](fig_transfer.png)\n\n![odds](fig_odds.png)\n\n![forecasts](fig_forecasts.png)\n")

    w("## Forecast accuracy per system\n")
    w("CRPS scores the whole predicted distribution (lower is better). The baseline repeats the last observation.\n")
    w("| system | field | baseline | specialist | joint LCM | LCM, never saw it | joint 90% coverage |")
    w("|---|---|---|---|---|---|---|")
    for n in domains:
        z = f"{Z[n]['crps']:.3f}" if n in Z else "-"
        w(f"| {n} | {SIMULATORS[n].field} | {J[n]['persist_crps']:.3f} | {S[n]['crps']:.3f} | {J[n]['crps']:.3f} | {z} | {J[n]['cover90']:.0%} |")
    w("\n## Learning a new system from little data\n")
    w("| held-out system | fresh model, few examples | LCM, zero examples | LCM + few examples | specialist, full data |")
    w("|---|---|---|---|---|")
    for n in held_out:
        w(f"| {n} | {SC[n]['crps']:.3f} | {Z[n]['crps']:.3f} | {FT[n]['crps']:.3f} | {S[n]['crps']:.3f} |")
    w("\n## Odds of an outcome\n")
    w("Event: *the key variable of entity 0 ends the forecast window higher than it is now.* "
      "True odds come from replaying each situation 64 times in the simulator with fresh randomness. "
      "The informed guess knows each system's average odds; the LCM has to work out the odds for each situation.\n")
    w("| system | LCM odds error | informed-guess error |")
    w("|---|---|---|")
    for n in domains:
        w(f"| {n} | {J[n]['prob_error']:.3f} | {J[n]['prob_error_baserate']:.3f} |")
    open(os.path.join(out, "REPORT.md"), "w").write("\n".join(lines) + "\n")
    print(f"report written to {out}/REPORT.md")
