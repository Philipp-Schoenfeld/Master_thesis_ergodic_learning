#!/usr/bin/env python3
r"""
plot_policy_a_masked.py
=========================
Diagrams for the newly trained MLP value/regret model "Policy A"
(`exploration_optimierung/policy/ablage/policy_a_maske.pt`, trained under the
random-unknown-region masking, `--zufallsmaske`). Read-only: no new
missions/trajectories are generated here, only the already-computed
cross-validation results in
`exploration_optimierung/results/policy_a_training_maske.json` and the
oracle upper-bound table in `.../policy_orakel_maske.json` are plotted.

Kept separate from `run_eval_matrix.py`'s `plot_metric_bars`: that pipeline
scores single generated trajectories per shape against a fixed
`knowledge_condition` (ground_truth/half_known/none_known/ten_samples), with
no `n_exec` axis. Policy A instead picks one of 48 candidate configs per
*round* of a multi-round mission under a randomly sized unknown region
(`unbekannt_bereich=[0.5, 0.9]`) -- the two row formats don't line up, so
this gets its own plots rather than being folded into `plot_metric_bars`.

    python plot_policy_a_masked.py
"""

import json
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
RESULTS_DIR = os.path.join(_arch, 'exploration_optimierung', 'results')
TRAINING_JSON = os.path.join(RESULTS_DIR, 'policy_a_training_maske.json')
ORACLE_JSON = os.path.join(RESULTS_DIR, 'policy_orakel_maske.json')
OUT_DIR = os.path.join(_here, 'results', 'policy_a_masked', 'plots')

#: Same blue used elsewhere in the project to mark "the thing to look at
#: first" in a diagram (see `run_eval_matrix.OPTUNA_IDEAL_COLOR`, which uses
#: green for the same purpose on the Optuna-tuned config -- blue here keeps
#: the two clearly distinct since both can appear in the same write-up).
POLICY_A_COLOR = '#1565C0'
NEUTRAL_COLOR = '#9E9E9E'
BAR_COLOR = '#607D8B'


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def plot_regret_by_fold(training, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    folds = training['falten']
    mean = training['mittel']
    series = ['bedauern_modell', 'bedauern_fest', 'bedauern_zufall']
    series_label = {'bedauern_modell': 'model (Policy A / MLP)',
                    'bedauern_fest': 'fixed policy',
                    'bedauern_zufall': 'random policy'}

    rows = []  # (label, value, is_policy_a)
    for f in folds:
        for s in series:
            rows.append((f"fold {f['falte']} – {series_label[s]}", f[s],
                        s == 'bedauern_modell'))
    for s in series:
        rows.append((f"mean – {series_label[s]}", mean[s],
                    s == 'bedauern_modell'))

    fig, ax = plt.subplots(figsize=(8, 7), facecolor='white')
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    ax.barh(names, vals, color=BAR_COLOR, alpha=0.85)
    for label, (name, val, is_a) in zip(ax.get_yticklabels(), rows):
        if is_a:
            label.set_color(POLICY_A_COLOR)
            label.set_fontweight('bold')
    ax.set_xlabel('regret (mean per-decision quality gap to the best available candidate)')
    ax.set_title('Policy A: 5-fold cross-validated regret vs. fixed/random baselines\n'
                '(random-unknown-region masking)', color='#1A1A2E', fontsize=11)
    fig.text(0.5, 0.965,
             r'regret$(\pi)$ = mean over decisions of $(q^*_t - q_{\pi,t})$',
             ha='center', va='top', fontsize=10, color='#1A1A2E')
    fig.text(0.5, 0.935, 'smaller = better', ha='center', va='top',
             fontsize=9, color='#555', style='italic')
    ax.grid(alpha=0.2, axis='x')
    ax.set_facecolor('white')
    ax.tick_params(labelsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(os.path.join(out_dir, 'regret_by_fold.png'), dpi=130, facecolor='white')
    plt.close(fig)


def plot_value_model_diagnostics(training, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    folds = training['falten']
    fold_ids = [f['falte'] for f in folds]
    mse = [f['mse_val'] for f in folds]
    hit = [f['trefferquote'] for f in folds]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor='white')
    axes[0].bar([str(i) for i in fold_ids], mse, color=POLICY_A_COLOR, alpha=0.85)
    axes[0].set_title('value-model MSE per fold (smaller = better)', fontsize=9, color='#1A1A2E')
    axes[0].set_xlabel('fold'); axes[0].set_ylabel('MSE (predicted vs. realised quality)')
    axes[1].bar([str(i) for i in fold_ids], hit, color=POLICY_A_COLOR, alpha=0.85)
    axes[1].set_title('hit rate per fold: picked the truly best candidate\n(bigger = better)',
                      fontsize=9, color='#1A1A2E')
    axes[1].set_xlabel('fold'); axes[1].set_ylabel('hit rate')
    for ax in axes:
        ax.grid(alpha=0.2, axis='y'); ax.set_facecolor('white')
    fig.suptitle('Policy A (MLP) -- value-model diagnostics',
                color=POLICY_A_COLOR, fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(os.path.join(out_dir, 'value_model_diagnostics.png'), dpi=130, facecolor='white')
    plt.close(fig)


def plot_oracle_curve(oracle, out_dir):
    """Context, not Policy A itself: the greedy one-step-lookahead oracle's
    J(n)/q(n) upper bound over mission rounds, same random-mask condition.
    Plotted in the project's neutral color (not blue) since it isn't the
    trained policy -- see module docstring.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    table = sorted(oracle['tabelle'], key=lambda r: r['n_exec'])
    n = [r['n_exec'] for r in table]
    J = [r['J'] for r in table]
    q = [r['q'] for r in table]
    erg = [r['erg_truth'] for r in table]
    best_n = oracle['n_exec']

    fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
    ax.plot(n, J, marker='o', color='#E65100', label='J (n)')
    ax.plot(n, q, marker='o', color=NEUTRAL_COLOR, label='q (n), relative coverage error')
    ax.plot(n, erg, marker='o', color='#8E24AA', label='erg_truth (n)')
    ax.axvline(best_n, color='#1A1A2E', linestyle='--', linewidth=1,
              label=f'oracle optimum n={best_n}')
    ax.set_xlabel('mission rounds executed (n_exec)')
    ax.set_ylabel('metric value')
    ax.set_title('Greedy one-step-lookahead oracle (upper bound, not Policy A)\n'
                'random-unknown-region masking', fontsize=10, color='#1A1A2E')
    ax.legend(fontsize=8); ax.grid(alpha=0.2); ax.set_facecolor('white')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'oracle_J_q_vs_rounds.png'), dpi=130, facecolor='white')
    plt.close(fig)


def main():
    if not os.path.isfile(TRAINING_JSON):
        raise FileNotFoundError(TRAINING_JSON)
    if not os.path.isfile(ORACLE_JSON):
        raise FileNotFoundError(ORACLE_JSON)
    os.makedirs(OUT_DIR, exist_ok=True)

    training = load(TRAINING_JSON)
    oracle = load(ORACLE_JSON)

    plot_regret_by_fold(training, OUT_DIR)
    plot_value_model_diagnostics(training, OUT_DIR)
    plot_oracle_curve(oracle, OUT_DIR)

    print(f"Policy A (masked) plots -> {OUT_DIR}")
    print(f"  mean regret: model={training['mittel']['bedauern_modell']:.4f}  "
         f"fixed={training['mittel']['bedauern_fest']:.4f}  "
         f"random={training['mittel']['bedauern_zufall']:.4f}")
    print(f"  oracle optimum: n_exec={oracle['n_exec']}  J={oracle['J']:.4f}  q={oracle['q']:.4f}")


if __name__ == '__main__':
    main()
