"""Live demo: classify a subject that the final model never saw during training.

    python scripts/demo.py                    # random held-out subject
    python scripts/demo.py --subject SIM-0042
    python scripts/demo.py --subject SIM-0042 --age 16.2   # "what-if" on age
    python scripts/demo.py --all              # score every held-out subject
    python scripts/demo.py --parcellation craddock100   # use the Craddock-100 model
    python scripts/demo.py --parcellation craddock100 --all

Requires results/demo_model_<parcellation>.joblib
(created by: python scripts/run_experiments.py --demo-model-only).
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fmri_alcohol.data import load_cohort  # noqa: E402
from fmri_alcohol.evaluate import classification_metrics  # noqa: E402
from fmri_alcohol.features import build_features, connectivity_matrices  # noqa: E402
from fmri_alcohol.preprocess import zscore_timeseries  # noqa: E402

LABELS = {0: "non-heavy drinker", 1: "HEAVY DRINKER"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", help="subject id, e.g. SIM-0042 (default: random held-out)")
    ap.add_argument("--age", type=float, help="override the subject's age (what-if analysis)")
    ap.add_argument("--all", action="store_true", help="evaluate all held-out subjects")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--parcellation", choices=["ica25", "craddock100"], default="ica25",
                    help="which trained model / atlas to use (default: ica25)")
    args = ap.parse_args()

    model_path = ROOT / f"results/demo_model_{args.parcellation}.joblib"
    if not model_path.exists():
        sys.exit(f"{model_path.relative_to(ROOT)} not found. Create it with:\n"
                 f"    python scripts/run_experiments.py --demo-model-only")
    bundle = joblib.load(model_path)
    cohort = load_cohort(ROOT / f"data/{bundle['parcellation']}.npz")
    demo = cohort.demographics.copy()
    held_out = bundle.get("holdout_subjects",
                          np.setdiff1d(np.arange(len(cohort.y)), bundle["train_subjects"]))
    Xz = zscore_timeseries(cohort.X)
    model = bundle["model"]

    if args.all:
        F = build_features(Xz[held_out], demo.iloc[held_out], bundle["brain"], bundle["demographics"])
        p = model.predict_proba(F)[:, 1]
        m = classification_metrics(cohort.y[held_out], p)
        print(f"Held-out subjects: {len(held_out)} "
              f"({int(cohort.y[held_out].sum())} heavy drinkers, {int((cohort.y[held_out] == 0).sum())} controls)")
        for k in ("accuracy", "balanced_accuracy", "recall", "specificity", "auc"):
            print(f"  {k:18s} {m[k]:.3f}")
        return

    rng = np.random.default_rng(args.seed)
    if args.subject:
        matches = np.flatnonzero(demo.subject_id.to_numpy() == args.subject)
        if not len(matches):
            sys.exit(f"unknown subject {args.subject}")
        s = int(matches[0])
        if s not in held_out:
            print("note: this subject was part of the training set")
    else:
        # pick the class first so the demo shows heavy drinkers as often as controls
        label = int(rng.integers(0, 2))
        s = int(rng.choice(held_out[cohort.y[held_out] == label]))

    row = demo.iloc[[s]].copy()
    if args.age is not None:
        row["age"] = args.age
    F = build_features(Xz[[s]], row, bundle["brain"], bundle["demographics"])
    p = float(model.predict_proba(F)[0, 1])
    pred = int(p > 0.5)
    d = row.iloc[0]

    print("=" * 60)
    print(f" Subject {d.subject_id}   age {d.age:.1f}   sex {'M' if d.sex else 'F'}   "
          f"scanner {'Siemens' if d.scanner else 'GE'}")
    print(f" fMRI: {cohort.X.shape[1]} timesteps x {cohort.X.shape[2]} regions ({bundle['parcellation']})")
    print("-" * 60)
    bar = "#" * int(round(p * 40))
    print(f" P(heavy drinker) = {p:.3f}  |{bar:<40}|")
    print(f" Prediction : {LABELS[pred]}")
    print(f" Ground truth: {LABELS[int(cohort.y[s])]}   ->  {'CORRECT' if pred == cohort.y[s] else 'WRONG'}")
    print("=" * 60)

    C = connectivity_matrices(Xz[[s]])[0]
    order = np.argsort(cohort.region_networks, kind="stable")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), gridspec_kw={"width_ratios": [2, 1]})
    for r in range(min(8, Xz.shape[2])):
        axes[0].plot(np.arange(Xz.shape[1]) * 2.2, Xz[s, :, r] + 4 * r, lw=0.7)
    axes[0].set_yticks([])
    axes[0].set_xlabel("time (s)")
    axes[0].set_title(f"{d.subject_id}: BOLD signal (8 regions)")
    im = axes[1].imshow(C[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_title("Functional connectivity")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.suptitle(f"P(heavy drinker) = {p:.2f}  ->  {LABELS[pred]}  (truth: {LABELS[int(cohort.y[s])]})")
    out = ROOT / f"results/demo_subject_{bundle['parcellation']}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f" figure saved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()