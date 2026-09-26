"""Command-line entry point.

Examples
--------
Run the full benchmark (all bundled datasets, all models) and print
markdown tables:

    python -m forecasting.cli benchmark

Run one dataset and also save a forecast+interval plot:

    python -m forecasting.cli benchmark --dataset airline --plot results/airline.png
"""
from __future__ import annotations

import argparse

from forecasting.experiment import format_results_table, run_experiment


def _plot(dataset: str, results, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lstm = next(r for r in results if r.name == "LSTM (delta)")
    n = len(lstm.y_true)
    x = range(n)

    nominal_pct = int(round(lstm.coverage["nominal_coverage"] * 100))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, lstm.y_true, label="actual", color="#222222", linewidth=1.5)
    ax.plot(x, lstm.point_pred.reshape(-1), label="LSTM forecast", color="#1f77b4", linewidth=1.2)
    ax.fill_between(
        x,
        lstm.lower.reshape(-1),
        lstm.upper.reshape(-1),
        color="#1f77b4",
        alpha=0.2,
        label=f"{nominal_pct}% conformal interval",
    )
    ax.set_title(
        f"{dataset}: LSTM forecast with split-conformal interval "
        f"(empirical coverage {lstm.coverage['empirical_coverage'] * 100:.1f}%, "
        f"nominal {lstm.coverage['nominal_coverage'] * 100:.0f}%)"
    )
    ax.set_xlabel("test-set time step")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="conformal-forecast benchmark runner")
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("benchmark", help="run the model benchmark on one or all datasets")
    bench.add_argument(
        "--dataset",
        choices=["airline", "temperature", "synthetic", "all"],
        default="all",
    )
    bench.add_argument("--alpha", type=float, default=0.1, help="miscoverage rate (default 0.1 -> 90%% intervals)")
    bench.add_argument("--plot", type=str, default=None, help="save a forecast+interval PNG for --dataset (not 'all')")
    bench.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.command == "benchmark":
        datasets = ["airline", "temperature", "synthetic"] if args.dataset == "all" else [args.dataset]
        if args.plot and args.dataset == "all":
            parser.error("--plot requires a single --dataset, not 'all'")
        for ds in datasets:
            results = run_experiment(ds, alpha=args.alpha, seed=args.seed)
            print(format_results_table(ds, args.alpha, results))
            print()
            if args.plot:
                _plot(ds, results, args.plot)
                print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
