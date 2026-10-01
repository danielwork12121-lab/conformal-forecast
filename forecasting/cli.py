"""Command-line entry point.

Examples
--------
Run the full benchmark (all bundled datasets, all models) and print
markdown tables:

    python -m forecasting.cli benchmark

Run one dataset and also save a forecast+interval plot:

    python -m forecasting.cli benchmark --dataset airline --plot results/airline.png

Compare static split conformal against Adaptive Conformal Inference on the
LSTM model (the one the plain benchmark shows under-covering) and save a
plot of how ACI's alpha_t adapts over the test window:

    python -m forecasting.cli adaptive --dataset airline --plot results/airline_adaptive.png

Compare static split conformal, fixed-pool ACI, and the sliding/growing-pool
ACI variant (which folds each step's own residual into the pool instead of
staying frozen at the calibration set) and save a plot of the pool growing
over the test window:

    python -m forecasting.cli sliding-window --dataset airline --plot results/airline_sliding_window.png

Sweep the sliding-pool ACI's `window` size and see how coverage gap responds
(a bounded window can beat the unbounded growing buffer -- see the README):

    python -m forecasting.cli window-sweep --dataset airline --windows 10,15,20,30,50,unbounded --plot results/airline_window_sweep.png
"""
from __future__ import annotations

import argparse

from forecasting.experiment import (
    format_adaptive_comparison,
    format_results_table,
    format_sliding_window_comparison,
    format_window_sweep_comparison,
    run_adaptive_comparison,
    run_experiment,
    run_sliding_window_comparison,
    run_window_sweep_comparison,
)


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


def _plot_adaptive(dataset: str, comparison, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alpha_t = comparison.alpha_t
    x = range(len(alpha_t))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, alpha_t, color="#d62728", linewidth=1.3, label="ACI's adapted alpha_t")
    ax.axhline(comparison.alpha, color="#222222", linewidth=1.0, linestyle="--", label=f"nominal alpha={comparison.alpha}")
    ax.set_title(
        f"{dataset}: ACI's miscoverage rate over the test window "
        f"(static coverage {comparison.static['empirical_coverage'] * 100:.1f}% -> "
        f"adaptive {comparison.adaptive['empirical_coverage'] * 100:.1f}%, nominal "
        f"{int(round((1 - comparison.alpha) * 100))}%)"
    )
    ax.set_xlabel("test-set time step")
    ax.set_ylabel("alpha_t (lower = wider interval)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_sliding_window(dataset: str, comparison, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pool_size = comparison.pool_size
    x = range(len(pool_size))

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(x, pool_size, color="#2ca02c", linewidth=1.5, label="residual pool size")
    ax1.set_xlabel("test-set time step")
    ax1.set_ylabel("pool size (# residuals)", color="#2ca02c")
    ax1.tick_params(axis="y", labelcolor="#2ca02c")
    window_desc = "unbounded" if comparison.window is None else str(comparison.window)
    s, f, w = comparison.static, comparison.fixed_pool, comparison.sliding
    ax1.set_title(
        f"{dataset}: sliding-pool ACI's residual pool size over the test window (window={window_desc})\n"
        f"coverage -- static {s['empirical_coverage'] * 100:.1f}%, fixed-pool ACI {f['empirical_coverage'] * 100:.1f}%, "
        f"sliding-pool ACI {w['empirical_coverage'] * 100:.1f}% (nominal {int(round((1 - comparison.alpha) * 100))}%)",
        fontsize=10,
    )
    ax1.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_window_sweep(dataset: str, comparison, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    windows = [r.window for r in comparison.results]
    gaps = [r.sliding["coverage_gap"] * 100 for r in comparison.results]
    # Plot against an evenly-spaced index, not the raw window values, so an
    # "unbounded" entry (window=None) doesn't have to pretend to be a number
    # on the x-axis -- it's drawn as its own labeled tick instead.
    x = range(len(windows))
    labels = ["unbounded" if w is None else str(w) for w in windows]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, gaps, color="#2ca02c", linewidth=1.5, marker="o", label="sliding-pool ACI")
    ax.axhline(
        comparison.fixed_pool["coverage_gap"] * 100,
        color="#1f77b4",
        linewidth=1.2,
        linestyle="--",
        label="fixed-pool ACI (window-independent)",
    )
    ax.axhline(
        comparison.static["coverage_gap"] * 100,
        color="#888888",
        linewidth=1.0,
        linestyle=":",
        label="static split conformal",
    )
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("window (max residual-pool size)")
    ax.set_ylabel("coverage gap (pp, closer to 0 is better)")
    ax.set_title(f"{dataset}: sliding-pool ACI coverage gap by window size")
    ax.legend(loc="best", fontsize=8)
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

    adaptive = sub.add_parser(
        "adaptive",
        help="compare static split conformal vs. Adaptive Conformal Inference on the LSTM model",
    )
    adaptive.add_argument("--dataset", choices=["airline", "temperature", "synthetic"], default="airline")
    adaptive.add_argument("--alpha", type=float, default=0.1)
    adaptive.add_argument("--gamma", type=float, default=0.05, help="ACI step size")
    adaptive.add_argument("--plot", type=str, default=None, help="save a plot of alpha_t over the test window")
    adaptive.add_argument("--seed", type=int, default=0)

    sliding = sub.add_parser(
        "sliding-window",
        help="compare static split conformal, fixed-pool ACI, and sliding/growing-pool ACI on the LSTM model",
    )
    sliding.add_argument("--dataset", choices=["airline", "temperature", "synthetic"], default="airline")
    sliding.add_argument("--alpha", type=float, default=0.1)
    sliding.add_argument("--gamma", type=float, default=0.05, help="ACI step size")
    sliding.add_argument(
        "--window",
        type=int,
        default=None,
        help="max residual-pool size (default: unbounded growing buffer)",
    )
    sliding.add_argument("--plot", type=str, default=None, help="save a plot of the residual pool size over the test window")
    sliding.add_argument("--seed", type=int, default=0)

    sweep = sub.add_parser(
        "window-sweep",
        help="sweep the sliding-pool ACI's window size and compare coverage gap across values",
    )
    sweep.add_argument("--dataset", choices=["airline", "temperature", "synthetic"], default="airline")
    sweep.add_argument("--alpha", type=float, default=0.1)
    sweep.add_argument("--gamma", type=float, default=0.05, help="ACI step size")
    sweep.add_argument(
        "--windows",
        type=str,
        default="10,15,20,30,50,unbounded",
        help="comma-separated list of window sizes to try; include 'unbounded' for the growing buffer",
    )
    sweep.add_argument("--plot", type=str, default=None, help="save a plot of coverage gap vs. window size")
    sweep.add_argument("--seed", type=int, default=0)

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
    elif args.command == "adaptive":
        comparison = run_adaptive_comparison(args.dataset, alpha=args.alpha, gamma=args.gamma, seed=args.seed)
        print(format_adaptive_comparison(comparison))
        print()
        if args.plot:
            _plot_adaptive(args.dataset, comparison, args.plot)
            print(f"Saved plot to {args.plot}")
    elif args.command == "sliding-window":
        comparison = run_sliding_window_comparison(
            args.dataset, alpha=args.alpha, gamma=args.gamma, window=args.window, seed=args.seed
        )
        print(format_sliding_window_comparison(comparison))
        print()
        if args.plot:
            _plot_sliding_window(args.dataset, comparison, args.plot)
            print(f"Saved plot to {args.plot}")
    elif args.command == "window-sweep":
        windows = [None if w.strip().lower() == "unbounded" else int(w) for w in args.windows.split(",")]
        comparison = run_window_sweep_comparison(
            args.dataset, windows=windows, alpha=args.alpha, gamma=args.gamma, seed=args.seed
        )
        print(format_window_sweep_comparison(comparison))
        print()
        if args.plot:
            _plot_window_sweep(args.dataset, comparison, args.plot)
            print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
