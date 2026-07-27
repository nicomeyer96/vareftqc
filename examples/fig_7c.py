# This code is part of the VarEFTQC module.
#
# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Plot encoding and operation losses for Figure 7c.

This script loads the stored experiment logs for a selected Figure 7c run,
extracts encoding and logical-operation training metrics, and generates the
corresponding two-panel plot.
"""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt

from plot_helper import setup_figure_latex_layout


_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "fig_7c"
_PLOTS_DIR = Path(__file__).resolve().parent / "plots"
_STEPS_PER_EPOCH = 20


def _load_logs(path: Path):
    """Load pickled experiment logs from disk.

    Args:
        path (Path): Path to the pickled log file.

    Returns:
        Any: Loaded log object.
    """
    with path.open("rb") as file:
        return pickle.load(file)


def _flatten_and_pad_epoch_history(epoch_history, steps_per_epoch: int):
    """Flatten and pad per-epoch training histories.

    Short epochs are padded with their final value up to ``steps_per_epoch``.
    This keeps the plotted step axis aligned even when training converges early.

    Args:
        epoch_history (Iterable[Iterable[float]]): Nested per-epoch history.
        steps_per_epoch (int): Expected number of steps per epoch.

    Returns:
        list[float]: Flattened and padded history.
    """
    flattened_history = []
    for epoch_values in epoch_history:
        flattened_history.extend(epoch_values)
        if epoch_values and len(epoch_values) < steps_per_epoch:
            flattened_history.extend([epoch_values[-1]] * (steps_per_epoch - len(epoch_values)))
    return flattened_history


def plot(pad: float = 0.25, epochs: int = 1000, number: int = 0) -> None:
    """Create and store one Figure 7c plot.

    Args:
        pad (float): Padding passed to ``plt.tight_layout``.
        epochs (int): Maximum number of training steps to display.
        number (int): Experiment index used to locate the stored logs.

    Returns:
        None
    """
    figsize = setup_figure_latex_layout(
        aspect_ratio=1 / 1.62,
        width_ratio=1,
        columns="twocolumn",
    )
    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=figsize,
        sharex=True,
        dpi=120,
        facecolor="white",
    )

    logs_path = _RESULTS_DIR / "logs.pkl"
    logs = _load_logs(logs_path)

    train_avg_epochs, train_max_epochs, _ = logs.logger_encoding.get_train()
    validation_avg, validation_max, _ = logs.logger_encoding.get_validation()

    train_avg = _flatten_and_pad_epoch_history(train_avg_epochs[1:], _STEPS_PER_EPOCH)
    train_max = _flatten_and_pad_epoch_history(train_max_epochs[1:], _STEPS_PER_EPOCH)
    validation_steps = [_STEPS_PER_EPOCH * step for step in range(len(validation_avg))]
    validation_limit = epochs // _STEPS_PER_EPOCH + 1

    axes[0].plot(
        range(len(train_avg[: epochs + 1])),
        train_avg[: epochs + 1],
        color="tab:blue",
    )
    axes[0].plot(
        validation_steps[:validation_limit],
        validation_avg[:validation_limit],
        color="tab:blue",
        linestyle=":",
    )

    axes[0].plot(
        range(len(train_max[: epochs + 1])),
        train_max[: epochs + 1],
        color="tab:green",
    )
    axes[0].plot(
        validation_steps[:validation_limit],
        validation_max[:validation_limit],
        color="tab:green",
        linestyle=":",
    )

    axes[0].set_yticks([0.05, 0.10, 0.15])
    axes[0].set_ylim(bottom=0.045, top=0.19)
    axes[0].set_ylabel("D-Loss")

    _, train_t_max_epochs, _ = logs.logger_operation_transversal["T"].get_train()
    _, train_cx_max_epochs, _ = logs.logger_operation_weakly_transversal["CX"].get_train()

    train_t_max = _flatten_and_pad_epoch_history(train_t_max_epochs[1:], _STEPS_PER_EPOCH)
    train_cx_max = _flatten_and_pad_epoch_history(train_cx_max_epochs[1:], _STEPS_PER_EPOCH)

    axes[1].plot(
        range(len(train_t_max[: epochs + 1])),
        train_t_max[: epochs + 1],
        color="tab:red",
        label=r"$\overline{\mathcal{O}}_{\mathrm{block}}(T)$",
    )
    axes[1].plot(
        range(len(train_cx_max[: epochs + 1])),
        train_cx_max[: epochs + 1],
        color="tab:orange",
        label=r"$\overline{\mathcal{O}}_{\mathrm{block}}(CX)$",
    )

    axes[1].set_ylabel("O-Loss")
    axes[1].set_xlabel("Training Steps")
    axes[1].set_yscale("log")
    axes[1].set_yticks([1, 0.001, 0.000001])
    axes[1].set_ylim(bottom=2e-7, top=5)
    axes[1].hlines(y=1e-5, xmin=0, xmax=epochs, linestyle="--", color="black", linewidth=1)

    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _PLOTS_DIR / f"fig_7c.pdf"

    plt.tight_layout(pad=pad)
    figure.savefig(output_path)
    plt.close(figure)

    return None


if __name__ == "__main__":
    plot(pad=0.35, epochs=1000)
