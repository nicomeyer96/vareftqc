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

"""Plot training and validation distinguishability losses for Figure 7a.

This script loads the stored experiment logs for Figure 7a, extracts the
encoding training and validation losses, and generates the corresponding plot.
"""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt

from plot_helper import setup_figure_latex_layout


_RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "fig_7a" / "logs.pkl"
_PLOTS_DIR = Path(__file__).resolve().parent / "plots"
_OUTPUT_FILENAME = "fig_7a.pdf"
_MAX_TRAINING_POINTS = 101
_MAX_VALIDATION_POINTS = 6
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


def _flatten_epoch_history(epoch_history):
    """Flatten a per-epoch history into a single list.

    Args:
        epoch_history (Iterable[Iterable[float]]): Nested training history.

    Returns:
        list[float]: Flattened training history.
    """
    flattened_history = []
    for epoch_values in epoch_history:
        flattened_history.extend(epoch_values)
    return flattened_history


def plot(pad: float = 0.25) -> None:
    """Create and store the Figure 7a plot.

    Args:
        pad (float): Padding passed to ``plt.tight_layout``.

    Returns:
        None
    """
    figsize = setup_figure_latex_layout(
        aspect_ratio=1 / 1.62,
        width_ratio=1,
        columns="twocolumn",
    )
    figure, axis = plt.subplots(figsize=figsize, dpi=120, facecolor="white")

    logs = _load_logs(_RESULTS_PATH)

    train_avg_epochs, train_max_epochs, _ = logs.logger_encoding.get_train()
    validation_avg, validation_max, _ = logs.logger_encoding.get_validation()

    train_avg = _flatten_epoch_history(train_avg_epochs[1:])
    train_max = _flatten_epoch_history(train_max_epochs[1:])
    validation_steps = [_STEPS_PER_EPOCH * step for step in range(len(validation_avg))]

    axis.plot(
        range(len(train_avg[:_MAX_TRAINING_POINTS])),
        train_avg[:_MAX_TRAINING_POINTS],
        label=r"Average-Case $\mathcal{D}_{\mathcal{S}}$",
        color="tab:blue",
    )
    axis.plot(
        validation_steps[:_MAX_VALIDATION_POINTS],
        validation_avg[:_MAX_VALIDATION_POINTS],
        label=r"Average-Case $\mathcal{D}$",
        color="tab:blue",
        linestyle=":",
        marker="o",
    )

    axis.plot(
        range(len(train_max[:_MAX_TRAINING_POINTS])),
        train_max[:_MAX_TRAINING_POINTS],
        label=r"Worst-Case $\overline{\mathcal{D}}_{\mathcal{S}}$",
        color="tab:green",
    )
    axis.plot(
        validation_steps[:_MAX_VALIDATION_POINTS],
        validation_max[:_MAX_VALIDATION_POINTS],
        label=r"Worst-Case $\overline{\mathcal{D}}$",
        color="tab:green",
        linestyle=":",
        marker="o",
    )

    axis.set_yticks([0.05, 0.10, 0.15])
    axis.set_ylim(bottom=0.045, top=0.19)
    axis.set_xlabel("Training Steps")
    axis.set_ylabel("Distinguishability Loss")
    axis.legend()

    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _PLOTS_DIR / _OUTPUT_FILENAME

    plt.tight_layout(pad=pad)
    figure.savefig(output_path)
    plt.close(figure)

    return None


if __name__ == "__main__":
    plot(pad=0.35)
