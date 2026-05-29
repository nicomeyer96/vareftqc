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

"""Loss functions for encoding, recovery, and logical operations in VarEFTQC.

This module defines:

* :func:`distinguishability_loss` for encoding under noise,
* :func:`fidelity_loss` for recovery, and
* :func:`operation_loss` for logical operations,

as well as ground-truth structures used to evaluate operation losses.
"""

import pennylane as qml
import torch


def distinguishability_loss(prediction, groundtruth, display: str = None):
    """Distinguishability loss for training encodings.

        Given two batches of density matrices, this loss measures how much
        distinguishability is lost under noise by comparing pairwise trace
        distances. The function then returns the
        average and maximum loss over all unordered index pairs.

        Args:
            prediction (torch.Tensor): Batch of predicted density matrices of
                shape ``(B, d, d)``.
            groundtruth (torch.Tensor): Batch of ground-truth density matrices of
                shape ``(B, d, d)``.
            display (str | None): Optional prefix used when printing the loss
                values. If ``None``, no printing occurs.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Average and maximum lost
            distinguishability (scalars).

        Raises:
            ValueError: If inputs are not batches of density matrices or if the
                batch sizes do not match.
            RuntimeError: If the number of pairwise combinations is inconsistent.
        """

    # check input
    if 3 != len(prediction.shape) or 3 != len(groundtruth.shape):
        raise ValueError(f'Prediction and groundtruth have to be provided as a batch of density matrices.')
    number_density_matrices = prediction.shape[0]
    if not number_density_matrices == groundtruth.shape[0]:
        raise ValueError(f'The number of groundtruth elements ({number_density_matrices}) and '
                         f'predictions ({groundtruth.shape[0]}) does not match.')

    # computes the distinguishability loss without explicitly constructing all combinations of density matrices
    pairwise_lost_distinguishability = []
    # iterate over upper triangle matrix (excluding diagonals)
    for row in range(0, number_density_matrices):
        for col in range(1 + row, number_density_matrices):
            # determine trace for groundtruth first
            trace_groundtruth = qml.math.trace_distance(groundtruth[row], groundtruth[col])

            # determine trace for prediction (noisy density matrices)
            trace_prediction = qml.math.trace_distance(prediction[row], prediction[col])

            # subtract from the groundtruth traces to get the `loss in distinguishability`
            # (the values are in principle always positive, but the `relu` cleans some numerical inaccuracies)
            pairwise_lost_distinguishability.append(torch.nn.functional.relu(trace_groundtruth - trace_prediction))  # noqa

    # stack results and perform sanity check
    lost_distinguishability = torch.stack(pairwise_lost_distinguishability, dim=0)
    if lost_distinguishability.shape[0] != (number_density_matrices ** 2 - number_density_matrices) // 2:
        raise RuntimeError(f'Expected {(number_density_matrices ** 2 - number_density_matrices) // 2} unique pairs,'
                           f' but got {lost_distinguishability.shape[0]}.')

    # extract average and maximum value (take into account symmetry and diagonal elements, which are always 0)
    max_lost_distinguishability = torch.max(lost_distinguishability)  # noqa
    avg_lost_distinguishability = (2 * torch.sum(lost_distinguishability)) / (number_density_matrices ** 2)

    if display is not None:
        print(f'{display} >>> AVG: {avg_lost_distinguishability.detach():.7f} | MAX: {max_lost_distinguishability.detach():.7f} [d-loss]')  # noqa
    return avg_lost_distinguishability, max_lost_distinguishability


def fidelity_loss(prediction, groundtruth, display: str = None):
    """Fidelity-based loss for training recovery.

    Given two batches of density matrices, this loss computes the fidelity
    between corresponding states and returns the **lost** fidelity.
    Negative values due to numerical noise are clipped to zero via ReLU.

    Args:
        prediction (torch.Tensor): Batch of predicted density matrices of
            shape ``(B, d, d)``.
        groundtruth (torch.Tensor): Batch of ground-truth density matrices of
            shape ``(B, d, d)``.
        display (str | None): Optional prefix used when printing the loss
            values. If ``None``, no printing occurs.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Average and maximum lost fidelity
        (scalars).

    Raises:
        ValueError: If inputs are not batches of density matrices or if the
            batch sizes do not match.
    """

    # check input
    if 3 != len(prediction.shape) or 3 != len(groundtruth.shape):
        raise ValueError(f'Prediction and groundtruth have to be provided as a batch of density matrices.')
    number_density_matrices = prediction.shape[0]
    if not number_density_matrices == groundtruth.shape[0]:
        raise ValueError(f'The number of groundtruth elements ({number_density_matrices}) and '
                         f'predictions ({groundtruth.shape[0]}) does not match.')

    # determine fidelity between state pairs (relu removes numerical inaccuracies)
    fidelities = torch.nn.functional.relu(qml.math.fidelity(prediction, groundtruth))  # noqa

    # return (and optionally show) the average lost value, subtract from optimum 1 to indicate `lost` value
    max_lost_fidelity = 1 - torch.min(fidelities)
    avg_lost_fidelity = 1 - torch.mean(fidelities)
    if display is not None:
        print(f'{display} >>> AVG: {avg_lost_fidelity.detach():.7f} | MAX: {max_lost_fidelity.detach():.7f} [f-loss]')  # noqa
    return avg_lost_fidelity, max_lost_fidelity


GROUNDTRUTH_STRUCTURE_1Q = [
    ['d', 'b', 'f', 'f', 'f', 'f'],  # |0>
    ['b', 'd', 'f', 'f', 'f', 'f'],  # |1>
    ['f', 'f', 'd', 'b', 'f', 'f'],  # |+>
    ['f', 'f', 'b', 'd', 'f', 'f'],  # |->
    ['f', 'f', 'f', 'f', 'd', 'b'],  # |+i>
    ['f', 'f', 'f', 'f', 'b', 'd']   # |-i>
]
GROUNDTRUTH_1Q = {'d': 1.0, 'b': 0.0, 'f': 1/2}

GROUNDTRUTH_STRUCTURE_2Q = [
    ['d', 'b', 'b', 'b', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'x', 'x', 'y', 'y'],  # |00>
    ['b', 'd', 'b', 'b', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'y', 'y', 'x', 'x'],  # |01>
    ['b', 'b', 'd', 'b', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'y', 'y', 'x', 'x'],  # |10>
    ['b', 'b', 'b', 'd', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'x', 'x', 'y', 'y'],  # |11>
    ['f', 'f', 'f', 'f', 'd', 'b', 'b', 'b', 'f', 'f', 'f', 'f', 'x', 'y', 'x', 'y'],  # |++>
    ['f', 'f', 'f', 'f', 'b', 'd', 'b', 'b', 'f', 'f', 'f', 'f', 'y', 'x', 'y', 'x'],  # |+->
    ['f', 'f', 'f', 'f', 'b', 'b', 'd', 'b', 'f', 'f', 'f', 'f', 'y', 'x', 'y', 'x'],  # |-+>
    ['f', 'f', 'f', 'f', 'b', 'b', 'b', 'd', 'f', 'f', 'f', 'f', 'x', 'y', 'x', 'y'],  # |-->
    ['f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'd', 'b', 'b', 'b', 'y', 'x', 'x', 'y'],  # |+i+i>
    ['f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'b', 'd', 'b', 'b', 'x', 'y', 'y', 'x'],  # |+i-i>
    ['f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'b', 'b', 'd', 'b', 'x', 'y', 'y', 'x'],  # |+i-i>
    ['f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'b', 'b', 'b', 'd', 'y', 'x', 'x', 'y'],  # |-i-i>
    ['x', 'y', 'y', 'x', 'x', 'y', 'y', 'x', 'y', 'x', 'x', 'y', 'd', 'b', 'b', 'b'],  # |00>+|11>
    ['x', 'y', 'y', 'x', 'y', 'x', 'x', 'y', 'x', 'y', 'y', 'x', 'b', 'd', 'b', 'b'],  # |00>-|11>
    ['y', 'x', 'x', 'y', 'x', 'y', 'y', 'x', 'x', 'y', 'y', 'x', 'b', 'b', 'd', 'b'],  # |01>+|10>
    ['y', 'x', 'x', 'y', 'y', 'x', 'x', 'y', 'y', 'x', 'x', 'y', 'b', 'b', 'b', 'd']   # |01>-|10>
]
GROUNDTRUTH_2Q = {'d': 1.0, 'b': 0.0, 'f': 1/4, 'x': 1/2, 'y': 0.0}

GROUNDTRUTH = {1: (GROUNDTRUTH_STRUCTURE_1Q, GROUNDTRUTH_1Q), 2: (GROUNDTRUTH_STRUCTURE_2Q, GROUNDTRUTH_2Q)}


def operation_loss(prediction, target, order_operation: int, method: str = 'diag', display: str = None):
    """Fidelity-based loss for training logical operations.

    This loss compares predicted logical operations to target operations
    using a structured set of input states (typically drawn from a
    two-design) and analytic ground-truth fidelities encoded in
    :data:`GROUNDTRUTH`.

    Depending on ``method``, different subsets of state pairs are used:

    * ``"diag"``: only diagonal pairs (same input state),
    * ``"block"``: diagonal pairs plus selected orthogonal pairs (labels
      ``'b'``),
    * ``"block_ext"`` (order 2 only): diagonal pairs plus selected
      orthogonal and entangled pairs (labels ``'b'``, ``'x'``, ``'y'``),
    * ``"full"``: diagonal pairs plus all other labeled pairs.

    Args:
        prediction (torch.Tensor): Batch of predicted states, either as
            statevectors of shape ``(B, d)`` or density matrices of shape
            ``(B, d, d)``.
        target (torch.Tensor): Batch of target states, with the same shape as
            ``prediction``.
        order_operation (int): Order of the logical operation (1 or 2). This
            determines which ground-truth structure is used.
        method (str): Loss variant to use. One of ``"diag"``, ``"block"``,
            ``"block_ext"``, or ``"full"``. ``"block_ext"`` is only
            supported for ``order_operation == 2``.
        display (str | None): Optional prefix used when printing the loss
            values. If ``None``, no printing occurs.

    Returns:
        tuple:
            avg_lost_fidelity (torch.Tensor): Average absolute loss over all
                selected pairs.
            max_lost_fidelity (torch.Tensor): Maximum absolute loss over all
                selected pairs.
            loss_log (dict): Dictionary summarizing the loss, with keys:

                * ``"method"`` (str): Selected loss method.
                * ``"avg"`` (float): Average loss (selected method).
                * ``"max"`` (float): Maximum loss (selected method).
                * ``"avg_diag"`` (float | None): Average diagonal loss
                  (always computed), or ``None`` for ``"diag"``.
                * ``"max_diag"`` (float | None): Maximum diagonal loss, or
                  ``None`` for ``"diag"``.

    Raises:
        ValueError: If the number of states does not match, if the method is
            unsupported, or if the chosen method requires two-design states
            and the batch size or order is incompatible.
    """

    number_states = prediction.shape[0]
    if not number_states == target.shape[0]:
        raise ValueError(f'The number of trainable ({number_states}) and target states {target.shape[0]} '
                         f'does not match!')

    # Evaluating anything but with the `diag` method is only supported for states from a two-design (see paper)
    if 'block_ext' == method and 2 != order_operation:
        raise ValueError(f'The method {method} is only supported for operation order 2.')
    if method in ['block', 'block_ext', 'full']:
        if (1 == order_operation and 6 != number_states) or (2 == order_operation and 16 != number_states):  # noqa
            raise ValueError(f'The loss method {method} requires states from a two-design.')
        if order_operation > 2:
            raise ValueError(f'The loss method {method} allows an operation order of at most 2.')
    else:
        if 'diag' != method:
            raise ValueError(f'The loss method {method} is not supported.')

    def compute_fidelity(phi, psi):
        """
        Helper function, accounts for possibility of statevector and density matrix representation.
        """
        return qml.math.fidelity(phi, psi) if 2 == len(phi.shape) else qml.math.fidelity_statevector(phi, psi)

    # extract labels and groundtruth values
    groundtruth_structure, groundtruth = GROUNDTRUTH[order_operation]

    fidelities_diagonal = []
    # Compute fidelities for the diagonal elements first (i.e. combinations marked with `d`)
    for index in range(number_states):
        if 'd' == groundtruth_structure[index][index]:
            fidelity = compute_fidelity(prediction[index], target[index])
            fidelities_diagonal.append(1 - fidelity)  # F(phi,phi) - F(phi,phi^) = 1 - F(phi,phi^)

    fidelities_offdiagonal = []
    # Compute fidelities of off-diagonal elements
    for row in range(number_states):
        for col in range(number_states):
            if 'block' == method:  # elements marked with `b`  (pairwise orthogonal)
                if 'b' == groundtruth_structure[row][col]:
                    fidelity = compute_fidelity(prediction[row], target[col])
                    fidelities_offdiagonal.append(groundtruth[groundtruth_structure[row][col]] - fidelity)
            if 'block_ext' == method:  # elements marked with `b` and elements marked with `x` & `y` (entangled states)
                if groundtruth_structure[row][col] in ['b', 'x', 'y']:
                    fidelity = compute_fidelity(prediction[row], target[col])
                    fidelities_offdiagonal.append(groundtruth[groundtruth_structure[row][col]] - fidelity)
            if 'full' == method:  # all state pairs (apart from `d` which are already appended to diagonal buffer)
                if groundtruth_structure[row][col] in ['b', 'x', 'y', 'f']:
                    fidelity = compute_fidelity(prediction[row], target[col])
                    fidelities_offdiagonal.append(groundtruth[groundtruth_structure[row][col]] - fidelity)

    # stack results
    fidelities = torch.stack(fidelities_diagonal + fidelities_offdiagonal, dim=0)
    fidelities_diagonal = torch.stack(fidelities_diagonal, dim=0)

    # always compute `diag` loss for logging
    with torch.no_grad():
        avg_lost_fidelity_diagonal = torch.mean(torch.abs(fidelities_diagonal))  # noqa
        max_lost_fidelity_diagonal = torch.max(torch.abs(fidelities_diagonal))  # noqa

    # compute loss with selected method
    avg_lost_fidelity = torch.mean(torch.abs(fidelities))  # noqa
    max_lost_fidelity = torch.max(torch.abs(fidelities))  # noqa

    # show loss value
    if display is not None:
        print(f'{display} >>> AVG: {avg_lost_fidelity.detach():.7f} | MAX={max_lost_fidelity.detach():.7f} '
              f'[`{method}` o-loss]')

    # log loss values with `diag` and selected method
    if 'diag' == method:
        loss_log = {
            'method': method,
            'avg': avg_lost_fidelity.detach().numpy(), 'max': max_lost_fidelity.detach().numpy(),
            'avg_diag': None, 'max_diag': None,
        }
    else:
        loss_log = {
            'method': method,
            'avg': avg_lost_fidelity.detach().numpy(), 'max': max_lost_fidelity.detach().numpy(),
            'avg_diag': avg_lost_fidelity_diagonal.detach().numpy(),
            'max_diag': max_lost_fidelity_diagonal.detach().numpy(),
        }

    return avg_lost_fidelity, max_lost_fidelity, loss_log
