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

"""State preparation for Haar-random states in VarEFTQC.

This module defines :class:`HaarRandomState`, a PennyLane operation that
prepares single- or multi-qubit states distributed according to the Haar
measure, with an optional seed for reproducibility and an optional batch
dimension.
"""

import pennylane as qml
import numpy as np
import torch
from scipy.stats import unitary_group


class HaarRandomState(qml.operation.Operation):
    """State-preparation operation for Haar-random pure states.

    This operation samples Haar-random unitaries of appropriate dimension
    and uses them to prepare random pure states on 1 or more qubits.
    A seed can be provided for reproducibility. A batch of states can be
    prepared by setting ``number > 1``.

    Attributes:
        num_wires (None): Operation can act on an arbitrary number of wires.
        grad_recipe (None): This operation is treated as non-differentiable.
    """

    # One or multiple wires
    num_wires = None

    # Non-differentiable
    grad_recipe = None

    def __init__(self,
                 wires: list,
                 number: int = 2,
                 seed: int = None
                 ):
        """Initialize Haar-random state preparation.

        Args:
            wires (list): List of wire labels on which to prepare the
                random states.
            number (int): Number of Haar-random states to prepare. Must be
                at least 2 (as the framework typically uses state pairs).
            seed (int | None): Random seed for the underlying NumPy RNG.
                If ``None``, each instantiation samples independent states.

        Raises:
            AssertionError: If ``number < 2``.
        """

        # check inputs
        assert number >= 2, 'The `number` argument has to be at least 2 (as working with state pairs).'

        # if seed is None:
        #     warnings.warn('No seed has been set, different states will be produced for every consecutive use.')

        # encapsulated random seed generator for reproducibility
        rng = np.random.default_rng(seed=seed)

        # sample Haar unitaries
        parameters = torch.tensor(unitary_group.rvs(2**len(wires), size=number, random_state=rng))

        # define non-trainable hyperparameters, none in this case
        self._hyperparameters = {}

        # initialize the parent class
        super().__init__(parameters, wires=qml.wires.Wires(wires), id=f'size={number},seed={seed}')  # noqa

    @property
    def num_params(self):
        """int: Number of parameter sets (1) for this operation.

        The single parameter is the unitary matrix (possibly batched) used
        in :class:`qml.QubitUnitary`.
        """

        # single parameter set, i.e. unitary matrix
        return 1

    @property
    def ndim_params(self):
        """tuple[int]: Dimensionality of the parameter tensor.

        Returns:
            tuple[int]: A single-element tuple ``(2,)``, indicating that the
            parameter is a 2D unitary matrix (with an optional leading batch
            dimension internally).
        """

        # two-dimensional unitary matrix (plus optional batch dimension)
        return (2,)

    @staticmethod
    def compute_decomposition(parameters, wires):  # pylint: disable=arguments-differ  # noqa
        """Decompose Haar-random state preparation into a QubitUnitary.

        Args:
            parameters (torch.Tensor | np.ndarray): Unitary matrix (or batch
                of matrices) sampled from the Haar measure, of shape
                ``(..., 2**n, 2**n)`` where ``n = len(wires)``.
            wires (Sequence[Any]): Wires on which to apply the unitary.

        Returns:
            list[qml.operation.Operator]: A single :class:`qml.QubitUnitary`
            operation that prepares the Haar-random state(s).
        """

        return [qml.QubitUnitary(parameters, wires=wires)]


if __name__ == '__main__':
    @qml.qnode(qml.device("default.mixed", wires=[0, 1]), interface='torch')
    # @qml.qnode(qml.device("default.qubit", wires=[0, 1]), interface='torch')
    def circuit(number=2):
        HaarRandomState(wires=[0], number=number, seed=1)
        # HaarRandomState(wires=[0, 1], number=number, seed=1)
        return qml.state()
    _drawer = qml.draw(circuit, level=1, show_matrices=False)
    # _drawer = qml.draw(circuit, level=3, show_matrices=False)
    print(_drawer())
    _state = circuit()
    print(_state)
