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

"""State-preparation utilities for projective two-designs in VarEFTQC.

This module defines :class:`ProjectiveTwoDesign`, a PennyLane operation that
prepares single- and two-qubit states forming a projective 2-design. It
supports both single states and batched preparation via basis labels.
"""

import pennylane as qml
import numpy as np
import torch


class ProjectiveTwoDesign(qml.operation.Operation):
    """State-preparation operation for 1- and 2-qubit complex projective two-designs.
    (See A. Klappenecker et al., Mutually unbiased bases are complex projective 2-designs. In Proceedings of the
    International Symposium on Information Theory (ISIT), pp. 1740–1744, IEEE (2005) for background)

    For 1 qubit, the supported basis states are:

    * ``"z0"``, ``"z1"`` (Z-basis),
    * ``"x0"``, ``"x1"`` (X-basis),
    * ``"y0"``, ``"y1"`` (Y-basis).

    For 2 qubits, the supported basis states are the 20 states from the 5 MUBs:

    * ``"zz0"``, ..., ``"zz3"``,
    * ``"xx0"``, ..., ``"xx3"``,
    * ``"yy0"``, ..., ``"yy3"``,
    * ``"czxy0"``, ..., ``"czxy3"``,
    * ``"czyx0"``, ..., ``"czyx3"``.

    We use the index convention

    * ``0 -> 00``
    * ``1 -> 01``
    * ``2 -> 10``
    * ``3 -> 11``

    so, for example:

    * ``"xx2" = |x1> ⊗ |x0>``
    * ``"czxy1" = CZ(|x0> ⊗ |y1>)``
    * ``"czyx3" = CZ(|y1> ⊗ |x1>)``

    The operation supports batched preparation by providing a list of basis
    labels; internally these are mapped to parameter vectors for U3 gates
    (and a controlled-U3) that realize the desired states.

    Attributes:
        num_wires (None): Operation can act on 1 or 2 wires.
        grad_recipe (None): This operation is treated as non-differentiable.
        BASIS_1QUBIT (list[str]): Supported 1-qubit basis labels.
        PARAMETERS_1QUBIT (dict[str, list[float]]): U3 parameters for each
            1-qubit basis label.
        BASIS_2QUBITS (list[str]): Supported 2-qubit basis labels.
        PARAMETERS_2QUBITS (dict[str, list[float]]): U3/CU3 parameters for
            each 2-qubit basis label.
    """

    # One or two wires
    num_wires = None

    # Non-differentiable
    grad_recipe = None

    # parameters to enter into U3 to realize ...
    ID = [0.0, 0.0, 0.0]  # identity
    CZ = [0.0, 0.0, np.pi]  # controlled-Z

    BASIS_1QUBIT = [
        'z0', 'z1',  # Z-basis
        'x0', 'x1',  # X-basis
        'y0', 'y1'   # Y-basis
    ]

    BASIS_2QUBITS = [
        'zz0', 'zz1', 'zz2', 'zz3',          # |z_a> ⊗ |z_b>
        'xx0', 'xx1', 'xx2', 'xx3',          # |x_a> ⊗ |x_b>
        'yy0', 'yy1', 'yy2', 'yy3',          # |y_a> ⊗ |y_b>
        'czxy0', 'czxy1', 'czxy2', 'czxy3',  # CZ(|x_a> ⊗ |y_b>)
        'czyx0', 'czyx1', 'czyx2', 'czyx3'   # CZ(|y_a> ⊗ |x_b>)
    ]

    # parameters producing the respective basis states when using a U3 gate
    PARAMETERS_1QUBIT = {
        'z0': ID,
        'z1': [np.pi, 0.0, np.pi],
        'x0': [np.pi / 2, 0.0, np.pi],
        'x1': [np.pi / 2, -np.pi, -np.pi],
        'y0': [np.pi / 2, np.pi / 2, -np.pi],
        'y1': [np.pi / 2, -np.pi / 2, -np.pi]
    }

    # parameters producing the respective 2-qubit basis states when using
    # U3 on both qubits followed by a controlled-U3 from wire 0 to wire 1
    PARAMETERS_2QUBITS = {
        # zz basis
        'zz0': [*ID, *ID, *ID],
        'zz1': [*ID, *PARAMETERS_1QUBIT['z1'], *ID],
        'zz2': [*PARAMETERS_1QUBIT['z1'], *ID, *ID],
        'zz3': [*PARAMETERS_1QUBIT['z1'], *PARAMETERS_1QUBIT['z1'], *ID],

        # xx basis
        'xx0': [*PARAMETERS_1QUBIT['x0'], *PARAMETERS_1QUBIT['x0'], *ID],
        'xx1': [*PARAMETERS_1QUBIT['x0'], *PARAMETERS_1QUBIT['x1'], *ID],
        'xx2': [*PARAMETERS_1QUBIT['x1'], *PARAMETERS_1QUBIT['x0'], *ID],
        'xx3': [*PARAMETERS_1QUBIT['x1'], *PARAMETERS_1QUBIT['x1'], *ID],

        # yy basis
        'yy0': [*PARAMETERS_1QUBIT['y0'], *PARAMETERS_1QUBIT['y0'], *ID],
        'yy1': [*PARAMETERS_1QUBIT['y0'], *PARAMETERS_1QUBIT['y1'], *ID],
        'yy2': [*PARAMETERS_1QUBIT['y1'], *PARAMETERS_1QUBIT['y0'], *ID],
        'yy3': [*PARAMETERS_1QUBIT['y1'], *PARAMETERS_1QUBIT['y1'], *ID],

        # 4th MUB: CZ(H (x) SH) |ab> = CZ(|x_a> (x) |y_b>)
        'czxy0': [*PARAMETERS_1QUBIT['x0'], *PARAMETERS_1QUBIT['y0'], *CZ],
        'czxy1': [*PARAMETERS_1QUBIT['x0'], *PARAMETERS_1QUBIT['y1'], *CZ],
        'czxy2': [*PARAMETERS_1QUBIT['x1'], *PARAMETERS_1QUBIT['y0'], *CZ],
        'czxy3': [*PARAMETERS_1QUBIT['x1'], *PARAMETERS_1QUBIT['y1'], *CZ],

        # 5th MUB: CZ(SH (x) H) |ab> = CZ(|y_a> (x) |x_b>)
        'czyx0': [*PARAMETERS_1QUBIT['y0'], *PARAMETERS_1QUBIT['x0'], *CZ],
        'czyx1': [*PARAMETERS_1QUBIT['y0'], *PARAMETERS_1QUBIT['x1'], *CZ],
        'czyx2': [*PARAMETERS_1QUBIT['y1'], *PARAMETERS_1QUBIT['x0'], *CZ],
        'czyx3': [*PARAMETERS_1QUBIT['y1'], *PARAMETERS_1QUBIT['x1'], *CZ]
    }

    def __init__(self,
                 wires: list,
                 basis: list = None):
        """Initialize the projective two-design state preparation.

        Args:
            wires (list): List of wire labels. Must be of length 1 or 2.
            basis (list[str] | None): List of basis labels indicating which
                states to prepare. If ``None``, all supported basis states
                for the given number of wires are used:

                * 1 qubit: all labels in :data:`BASIS_1QUBIT`,
                * 2 qubits: all labels in :data:`BASIS_2QUBITS`.

        Raises:
            ValueError: If more than 2 wires are provided.
            AssertionError: If any basis label is not in the corresponding
                supported basis set.
        """

        # check inputs and set up parameters
        if 1 == len(wires):
            if basis is None:
                basis = self.BASIS_1QUBIT
            else:
                for base in basis:
                    assert base in self.BASIS_1QUBIT, f'Basis elements have to be from {self.BASIS_1QUBIT}.'
            parameters = torch.tensor([self.PARAMETERS_1QUBIT.get(base) for base in basis])
        elif 2 == len(wires):
            if basis is None:
                basis = self.BASIS_2QUBITS
            else:
                for base in basis:
                    assert base in self.BASIS_2QUBITS, f'Basis elements have to be from {self.BASIS_2QUBITS}.'
            parameters = torch.tensor([self.PARAMETERS_2QUBITS.get(base) for base in basis])
        else:
            raise ValueError(f'The ProjectiveTwoDesign is only supported for up to 2 wires, '
                             f'but {len(wires)} were provided.')

        # define non-trainable hyperparameters
        self._hyperparameters = {}

        # initialize the parent class
        super().__init__(parameters, wires=qml.wires.Wires(wires), id=f'size={len(basis)}')  # noqa

    @property
    def num_params(self):
        """int: Number of parameter sets (1) for this operation."""
        # only one set of parameters for this operation
        return 1

    @property
    def ndim_params(self):
        """tuple[int]: Dimensionality of the parameter tensor.

        Returns:
            tuple[int]: A single-element tuple ``(1,)``, indicating that
            parameters are provided as a single tensor (with an optional
            batch dimension internally).
        """

        # internally produced, input for U3 gate (with optional batch dimension)
        return (1,)

    @staticmethod
    def compute_decomposition(parameters, wires):  # pylint: disable=arguments-differ  # noqa
        """Decompose the two-design preparation into native U3/CU3 gates.

        This method maps the stored parameter tensor to a sequence of
        PennyLane operations:

        * For 1 qubit: a single U3 gate on the given wire.
        * For 2 qubits:
          - U3 on the first wire,
          - U3 on the second wire,
          - a controlled-U3 from the first to the second wire.

        If a batch dimension is present (for certain devices), it is
        transposed away before use.

        Args:
            parameters (torch.Tensor): Parameter tensor generated in
                ``__init__``; may contain a batch dimension in the first
                axis.
            wires (Sequence[Any]): Wire labels on which to prepare the
                states (length 1 or 2).

        Returns:
            list[qml.operation.Operator]: List of PennyLane operations that
            implement the desired state preparation.

        Raises:
            ValueError: If more than 2 wires are provided.
        """

        if 2 == len(parameters.shape):  # revert batch dimension if it exists (required for `default.qubit` device)
            parameters = torch.transpose(parameters, 0, 1)

        if 1 == len(wires):
            return [qml.U3(*parameters, wires=wires)]
        elif 2 == len(wires):
            return [qml.U3(*parameters[0:3], wires=wires[0]),  # U3 on first qubit
                    qml.U3(*parameters[3:6], wires=wires[1]),  # U3 on second qubit
                    qml.ctrl(qml.U3(*parameters[6:9], wires=wires[1]), control=wires[0])]  # CU3 from first to second
        else:
            raise ValueError(f'The ProjectiveTwoDesign is only supported for up to 2 wires, '
                             f'but {len(wires)} were provided.')


if __name__ == '__main__':
    # @qml.qnode(qml.device("default.mixed", wires=[0, 1]), interface='torch')
    @qml.qnode(qml.device("default.qubit", wires=[0, 1]), interface='torch')
    def circuit(basis=None):
        # ProjectiveTwoDesign(wires=[0], basis=basis)
        ProjectiveTwoDesign(wires=[0, 1], basis=basis)
        return qml.state()
    _drawer = qml.draw(circuit, level=1, show_matrices=False)
    # _drawer = qml.draw(circuit, level=3, show_matrices=False)
    print(_drawer())
    _state = circuit()
    print(_state)
