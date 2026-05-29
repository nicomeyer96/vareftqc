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

"""State-preparation utilities for spherical two-designs in VarEFTQC.

This module defines :class:`SphericalTwoDesign`, a PennyLane operation that
prepares single- and two-qubit states forming a spherical 2-design. It
supports both single states and batched preparation via basis labels.
"""

import pennylane as qml
import numpy as np
import torch


class SphericalTwoDesign(qml.operation.Operation):
    """State-preparation operation for 1- and 2-qubit spherical two-designs.

    For 1 qubit, the supported basis states are:

    * ``"|0>"``, ``"|1>"`` (Z-basis),
    * ``"|+>"``, ``"|->"`` (X-basis),
    * ``"|+i>"``, ``"|-i>"`` (Y-basis),

    encoded as labels ``"0"``, ``"1"``, ``"+"``, ``"-"``, ``"+i"``,
    ``"-i"``.

    For 2 qubits, the supported basis states include products of these
    single-qubit states and Bell-like states forming a complete 2-design,
    encoded by strings such as ``"00"``, ``"++"``, ``"+i-i"``, ``"00+11"``,
    etc.

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

    BASIS_1QUBIT = ['0', '1',  # of z-basis states
                    '+', '-',  # of x-basis states
                    '+i', '-i']  # of y-basis states
    # parameter producing the respective basis states when using an U3 gate
    ID = [0.0, 0.0, 0.0]  # realized identity, i.e. `empty` U3
    PARAMETERS_1QUBIT = {
        '0': ID, '1': [np.pi, 0.0, np.pi],
        '+': [np.pi / 2, 0.0, np.pi], '-': [np.pi / 2, -np.pi, -np.pi],
        '+i': [np.pi / 2, np.pi / 2, -np.pi], '-i': [np.pi / 2, -np.pi / 2, -np.pi]
    }

    BASIS_2QUBITS = ['00', '01', '10', '11',  # products of z-basis states
                     '++', '+-', '-+', '--',  # products of x-basis states
                     '+i+i', '+i-i', '-i+i', '-i-i',  # products of y-basis states
                     '00+11', '00-11', '01+10', '01-10']  # bell states for a complete 2-design
    # parameter producing the respective basis states when using U3(x)U3 followed by a ControlledU3 gate
    CNOT = [np.pi, 0.0, np.pi]  # CNOT parameterized via (controlled) U3
    HZ = [np.pi / 2, np.pi, np.pi]  # realized Hadamard followed by Pauli-Z via U3
    PARAMETERS_2QUBITS = {
        '00': [*ID, *ID, *ID],
        '01': [*ID, *PARAMETERS_1QUBIT['1'], *ID],
        '10': [*PARAMETERS_1QUBIT['1'], *ID, *ID],
        '11': [*PARAMETERS_1QUBIT['1'], *PARAMETERS_1QUBIT['1'], *ID],
        '++': [*PARAMETERS_1QUBIT['+'], *PARAMETERS_1QUBIT['+'], *ID],
        '+-': [*PARAMETERS_1QUBIT['+'], *PARAMETERS_1QUBIT['-'], *ID],
        '-+': [*PARAMETERS_1QUBIT['-'], *PARAMETERS_1QUBIT['+'], *ID],
        '--': [*PARAMETERS_1QUBIT['-'], *PARAMETERS_1QUBIT['-'], *ID],
        '+i+i': [*PARAMETERS_1QUBIT['+i'], *PARAMETERS_1QUBIT['+i'], *ID],
        '+i-i': [*PARAMETERS_1QUBIT['+i'], *PARAMETERS_1QUBIT['-i'], *ID],
        '-i+i': [*PARAMETERS_1QUBIT['-i'], *PARAMETERS_1QUBIT['+i'], *ID],
        '-i-i': [*PARAMETERS_1QUBIT['-i'], *PARAMETERS_1QUBIT['-i'], *ID],
        '00+11': [*PARAMETERS_1QUBIT['+'], *ID, *CNOT],
        '00-11': [*HZ, *ID, *CNOT],
        '01+10': [*PARAMETERS_1QUBIT['+'], *PARAMETERS_1QUBIT['1'], *CNOT],
        '01-10': [*HZ, *PARAMETERS_1QUBIT['1'], *CNOT],
    }

    def __init__(self,
                 wires: list,
                 basis: list = None):
        """Initialize the spherical two-design state preparation.

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
            raise ValueError(f'The SphericalTwoDesign is only supported for up to 2 wires, '
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
            raise ValueError(f'The SphericalTwoDesign is only supported for up to 2 wires, '
                             f'but {len(wires)} were provided.')


if __name__ == '__main__':
    @qml.qnode(qml.device("default.mixed", wires=[0, 1]), interface='torch')
    # @qml.qnode(qml.device("default.qubit", wires=[0, 1]), interface='torch')
    def circuit(basis=None):
        SphericalTwoDesign(wires=[0], basis=basis)
        # SphericalTwoDesign(wires=[0, 1], basis=basis)
        return qml.state()
    _drawer = qml.draw(circuit, level=1, show_matrices=False)
    # _drawer = qml.draw(circuit, level=3, show_matrices=False)
    print(_drawer())
    _state = circuit()
    print(_state)
