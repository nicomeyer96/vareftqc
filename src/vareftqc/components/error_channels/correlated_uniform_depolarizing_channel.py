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

"""Correlated depolarizing noise channels for VarEFTQC.

This module defines :class:`CorrelatedUniformDepolarizingChannel`, a
two-qubit noise channel that applies uniform depolarizing noise over all
non-identity Pauli pairs with equal probability.
"""

import numpy as np
import pennylane as qml


pauli_ops = {
    'I': np.array([[1, 0], [0, 1]], dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex)
}


class CorrelatedUniformDepolarizingChannel(qml.operation.Operation):
    """Uniform two-qubit depolarizing channel with correlated errors.

    This channel acts on two qubits and implements a uniform depolarizing
    noise model over all non-identity two-qubit Pauli operators, i.e.
    :math:`p/15` weight on each of the 15 non-identity operators and
    :math:`1 - p` weight on the identity.

    Note:
        Using this operation requires a device capable of simulating
        noisy channels (e.g. ``"default.mixed"``).
    """

    # Defined for two wires
    num_wires = 2

    def __init__(self,
                 wires: list,
                 p: float
                 ):
        """Initialize the correlated uniform depolarizing channel.

        Args:
            wires (list): List of two wire labels on which the channel acts.
            p (float): Total depolarizing strength, distributed uniformly
                over all 15 non-identity two-qubit Pauli operators.

        Raises:
            ValueError: If the number of wires is not exactly 2.
        """

        if not 2 == len(wires):
            raise ValueError(f'The CorrelatedUniformDepolarizingChannel is defined for 2 qubits, but {len(wires)} '
                             f'were provided.')

        # determine the Kraus matrices for all wires
        kraus_matrices = self.get_kraus_matrices(p)

        # define non-trainable hyperparameters
        self._hyperparameters = {
            'kraus': kraus_matrices
        }

        # initialize the parent class
        super().__init__(wires=qml.wires.Wires(wires), id=f'p={p}')

    @property
    def num_params(self):
        """int: Number of trainable parameters (0)."""
        # no trainable parameters
        return 0

    @staticmethod
    def compute_decomposition(wires, kraus):  # pylint: disable=arguments-differ  # noqa
        """Decompose the channel into a QubitChannel with given Kraus operators.

        Args:
            wires (qml.wires.Wires): Wires on which the channel acts.
            kraus (list[np.ndarray]): List of Kraus matrices implementing
                the two-qubit depolarizing channel.

        Returns:
            list[qml.operation.Operator]: A single :class:`qml.QubitChannel`
            operation with the specified Kraus operators.
        """

        return [qml.QubitChannel(kraus, wires=wires)]

    @staticmethod
    def get_kraus_matrices(p: float):
        """Compute Kraus operators for correlated two-qubit depolarizing noise.

        The resulting channel is:

        * with probability ``1 - p``: apply the identity,
        * with total probability ``p``: apply one of the 15 non-identity
          two-qubit Pauli operators (``II`` excluded), each with probability
          ``p / 15``.

        Args:
            p (float): Depolarizing strength.

        Returns:
            list[np.ndarray]: List of Kraus matrices implementing the
            correlated two-qubit depolarizing channel.
        """

        # noise-free part
        kraus_matrices = [np.sqrt(1 - p + qml.math.eps) * qml.math.convert_like(
            qml.math.kron(pauli_ops['I'], pauli_ops['I']), p)]
        # all noisy components, with equal probability (p/15)
        pauli_labels = ['I', 'X', 'Y', 'Z']
        non_identity_pairs = [
            (a, b) for a in pauli_labels for b in pauli_labels
            if not (a == 'I' and b == 'I')
        ]
        for a, b in non_identity_pairs:
            operator = qml.math.convert_like(qml.math.kron(pauli_ops[a], pauli_ops[b]), p)
            ki = np.sqrt(p / 15 + qml.math.eps) * operator
            kraus_matrices.append(ki)
        return kraus_matrices


if __name__ == '__main__':
    @qml.qnode(qml.device("default.mixed", wires=[0, 1]), interface='torch')
    def circuit():
        CorrelatedUniformDepolarizingChannel(wires=[0, 1], p=0.1)
        return qml.state()
    _drawer = qml.draw(circuit, level=1, show_matrices=False)
    print(_drawer())
    _state = circuit()
    print(_state)
