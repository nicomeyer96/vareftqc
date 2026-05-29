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

"""Asymmetric depolarizing noise channel for VarEFTQC.

This module defines :class:`AsymmetricDepolarizingChannel`, a convenience
wrapper that applies a single-qubit depolarizing channel with an asymmetric
distribution over Pauli X, Y, and Z errors, potentially with different
strengths on different qubits.
"""

import numpy as np
import pennylane as qml
from scipy import optimize


class AsymmetricDepolarizingChannel(qml.operation.Operation):
    """Asymmetric depolarizing channel on multiple wires (independent).

    For each wire, this channel implements a single-qubit depolarizing
    noise model with total strength ``p`` and asymmetric weights over
    Pauli X, Y, and Z errors. The asymmetry is controlled by a parameter
    ``c`` that enters a non-linear equation to determine the individual
    probabilities ``p_x``, ``p_y``, and ``p_z``.

    The channel is applied independently to each wire, but each wire may
    have its own depolarizing strength ``p``.
    """

    # Can be defined on an arbitrary number of wires
    num_wires = None

    def __init__(self,
                 wires: list,
                 p: float | list,
                 c: float = 1.0
                 ):
        """Initialize the asymmetric depolarizing channel.

        Args:
            wires (list): List of wire labels on which to apply the channel.
            p (float | list[float]): Depolarizing strength( s ). If a single
                float is provided, the same strength is used for all wires.
                If a list is provided, it must match the length of
                ``wires`` and specifies per-wire strengths.
            c (float): Asymmetry parameter controlling the distribution of
                ``p`` over X, Y, and Z errors via a non-linear relation.

        Raises:
            ValueError: If a list of strengths is provided whose length does
                not match the number of wires.
        """

        # check inputs
        if isinstance(p, float):
            ps = [p] * len(wires)  # same error for all wires
        else:
            if len(p) != len(wires):
                raise ValueError(f'Inconsistent number of wires ({len(wires)}) and number of p`s ({len(p)}).')
            ps = p  # individual (potentially different) probability for each wire

        # determine the Kraus matrices for all wires
        kraus_matrices = []
        for p_ in ps:
            kraus_matrices.append(self.get_kraus_matrices(p_, c))

        # define non-trainable hyperparameters
        self._hyperparameters = {
            'kraus': kraus_matrices
        }

        # initialize the parent class
        super().__init__(wires=qml.wires.Wires(wires), id=f'p={p},c={c}')

    @property
    def num_params(self):
        """int: Number of trainable parameters (0)."""
        # no trainable parameters
        return 0

    @staticmethod
    def compute_decomposition(wires, kraus):  # pylint: disable=arguments-differ  # noqa
        """Decompose the channel into per-wire QubitChannel operations.

        Args:
            wires (qml.wires.Wires): Wires on which the channel acts.
            kraus (list[list[np.ndarray]]): List of Kraus-operator lists,
                one list per wire.

        Returns:
            list[qml.operation.Operator]: List of :class:`qml.QubitChannel`
            operations, one per wire.
        """

        op_list = []
        for wire, k in zip(wires, kraus):
            op_list.append(qml.QubitChannel(k, wires=wire))
        return op_list

    @staticmethod
    def construct_fn(p: float, c: float):
        """Construct the scalar equation used to determine ``p_x``.

        The equation

        .. math::

            f(x) = x^c + 2x - p = 0

        is solved to determine the X-error probability ``p_x``. The Y and
        Z probabilities are then taken as ``p_y = p_x`` and
        ``p_z = p - 2 p_x``.

        Args:
            p (float): Total depolarizing strength.
            c (float): Asymmetry parameter.

        Returns:
            callable: A scalar function ``f(x)`` whose root in ``[0, p]``
            yields ``p_x``.
        """

        def fn(x):
            return x ** c + 2 * x - p
        return fn

    def compute_p_xyz(self, p: float, c: float) -> (float, float, float):
        """Compute Pauli X, Y, Z error probabilities from ``p`` and ``c``.

        This method solves the equation constructed by :meth:`construct_fn`
        to obtain ``p_x`` and sets ``p_y = p_x`` and
        ``p_z = p - 2 p_x``.

        Args:
            p (float): Total depolarizing strength.
            c (float): Asymmetry parameter.

        Returns:
            tuple[float, float, float]: ``(p_x, p_y, p_z)``.
        """

        fn = self.construct_fn(p, c)
        px = optimize.brentq(fn, 0, p)
        return px, px, p - 2 * px  # noqa

    def get_kraus_matrices(self, p: float, c: float):
        """Compute Kraus operators for an asymmetric single-qubit depolarizing channel.

        The channel is defined such that:

        * with probability ``1 - p``: the identity is applied,
        * with probabilities ``p_x, p_y, p_z``: Pauli X, Y, Z are applied,
          respectively, where these probabilities satisfy:

          .. math::

              p_x = p_y,\\quad p_z = p - 2 p_x

          and are determined from ``p`` and ``c`` via :meth:`compute_p_xyz`.

        Args:
            p (float): Total depolarizing strength.
            c (float): Asymmetry parameter.

        Returns:
            list[np.ndarray]: List of four Kraus matrices ``[K0, K1, K2, K3]``
            implementing the asymmetric depolarizing channel.
        """

        px, py, pz = self.compute_p_xyz(p, c)
        # print(f'px:{px/p:.3f}, py:{py/p:.3f}, pz:{pz/p:.3f}')
        k0 = np.sqrt(1 - p + qml.math.eps) * qml.math.convert_like(np.eye(2, dtype=complex), p)
        k1 = np.sqrt(px + qml.math.eps) * qml.math.convert_like(np.array([[0, 1], [1, 0]], dtype=complex), p)
        k2 = np.sqrt(py + qml.math.eps) * qml.math.convert_like(
            np.array([[0, -1j], [1j, 0]], dtype=complex), p
        )
        k3 = np.sqrt(pz + qml.math.eps) * qml.math.convert_like(
            np.array([[1, 0], [0, -1]], dtype=complex), p
        )
        return [k0, k1, k2, k3]


if __name__ == '__main__':
    @qml.qnode(qml.device("default.mixed", wires=[0, 1]), interface='torch')
    def circuit():
        AsymmetricDepolarizingChannel(wires=[0, 1], p=[0.1, 0.2], c=0.5)
        return qml.state()
    _drawer = qml.draw(circuit, level=1, show_matrices=False)
    print(_drawer())
    _state = circuit()
    print(_state)
