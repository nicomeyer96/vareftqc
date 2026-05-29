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

"""Independent Pauli noise channel for VarEFTQC.

This module defines :class:`PauliChannel`, a convenience wrapper that
applies single-qubit Pauli noise with configurable X, Y, and Z weights to
one or more wires, each with its own overall strength.
"""

import numpy as np
import pennylane as qml


class PauliChannel(qml.operation.Operation):
    """Single-qubit Pauli channel on multiple wires (independent).

    For each wire, this channel applies a Pauli noise model with total
    strength ``p`` and relative weights for X, Y, and Z errors. The X and Z
    weights are given explicitly, and the Y weight is inferred as
    ``1 - px - pz``. Each wire can have its own values of ``p``, ``px``,
    and ``pz``.

    Note:
        Using this operation requires a device capable of simulating error
        channels (e.g. ``"default.mixed"``).
    """

    # Can be defined on an arbitrary number of wires
    num_wires = None

    def __init__(self,
                 wires: list,
                 p: float | list,
                 px: float | list = 1/3,
                 pz: float | list = 1/3
                 ):
        """Initialize the Pauli channel.

        Args:
            wires (list): List of wire labels on which to apply the channel.
            p (float | list[float]): Total Pauli error strength(s). If a
                single float is provided, the same strength is used for all
                wires. If a list is provided, it must match the length of
                ``wires`` and specifies per-wire strengths.
            px (float | list[float]): Relative weight(s) for Pauli X errors.
                If a list is provided, it must match the length of
                ``wires``. For each wire, the Y weight is computed as
                ``py = 1 - px - pz``.
            pz (float | list[float]): Relative weight(s) for Pauli Z errors.
                If a list is provided, it must match the length of
                ``wires``.

        Raises:
            ValueError: If the lengths of ``px``, ``pz``, or ``p`` lists do
                not match the number of wires, or if for any wire
                ``px + pz > 1.0``.
        """

        # check inputs
        if isinstance(px, float):
            pxs = [px] * len(wires)  # same Pauli-X weight for all wires
        else:
            if len(px) != len(wires):
                raise ValueError(f'Inconsistent number of wires ({len(wires)}) and number of px`s ({len(px)}).')
            pxs = px
        if isinstance(pz, float):
            pzs = [pz] * len(wires)  # same Pauli-Z weight for all wires
        else:
            if len(pz) != len(wires):
                raise ValueError(f'Inconsistent number of wires ({len(wires)}) and number of pz`s ({len(pz)}).')
            pzs = pz
        for px_, pz_ in zip(pxs, pzs):
            if px_ + pz_ > 1.0:
                raise ValueError('Sum of Pauli-X and Pauli-Z weights can be at most 1.')

        # extract values for different Pauli errors
        pys = [1 - px_ - pz_ for px_, pz_ in zip(pxs, pzs)]

        if isinstance(p, float):
            ps = [p] * len(wires)  # same error for all wires
        else:
            if len(p) != len(wires):
                raise ValueError(f'Inconsistent number of wires ({len(wires)}) and number of p`s ({len(p)}).')
            ps = p  # individual (potentially different) probability for each wire

        # determine the Kraus matrices for all wires
        kraus_matrices = []
        for p_, px_, py_, pz_ in zip(ps, pxs, pys, pzs):
            kraus_matrices.append(self.get_kraus_matrices(p_, px_, py_, pz_))

        # define non-trainable hyperparameters
        self._hyperparameters = {
            'kraus': kraus_matrices
        }

        # initialize the parent class
        super().__init__(wires=qml.wires.Wires(wires), id=f'p={p},X:{px}|Z:{pz}')

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
    def get_kraus_matrices(p: float, px: float, py: float, pz: float):
        """Compute Kraus operators for a single-qubit Pauli channel.

        The channel is defined such that:

        * with probability ``1 - p``: the identity is applied,
        * with probabilities ``p * px``, ``p * py``, and ``p * pz``:
          Pauli X, Y, and Z are applied, respectively.

        Args:
            p (float): Total Pauli error strength.
            px (float): Relative weight for Pauli X errors.
            py (float): Relative weight for Pauli Y errors.
            pz (float): Relative weight for Pauli Z errors.

        Returns:
            list[np.ndarray]: List of four Kraus matrices ``[K0, K1, K2, K3]``
            implementing the Pauli channel.
        """

        k0 = np.sqrt(1 - p + qml.math.eps) * qml.math.convert_like(np.eye(2, dtype=complex), p)
        k1 = np.sqrt(p * px + qml.math.eps) * qml.math.convert_like(
            np.array([[0, 1], [1, 0]], dtype=complex), p)
        k2 = np.sqrt(p * py + qml.math.eps) * qml.math.convert_like(
            np.array([[0, -1j], [1j, 0]], dtype=complex), p
        )
        k3 = np.sqrt(p * pz + qml.math.eps) * qml.math.convert_like(
            np.array([[1, 0], [0, -1]], dtype=complex), p
        )
        return [k0, k1, k2, k3]


if __name__ == '__main__':
    @qml.qnode(qml.device("default.mixed", wires=[0, 1]), interface='torch')
    def circuit():
        PauliChannel(wires=[0, 1], p=[0.1, 0.2], px=0.2, pz=0.2)
        return qml.state()
    _drawer = qml.draw(circuit, level=1, show_matrices=False)
    print(_drawer())
    _state = circuit()
    print(_state)
