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

"""Independent phase-flip noise channel for VarEFTQC.

This module defines :class:`PhaseFlipChannel`, a convenience wrapper that
applies phase-flip noise independently to one or more wires, each with its
own error probability.
"""

import pennylane as qml


class PhaseFlipChannel(qml.operation.Operation):
    """Phase-flip channel on multiple wires (independent).

    This operation applies :class:`qml.PhaseFlip` to each wire in the given
    list. All wires may share the same error probability or have individual
    values.

    Note:
        Using this operation requires a device capable of simulating error
        channels (e.g. ``"default.mixed"``).
    """

    # Can be defined on an arbitrary number of wires
    num_wires = None

    def __init__(self,
                 wires: list,
                 p: float | list
                 ):
        """Initialize the phase-flip channel.

        Args:
            wires (list): List of wire labels on which to apply the channel.
            p (float | list[float]): Phase-flip probability(ies). If a
                single float is provided, the same value is used for all
                wires. If a list is provided, it must match the length of
                ``wires`` and specifies per-wire probabilities.

        Raises:
            ValueError: If a list of probabilities is provided whose length
                does not match the number of wires.
        """

        # check inputs
        if isinstance(p, float):
            ps = [p] * len(wires)  # same error for all wires
        else:
            if len(p) != len(wires):
                raise ValueError(f'Inconsistent number of wires ({len(wires)}) and number of p`s ({len(p)}).')
            ps = p  # individual (potentially different) probability for each wire

        # define non-trainable hyperparameters
        self._hyperparameters = {
            'ps': ps
        }

        # initialize the parent class
        super().__init__(wires=qml.wires.Wires(wires), id=f'p={p}')

    @property
    def num_params(self):
        """int: Number of trainable parameters (0)."""
        # no trainable parameters
        return 0

    @staticmethod
    def compute_decomposition(wires, ps):  # pylint: disable=arguments-differ  # noqa
        """Decompose the channel into per-wire PhaseFlip operations.

        Args:
            wires (qml.wires.Wires): Wires on which the channel acts.
            ps (list[float]): Phase-flip probabilities for each wire.

        Returns:
            list[qml.operation.Operator]: List of :class:`qml.PhaseFlip`
            operations, one per wire.
        """

        op_list = []
        for wire, p in zip(wires, ps):
            op_list.append(qml.PhaseFlip(p, wires=wire))
        return op_list


if __name__ == '__main__':
    @qml.qnode(qml.device("default.mixed", wires=[0, 1]), interface='torch')
    def circuit():
        PhaseFlipChannel(wires=[0, 1], p=[0.1, 0.2])
        return qml.state()
    _drawer = qml.draw(circuit, level=3, show_matrices=False)
    print(_drawer())
    _state = circuit()
    print(_state)
