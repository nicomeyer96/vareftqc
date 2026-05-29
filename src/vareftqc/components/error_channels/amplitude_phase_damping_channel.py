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

"""Combined amplitude and phase damping noise channel for VarEFTQC.

This module defines :class:`AmplitudePhaseDampingChannel`, a convenience
wrapper that applies amplitude damping followed by phase damping on each
wire, with per-wire or global damping strengths.
"""

import pennylane as qml


class AmplitudePhaseDampingChannel(qml.operation.Operation):
    """Amplitude followed by phase damping on multiple wires (independent).

    For each wire, this operation applies a :class:`qml.AmplitudeDamping`
    channel followed by a :class:`qml.PhaseDamping` channel. All wires can
    share the same damping probability or have individual values.

    Note:
        Using this operation requires a device capable of simulating error
        channels (e.g. ``"default.mixed"``).
    """

    # Can be defined on an arbitrary number of wires
    num_wires = None

    def __init__(self,
                 wires: list,
                 gamma: float | list
                 ):
        """Initialize the combined amplitude–phase damping channel.

        Args:
            wires (list): List of wire labels on which to apply the channel.
            gamma (float | list[float]): Damping probability(ies). If a
                single float is provided, the same value is used for all
                wires. If a list is provided, it must match the length of
                ``wires`` and specifies per-wire damping probabilities
                (used identically for amplitude and phase damping).

        Raises:
            ValueError: If a list of probabilities is provided whose length
                does not match the number of wires.
        """

        # check inputs
        if isinstance(gamma, float):
            gammas = [gamma] * len(wires)  # same error for all wires
        else:
            if len(gamma) != len(wires):
                raise ValueError(f'Inconsistent number of wires ({len(wires)}) and number of gamma`s ({len(gamma)}).')
            gammas = gamma  # individual (potentially different) probability for each wire

        # define non-trainable hyperparameters
        self._hyperparameters = {
            'gammas': gammas
        }

        # initialize the parent class
        super().__init__(wires=qml.wires.Wires(wires), id=f'gamma={gamma}')

    @property
    def num_params(self):
        """int: Number of trainable parameters (0)."""
        # no trainable parameters
        return 0

    @staticmethod
    def compute_decomposition(wires, gammas):  # pylint: disable=arguments-differ  # noqa
        """Decompose the channel into amplitude and phase damping per wire.

        Args:
            wires (qml.wires.Wires): Wires on which the channel acts.
            gammas (list[float]): Damping probabilities for each wire.

        Returns:
            list[qml.operation.Operator]: For each wire, a
            :class:`qml.AmplitudeDamping` followed by a
            :class:`qml.PhaseDamping` operation.
        """

        op_list = []
        for wire, gamma in zip(wires, gammas):
            op_list.append(qml.AmplitudeDamping(gamma, wires=wire))
            op_list.append(qml.PhaseDamping(gamma, wires=wire))
        return op_list


if __name__ == '__main__':
    @qml.qnode(qml.device("default.mixed", wires=3), interface='torch')
    def circuit():
        AmplitudePhaseDampingChannel(gamma=[0.1, 0.2, 0.3], wires=[0, 1, 2])
        return qml.state()
    _drawer = qml.draw(circuit, level=3, show_matrices=False)
    print(_drawer())
    _state = circuit()
    print(_state)
