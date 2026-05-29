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

"""Independent thermal-relaxation noise channel for VarEFTQC.

This module defines :class:`ThermalRelaxationChannel`, a convenience
wrapper that applies :class:`qml.ThermalRelaxationError` independently to
one or more wires, with per-wire idle times and T1/T2 parameters.
"""

import pennylane as qml


class ThermalRelaxationChannel(qml.operation.Operation):
    """Thermal relaxation channel on multiple wires (independent).

    For each wire, this operation applies a
    :class:`qml.ThermalRelaxationError` channel parameterized by an idle
    time (gate duration) ``tg``, T1 and T2 times, and an optional excited
    state population. All of these parameters can be given globally or
    per wire.

    Important:
        To be compatible with hardware backends, all times are expected in
        milliseconds (ms).

    Note:
        Using this operation requires a device capable of simulating error
        channels (e.g. ``"default.mixed"``).
    """

    # Can be defined on an arbitrary number of wires
    num_wires = None

    def __init__(self,
                 wires: list,
                 idle: float | list,
                 t1: float | list,
                 t2: float | list,
                 pe: float | list = 0.0
                 ):
        """Initialize the thermal-relaxation channel.

        Args:
            wires (list): List of wire labels on which to apply the channel.
            idle (float | list[float]): Idle (gate) duration ``tg`` in ms.
                If a single float is provided, the same value is used for
                all wires; otherwise the list must match the number of
                wires.
            t1 (float | list[float]): T1 relaxation time(s) in ms. Same
                broadcasting rules as ``idle``.
            t2 (float | list[float]): T2 relaxation time(s) in ms. Same
                broadcasting rules as ``idle``.
            pe (float | list[float]): Excited state population(s). If a
                single float is provided, the same population is used for
                all wires; otherwise the list must match the number of
                wires.

        Raises:
            ValueError: If any list argument length does not match the
                number of wires.
        """

        # check inputs
        if isinstance(idle, float):
            idles = [idle] * len(wires)  # same for all wires
        else:
            if len(idle) != len(wires):
                raise ValueError(f'Inconsistent number of wires ({len(wires)}) and number of idle`s ({len(idle)}).')
            idles = idle  # individual (potentially different) error duration for each wire
        if isinstance(t1, float):
            t1s = [t1] * len(wires)  # same for all wires
        else:
            if len(t1) != len(wires):
                raise ValueError(f'Inconsistent number of wires ({len(wires)}) and number of t1`s ({len(t1)}).')
            t1s = t1  # individual (potentially different) T1 times for each wire
        if isinstance(t2, float):
            t2s = [t2] * len(wires)  # same for all wires
        else:
            if len(t2) != len(wires):
                raise ValueError(f'Inconsistent number of wires ({len(wires)}) and number of t2`s ({len(t2)}).')
            t2s = t2  # individual (potentially different) T2 times duration for each wire
        if isinstance(pe, float):
            pes = [pe] * len(wires)  # same for all wires
        else:
            if len(pe) != len(wires):
                raise ValueError(f'Inconsistent number of wires ({len(wires)}) and number of pe`s ({len(pe)}).')
            pes = pe  # individual (potentially different) excited state populations for each wire

        # define non-trainable hyperparameters
        self._hyperparameters = {
            'idles': idles,
            't1s': t1s,
            't2s': t2s,
            'pes': pes
        }

        # initialize the parent class
        super().__init__(wires=qml.wires.Wires(wires), id=f'idles={idle},t1={t1},t2={t2}')

    @property
    def num_params(self):
        """int: Number of trainable parameters (0)."""
        # no trainable parameters
        return 0

    @staticmethod
    def compute_decomposition(wires, idles, t1s, t2s, pes):  # pylint: disable=arguments-differ  # noqa
        """Decompose into per-wire ThermalRelaxationError operations.

        Args:
            wires (qml.wires.Wires): Wires on which the channel acts.
            idles (list[float]): Idle durations ``tg`` (ms) for each wire.
            t1s (list[float]): T1 times (ms) for each wire.
            t2s (list[float]): T2 times (ms) for each wire.
            pes (list[float]): Excited state populations for each wire.

        Returns:
            list[qml.operation.Operator]: List of
            :class:`qml.ThermalRelaxationError` operations, one per wire.
        """

        op_list = []
        for wire, idle, t1, t2, pe in zip(wires, idles, t1s, t2s, pes):
            op_list.append(qml.ThermalRelaxationError(pe=pe, t1=t1, t2=t2, tg=idle, wires=wire))
        return op_list


if __name__ == '__main__':
    @qml.qnode(qml.device("default.mixed", wires=[0, 1]), interface='torch')
    def circuit():
        ThermalRelaxationChannel(wires=[0, 1], idle=0.001, t1=0.2, t2=0.1)
        return qml.state()
    _drawer = qml.draw(circuit, level=1, show_matrices=False)
    print(_drawer())
    _state = circuit()
    print(_state)
