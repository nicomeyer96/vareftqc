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

"""Strictly transversal encoded operations for VarEFTQC.

This module defines :class:`EncodedOperationStrictlyTransversal`, a
non-parameterized PennyLane operation that applies a logical gate
transversally across one or two code blocks.
"""

from typing import Any
import pennylane as qml


class EncodedOperationStrictlyTransversal(qml.operation.Operation):
    """Strictly transversal encoded logical operation.

    This operation applies a given 1- or 2-qubit gate strictly
    transversally:

    * For a 1-qubit operation, the gate is applied independently to each
      wire in ``wires_control``.
    * For a 2-qubit operation, the gate is applied pairwise between the
      corresponding wires in ``wires_control`` and ``wires_target``.

    The operation has no trainable parameters.
    """

    # Single or multi-qubit operations
    num_wires = None

    # Non-differentiable
    grad_recipe = None

    # no trainable parameters
    num_params = 0
    ndim_params = (0,)

    def __init__(self,
                 operation: Any,
                 wires_control: qml.wires.Wires,
                 wires_target: qml.wires.Wires = None,
                 operation_name: str = None
                 ):
        """Initialize a strictly transversal encoded operation.

        Args:
            operation (Any): PennyLane operation class implementing the
                logical gate (e.g. :class:`qml.X`, :class:`qml.CZ`). Its
                ``num_wires`` must be 1 or 2.
            wires_control (qml.wires.Wires): Control (or primary) wires for
                the transversal application.
            wires_target (qml.wires.Wires | None): Target wires for 2-qubit
                operations. For 1-qubit operations, this should be ``None``
                or an empty set.
            operation_name (str | None): Optional name override used in the
                operation ID. If ``None``, ``operation.name`` is used.

        Raises:
            ValueError: If the operation has more than 2 wires.
            RuntimeError: If the inferred number of registers does not match
                the operation order, or if the numbers of control and target
                wires are inconsistent for 2-qubit operations.
        """

        if operation.num_wires > 2:
            raise ValueError('Only strictly-transversal operations of an order up to two are supported.')
        num_registers = 2 if wires_target is not None and 0 < len(wires_target) else 1
        if num_registers != operation.num_wires:
            raise RuntimeError(f'Inconsistent number of registers ({num_registers}) '
                               f'and operation order ({operation.num_wires}).')
        if num_registers > 1 and len(wires_control) != len(wires_target):
            raise RuntimeError(f'Inconsistent number of control wires ({len(wires_control)}) '
                               f'and target wires ({len(wires_target)}).')

        # combine wires
        wires = wires_control + wires_target
        wire_indices_control = list(range(len(wires_control)))
        wire_indices_target = list(range(len(wires_control), len(wires_control) + len(wires_target)))

        # define non-trainable hyperparameters
        self._hyperparameters = {
            'operation': operation,
            'wire_indices_control': wire_indices_control,
            'wire_indices_target': wire_indices_target
        }

        # initialize the parent class
        super().__init__(wires=wires, id=f'{operation.name if operation_name is None else operation_name}')  # noqa

    @staticmethod
    def compute_decomposition(wires, operation, wire_indices_control, wire_indices_target):  # pylint: disable=arguments-differ  # noqa
        """Decompose the strictly transversal operation into native gates.

        For 1-qubit operations, applies ``operation`` to each wire in
        ``wires``. For 2-qubit operations, applies ``operation`` to each
        control–target pair specified by ``wire_indices_control`` and
        ``wire_indices_target``.

        Args:
            wires (qml.wires.Wires): Combined list of control and (optional)
                target wires.
            operation (Any): PennyLane operation class implementing the
                logical gate.
            wire_indices_control (list[int]): Indices of control wires within
                ``wires``.
            wire_indices_target (list[int]): Indices of target wires within
                ``wires``; empty for 1-qubit operations.

        Returns:
            list[qml.operation.Operator]: List of PennyLane operations
            implementing the transversal logical operation.
        """

        if len(wire_indices_target) > 0:  # two-qubit case
            return [operation(wires=[wires[c], wires[t]]) for c, t in zip(wire_indices_control, wire_indices_target)]
        else:
            return [operation(wires=wire) for wire in wires]
