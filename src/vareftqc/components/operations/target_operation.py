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

"""Target logical operations used in VarEFTQC.

This module defines :class:`TargetOperation`, a PennyLane operation that
wraps an ideal (non-parameterized) 1- or 2-qubit gate and is used as the
reference logical operation when training encoded implementations.
"""

from typing import Any
import pennylane as qml


class TargetOperation(qml.operation.Operation):
    """Wrapper for an ideal 1- or 2-qubit target operation.

    This operation applies a given PennyLane gate (e.g. ``qml.X``, ``qml.CZ``)
    as the logical target operation. It has no trainable parameters and is
    used as a reference when training encoded logical operations.
    """

    # Single or two-qubit operations
    num_wires = None

    # Non-differentiable
    grad_recipe = None

    # no trainable parameters
    num_params = 0
    ndim_params = (0,)

    def __init__(self,
                 operation: Any,
                 wire_control: qml.wires.Wires,
                 wire_target: qml.wires.Wires = None,
                 operation_name: str = None
                 ):
        """Initialize a target logical operation.

               Args:
                   operation (Any): PennyLane operation class implementing the
                       logical gate (e.g. :class:`qml.X`, :class:`qml.CZ`). Its
                       ``num_wires`` attribute must be 1 or 2.
                   wire_control (qml.wires.Wires): Wires corresponding to the
                       "control" or primary system (for 1-qubit gates, this is the
                       only wire set).
                   wire_target (qml.wires.Wires | None): Optional target wires for
                       2-qubit operations. For 1-qubit operations, this should be
                       an empty set or ``None``.
                   operation_name (str | None): Optional name override used in the
                       operation ID. If ``None``, ``operation.name`` is used.

               Raises:
                   ValueError: If the operation has more than 2 wires.
                   RuntimeError: If the total number of wires does not match
                       ``operation.num_wires``.
               """

        if operation.num_wires > 2:
            raise ValueError('Only target operations of an order up to two are supported.')
        wires = wire_control + wire_target
        if len(wires) != operation.num_wires:
            raise RuntimeError(f'Inconsistent number of wires ({len(wires)}) '
                               f'and operation order ({operation.num_wires}).')

        # define non-trainable hyperparameters
        self._hyperparameters = {
            'operation': operation
        }

        # initialize the parent class
        super().__init__(wires=wires, id=f'{operation.name if operation_name is None else operation_name}')  # noqa

    @staticmethod
    def compute_decomposition(wires, operation):  # pylint: disable=arguments-differ  # noqa
        """Decompose the target operation into its underlying gate.

        Args:
            wires (qml.wires.Wires): Wires on which to apply the operation.
            operation (Any): PennyLane operation class implementing the
                logical gate.

        Returns:
            list[qml.operation.Operator]: A single instance of the provided
            operation applied to the given wires.
        """

        return [operation(wires=wires)]
