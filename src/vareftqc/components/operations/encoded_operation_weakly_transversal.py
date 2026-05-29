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

"""Weakly transversal encoded operations for VarEFTQC.

This module defines :class:`EncodedOperationWeaklyTransversal`, a
parameterized PennyLane operation that implements a fixed, universal
2-qubit layout (U3–CZ pattern) with 24 parameters per control wire.
"""

import pennylane as qml


class EncodedOperationWeaklyTransversal(qml.operation.Operation):
    """Parameterized weakly transversal encoded logical operation.

    This operation implements a fixed 2-qubit layout (a sequence of U3 and
    CZ gates) that is universal on two qubits. For each control–target
    wire pair, it uses 24 parameters arranged in a specific pattern of
    U3–CZ layers. The same pattern is applied to all control–target pairs.

    The operation has a single parameter tensor per control wire of shape
    ``(24,)``; the full parameter tensor has shape
    ``(n_control_wires, 24)``.
    """

    # Single or two-qubit operations
    num_wires = None

    # only one set of parameters
    num_params = 1

    # shape: (wires, repeat_ansatz, parameters_per_wire)
    ndim_params = (3,)

    # Note: The current mode does not support parameter-shift gradient computation, but the module could be extended
    #       by designing a respective `grad_recipe`. Other modes of differentiation are supported.
    grad_method = "A"
    grad_recipe = None

    def __init__(self,
                 parameters,
                 order_operation: int,
                 wires_control: qml.wires.Wires,
                 wires_target: qml.wires.Wires
                 ):
        """Initialize a weakly transversal encoded logical operation.

        Args:
            parameters (torch.Tensor | np.ndarray): Parameter tensor of shape
                ``(n_control_wires, 24)``, where each length-24 row
                parameterizes one control–target pair.
            order_operation (int): Operation order; must be ``2`` for
                weakly transversal operations.
            wires_control (qml.wires.Wires): Control wires.
            wires_target (qml.wires.Wires): Target wires. Must have the same
                length as ``wires_control``.

        Raises:
            ValueError: If ``order_operation != 2`` or if parameter shape is
                inconsistent.
            RuntimeError: If the numbers of control and target wires differ.
        """

        self._check_gates(parameters, order_operation, wires_control, wires_target)

        # combine wires
        wires = wires_control + wires_target
        wire_indices_control = list(range(len(wires_control)))
        wire_indices_target = list(range(len(wires_control), len(wires_control) + len(wires_target)))

        # define non-trainable hyperparameters
        self._hyperparameters = {
            'wire_indices_control': wire_indices_control,
            'wire_indices_target': wire_indices_target
        }

        # initialize the parent class
        super().__init__(parameters, wires=wires, id=f'{order_operation}-qubit')  # noqa

    @staticmethod
    def compute_decomposition(parameters, wires, wire_indices_control, wire_indices_target):  # pylint: disable=arguments-differ  # noqa
        """Decompose the weakly transversal operation into native gates.

        For each control–target pair, this method applies a fixed sequence of
        U3 and CZ gates (with 24 parameters per pair):

        * U3 on control, U3 on target,
        * CZ(control → target),
        * U3 on control, U3 on target,
        * CZ(target → control),
        * U3 on control, U3 on target,
        * CZ(control → target),
        * U3 on control, U3 on target.

        Args:
            parameters (torch.Tensor | np.ndarray): Tensor of shape
                ``(n_control_wires, 24)`` containing all parameters.
            wires (qml.wires.Wires): Combined list of control and target
                wires.
            wire_indices_control (list[int]): Indices of control wires
                within ``wires``.
            wire_indices_target (list[int]): Indices of target wires within
                ``wires``.

        Returns:
            list[qml.operation.Operator]: List of PennyLane operations
            implementing the weakly transversal encoded logical operation.
        """

        op_list = []
        for wire_index_control, wire_index_target in zip(wire_indices_control, wire_indices_target):
            op_list.append(qml.U3(*parameters[wire_index_control, 0:3], wires=[wires[wire_index_control]]))
            op_list.append(qml.U3(*parameters[wire_index_control, 3:6], wires=[wires[wire_index_target]]))
            op_list.append(qml.CZ(wires=[wires[wire_index_control], wires[wire_index_target]]))
            op_list.append(qml.U3(*parameters[wire_index_control, 6:9], wires=[wires[wire_index_control]]))
            op_list.append(qml.U3(*parameters[wire_index_control, 9:12], wires=[wires[wire_index_target]]))
            op_list.append(qml.CZ(wires=[wires[wire_index_target], wires[wire_index_control]]))
            op_list.append(qml.U3(*parameters[wire_index_control, 12:15], wires=[wires[wire_index_control]]))
            op_list.append(qml.U3(*parameters[wire_index_control, 15:18], wires=[wires[wire_index_target]]))
            op_list.append(qml.CZ(wires=[wires[wire_index_control], wires[wire_index_target]]))
            op_list.append(qml.U3(*parameters[wire_index_control, 18:21], wires=[wires[wire_index_control]]))
            op_list.append(qml.U3(*parameters[wire_index_control, 21:24], wires=[wires[wire_index_target]]))
        return op_list

    @staticmethod
    def _check_gates(parameters, order_operation, wires_control, wires_target):
        """Validate parameter shapes and wire counts for weakly transversal operations.

        This method ensures that:

        * the operation order is exactly 2,
        * the numbers of control and target wires match, and
        * the parameter tensor has shape ``(n_control_wires, 24)``.

        Args:
            parameters (torch.Tensor | np.ndarray): Parameter tensor.
            order_operation (int): Operation order.
            wires_control (qml.wires.Wires): Control wires.
            wires_target (qml.wires.Wires): Target wires.

        Raises:
            ValueError: If ``order_operation != 2`` or if the parameter
                shape is inconsistent with ``(len(wires_control), 24)``.
            RuntimeError: If the numbers of control and target wires differ.
        """

        if order_operation != 2:
            raise ValueError('Only weakly-transversal operations of order two are supported.')
        if len(wires_control) != len(wires_target):
            raise RuntimeError(f'Inconsistent number of control wires ({len(wires_control)}) '
                               f'and target wires ({len(wires_target)}).')
        if 2 != len(parameters.shape) or len(wires_control) != parameters.shape[0] or 24 != parameters.shape[-1]:
            raise ValueError(f'Parameters have to be of shape [{len(wires_control)}, 24] '
                             f'(were {list(parameters.shape)}).')
