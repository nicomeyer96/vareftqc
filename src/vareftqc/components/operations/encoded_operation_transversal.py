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

"""Transversal encoded operations for VarEFTQC.

This module defines :class:`EncodedOperationTransversal`, a parameterized
PennyLane operation that applies a logical gate transversally across one or
two code blocks, with a repeated ansatz structure and optional control–target
flip.
"""

import pennylane as qml

from ...helpers.utils import CU3


class EncodedOperationTransversal(qml.operation.Operation):
    """Parameterized transversal encoded logical operation.

    This operation applies a given 1- or 2-qubit gate transversally in a
    repeated pattern:

    * For a 1-qubit operation, the chosen 1-qubit gate is applied to each
      control wire, repeated ``repetitions`` times.
    * For a 2-qubit operation, the chosen 2-qubit gate is applied pairwise
      between corresponding control and target wires, repeated
      ``repetitions`` times, with an optional flip of control and target.

    Parameters are provided as a 3D tensor of shape
    ``(n_control_wires, repetitions, gate.num_params)``.
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

    # default gateset: U3 for single-qubit operations, CU3 for two-qubit operations
    gate1q_default = qml.U3
    gate2q_default = CU3

    def __init__(self,
                 parameters,
                 order_operation: int,
                 wires_control: qml.wires.Wires,
                 wires_target: qml.wires.Wires = None,
                 gate1q: qml.operations.Operation = None,
                 gate2q: qml.operations.Operation = None,
                 flip: bool = False
                 ):
        """Initialize a transversal encoded logical operation.

                Args:
                    parameters (torch.Tensor | np.ndarray): Parameter tensor of shape
                        ``(n_control_wires, repetitions, n_params_per_gate)``, where
                        ``n_params_per_gate`` is determined by the chosen gate
                        (1-qubit or 2-qubit).
                    order_operation (int): Order of the logical operation (1 or 2).
                    wires_control (qml.wires.Wires): Control wires for the
                        transversal application.
                    wires_target (qml.wires.Wires | None): Target wires for 2-qubit
                        operations. For 1-qubit operations, this should be ``None``
                        or an empty set.
                    gate1q (qml.operations.Operation | None): 1-qubit gate class to
                        use. If ``None``, defaults to :data:`gate1q_default`.
                    gate2q (qml.operations.Operation | None): 2-qubit gate class to
                        use. If ``None``, defaults to :data:`gate2q_default`.
                    flip (bool): If ``True`` and ``order_operation == 2``, swap the
                        roles of control and target wires when applying the gate.

                Raises:
                    ValueError: If ``order_operation > 2``, if inappropriate wires
                        are provided for the selected order, or if parameter shapes
                        do not match the expected format.
                    RuntimeError: If the numbers of control and target wires are
                        inconsistent for 2-qubit operations.
                """

        if order_operation > 2:
            raise ValueError('Only transversal operations of an order up to two are supported.')
        num_registers = 2 if wires_target is not None and 0 < len(wires_target) else 1
        if num_registers > 1 and len(wires_control) != len(wires_target):
            raise RuntimeError(f'Inconsistent number of control wires ({len(wires_control)}) '
                               f'and target wires ({len(wires_target)}).')
        gate1q, gate2q = self._check_gates(gate1q, gate2q, parameters, order_operation, wires_control, wires_target)

        # combine wires
        wires = wires_control + wires_target
        wire_indices_control = list(range(len(wires_control)))
        wire_indices_target = list(range(len(wires_control), len(wires_control) + len(wires_target)))

        # define non-trainable hyperparameters
        self._hyperparameters = {
            'wire_indices_control': wire_indices_control,
            'wire_indices_target': wire_indices_target,
            'gate1q': gate1q,
            'gate2q': gate2q,
            'flip': flip
        }

        # initialize the parent class
        super().__init__(parameters, wires=wires, id=f'{order_operation}-qubit'
                                                     f'{f',repeat={parameters.shape[1]}' if parameters.shape[1] > 1 else ''}'
                                                     f'{f',flipped' if flip and 2 == order_operation else ''})')  # noqa

    @staticmethod
    def compute_decomposition(parameters, wires, wire_indices_control, wire_indices_target, gate1q, gate2q, flip):  # pylint: disable=arguments-differ  # noqa
        """Decompose the transversal operation into native gates.

        For each repetition, this method applies:

        * in the 1-qubit case: ``gate1q`` to each wire independently;
        * in the 2-qubit case: ``gate2q`` to each control–target pair,
          optionally swapping control and target if ``flip`` is ``True``.

        Args:
            parameters (torch.Tensor | np.ndarray): Parameter tensor of shape
                ``(n_control_wires, repetitions, n_params_per_gate)``.
            wires (qml.wires.Wires): Combined list of control and target
                wires.
            wire_indices_control (list[int]): Indices of control wires within
                ``wires``.
            wire_indices_target (list[int]): Indices of target wires within
                ``wires``; empty for 1-qubit operations.
            gate1q (qml.operations.Operation): 1-qubit gate class.
            gate2q (qml.operations.Operation): 2-qubit gate class.
            flip (bool): If ``True`` and in the 2-qubit case, apply
                ``gate2q`` with control/target swapped.

        Returns:
            list[qml.operation.Operator]: List of PennyLane operations
            implementing the transversal logical operation.
        """

        op_list = []
        if 0 == len(wire_indices_target):  # single-qubit case
            for repeat in range(parameters.shape[1]):
                for wire_index, wire in enumerate(wires):
                    op_list.append(gate1q(*parameters[wire_index, repeat], wires=wire))
        else:  # two-qubit case
            for repeat in range(parameters.shape[1]):
                for wire_index_control, wire_index_target in zip(wire_indices_control, wire_indices_target):
                    if flip:  # interchange control and target
                        op_list.append(gate2q(*parameters[wire_index_control, repeat],
                                              wires=[wires[wire_index_target], wires[wire_index_control]]))
                    else:
                        op_list.append(gate2q(*parameters[wire_index_control, repeat],
                                              wires=[wires[wire_index_control], wires[wire_index_target]]))
        return op_list

    def _check_gates(self, gate1q, gate2q, parameters, order_operation, wires_control, wires_target):
        """Validate gates and parameter shapes for transversal operations.

        This method:

        * sets default 1-qubit and 2-qubit gates if not provided,
        * checks that the gate arities (``num_wires``) are correct,
        * verifies that target wires are only provided for 2-qubit
          operations, and
        * checks that the parameter tensor has the expected shape:

          - 1-qubit case: ``(n_control_wires, repetitions, gate1q.num_params)``
          - 2-qubit case: ``(n_control_wires, repetitions, gate2q.num_params)``

        Args:
            gate1q (qml.operations.Operation | None): 1-qubit gate class, or
                ``None`` to use the default.
            gate2q (qml.operations.Operation | None): 2-qubit gate class, or
                ``None`` to use the default.
            parameters (torch.Tensor | np.ndarray): Parameter tensor.
            order_operation (int): Operation order (1 or 2).
            wires_control (qml.wires.Wires): Control wires.
            wires_target (qml.wires.Wires | None): Target wires.

        Returns:
            tuple[qml.operations.Operation, qml.operations.Operation]:
            Validated ``(gate1q, gate2q)`` gate classes.

        Raises:
            ValueError: If gates have incompatible arities, if target wires
                are provided for a 1-qubit operation or missing for a
                2-qubit operation, or if the parameter shape does not match
                the expected format.
        """

        # check and set initial parametrized gates
        if gate1q is None:  # use default gates
            gate1q = self.gate1q_default
        else:
            if 1 != gate1q.num_wires:
                raise ValueError(f'The gate {gate1q} cannot be applied to 1 wire.')
        if gate2q is None:  # use default gates
            gate2q = self.gate2q_default
        else:
            if 2 != gate2q.num_wires:
                raise ValueError(f'The gate {gate2q} cannot be applied to 2 wires.')

        # handle single-qubit and two-qubit case separately
        if 1 == order_operation:  # single-qubit case
            if wires_target is not None and len(wires_target) > 0:
                raise ValueError('Target wires were provided for operation of order one.')
            if (3 != len(parameters.shape) or len(wires_control) != parameters.shape[0]
                    or gate1q.num_params != parameters.shape[-1]):
                raise ValueError(f'Parameters have to be of shape [{len(wires_control)}, repetitions, '
                                 f'{gate1q.num_params}]  (were {list(parameters.shape)}).')
        else:  # two-qubit case
            if wires_target is None:
                raise ValueError('No target wires were provided for operation of order two.')
            if (3 != len(parameters.shape) or len(wires_control) != parameters.shape[0]
                    or gate2q.num_params != parameters.shape[-1]):
                raise ValueError(f'Parameters have to be of shape [{len(wires_control)}, repetitions, '
                                 f'{gate2q.num_params}] (were {list(parameters.shape)}).')
        return gate1q, gate2q
