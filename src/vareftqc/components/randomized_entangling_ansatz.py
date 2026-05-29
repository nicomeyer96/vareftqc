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

"""Randomized entangling ansatz used in VarEFTQC.

This module defines :class:`RandomizedEntanglingAnsatz`, a PennyLane
operation that implements a hardware-inspired variational circuit with:

* an initial layer of 1-qubit gates on all wires, and
* a sequence of blocks, each containing a 2-qubit gate followed by
  1-qubit gates on the involved wires,

with optional local and correlated depolarizing noise. The layout is
random but reproducible via an ``instance`` seed.
"""

import pennylane as qml
import numpy as np
import torch

from .error_channels import UniformDepolarizingChannel, CorrelatedUniformDepolarizingChannel


class RandomizedEntanglingAnsatz(qml.operation.Operation):
    """Randomized entangling variational ansatz.

    The ansatz consists of:

    * an initial layer of 1-qubit gates applied to each wire, and
    * a sequence of blocks, where each block:
      - selects a random control–target pair according to a connectivity
        graph,
      - applies a 2-qubit gate on that pair, and
      - applies 1-qubit gates on control and target wires.

    Optional single- and two-qubit depolarizing noise channels can be
    inserted after gates. The random layout is fixed by the ``instance``
    seed so that the same ansatz can be reproduced across runs.

    Attributes:
        num_wires (None): This operation can act on an arbitrary number of
            wires (at least two).
        grad_method (str): Gradient method; ``"A"`` indicates analytic
            gradients are not provided by a parameter-shift recipe here, but
            other differentiation modes are supported.
        gates1q_default (list[Operation]): Default 1-qubit gates (U3).
        gate2q_default (Operation): Default 2-qubit gate (CNOT).
        error1q_default (Operation): Default 1-qubit error channel
            (uniform depolarizing).
        error2q_default (Operation): Default 2-qubit error channel
            (correlated uniform depolarizing).
    """

    # Can be defined on an arbitrary number of wires (at least two)
    num_wires = None

    # Note: The current mode does not support parameter-shift gradient computation, but the module could be extended
    #       by designing a respective `grad_recipe`. Other modes of differentiation are supported.
    grad_method = "A"
    grad_recipe = None

    # default gateset: Rot for single-qubit operations, CNOT for two-qubit operations
    gates1q_default = [qml.U3]  # [qml.Rot]
    gate2q_default = qml.CNOT  # qml.CNOT

    # default error channels: (correlated) uniform depolarizing noise for (two-qubit) single-qubit operations
    error1q_default = UniformDepolarizingChannel
    error2q_default = CorrelatedUniformDepolarizingChannel

    def __init__(self,
                 parameters_initial,
                 parameters_block,
                 instance: int,
                 wires: list,
                 gates1q: list[qml.operation.Operation] = None,
                 gate2q: qml.operation.Operation = None,
                 error1q: float | tuple[qml.operation.Operation, float] = None,
                 error2q: float | tuple[qml.operation.Operation, float] = None,
                 connectivity: dict = None):
        """Construct a randomized entangling ansatz operation.

        Args:
            parameters_initial (array-like): Initial-layer parameters of
                shape ``(n_wires, n_params_per_wire)``.
            parameters_block (array-like): Block parameters of shape
                ``(n_blocks, n_params_per_block)``.
            instance (int): Seed used to fix the random layout (choice of
                control/target pairs per block).
            wires (list): List of wire labels on which the ansatz acts.
                Must contain at least two elements.
            gates1q (list[Operation] | None): List of 1-qubit gate classes
                used in the initial and block layers. If ``None``, defaults
                to :data:`gates1q_default`.
            gate2q (Operation | None): 2-qubit gate class used in each block.
                If ``None``, defaults to :data:`gate2q_default`.
            error1q (float | tuple[Operation, float] | None): 1-qubit error
                specification. If a float is given, it is interpreted as the
                strength of the default 1-qubit error channel.
                If ``None``, no 1-qubit error is applied.
            error2q (float | tuple[Operation, float] | None): 2-qubit error
                specification, analogous to ``error1q``.
            connectivity (dict | None): Optional connectivity dictionary
                mapping each wire to a list of allowed control/target partners.
                If ``None``, all-to-all connectivity is assumed.

        Raises:
            ValueError: If fewer than two wires are provided, or if the given
                parameters do not match the expected shapes for the chosen
                gateset, or if the connectivity is invalid.
        """

        # check for number of wires
        if len(wires) <= 1:
            raise ValueError("At least two wires have to be provided.")

        # check and set gates
        gates1q, gate2q = self._check_gates(gates1q, gate2q, wires, parameters_initial, parameters_block)

        # check and set gate errors
        error1q, error2q = self._check_errors(error1q, error2q)

        # all wires that the operator acts on
        wires = qml.wires.Wires(wires)

        # check and set connectivity structure
        connectivity = self._check_connectivity(connectivity, wires)

        blocks = parameters_block.shape[0]
        # define non-trainable hyperparameters
        self._hyperparameters = {
            'instance': instance,
            'blocks': blocks,
            'gates1q': gates1q,
            'gate2q': gate2q,
            'error1q': error1q,
            'error2q': error2q,
            'connectivity': connectivity
        }

        # initialize the parent class
        super().__init__(parameters_initial, parameters_block, wires=wires,
                         id=f'#={blocks},instance={instance}'
                            f'{f',error1q={error1q[1]}' if 0.0 < error1q[1] else ''}'
                            f'{f',error2q={error2q[1]}' if 0.0 < error2q[1] else ''}')  # noqa

    @property
    def num_params(self):
        """int: Number of parameter **sets** for this operation (2)."""
        # set of initial parameters, set of block parameters
        return 2

    @property
    def ndim_params(self):
        """tuple[int, int]: Parameter tensor dimensions for initial and block layers.

        Returns:
            tuple[int, int]: A pair ``(2, 2)``, indicating that both
            ``parameters_initial`` and ``parameters_block`` are rank-2
            tensors.
        """

        # for initial parameters: (wires, parameters_per_wire), for block parameters: (blocks, parameters_per_block)
        return 2, 2

    @staticmethod
    def compute_decomposition(parameters_initial, parameters_block, wires, instance, blocks,  # noqa
                              gates1q, gate2q, error1q, error2q, connectivity):  # pylint: disable=arguments-differ  # noqa
        """Decompose the ansatz into a list of PennyLane operations.

        This method is called by PennyLane to expand the custom operation
        into native operations. It:

        * sets a random number generator with the given ``instance`` seed,
        * applies an initial layer of 1-qubit gates (and optional 1-qubit
          noise) on each wire, and
        * for each block:
          - samples a target wire and a connected control wire,
          - applies a 2-qubit gate (and optional 2-qubit noise),
          - applies 1-qubit gates (and optional noise) on control and target.

        Args:
            parameters_initial (array-like): Initial-layer parameters.
            parameters_block (array-like): Block parameters.
            wires (qml.wires.Wires): Wires on which the ansatz acts.
            instance (int): Seed used for random layout.
            blocks (int): Number of blocks.
            gates1q (list[Operation]): 1-qubit gates used in the ansatz.
            gate2q (Operation): 2-qubit gate used in the ansatz.
            error1q (tuple[Operation | None, float]): 1-qubit error channel
                and strength.
            error2q (tuple[Operation | None, float]): 2-qubit error channel
                and strength.
            connectivity (dict): Connectivity mapping specifying allowed
                control/target pairs.

        Returns:
            list[qml.operation.Operator]: List of PennyLane operations
            implementing the ansatz.
        """

        # set seed to fix ansatz instance
        rng = np.random.default_rng(seed=instance)
        op_list = []

        # initial layer with single-qubit operation on each qubit
        for wire_index, wire in enumerate(wires):
            start_index = 0
            for gate in gates1q:
                op_list.append(gate(*parameters_initial[wire_index, start_index:start_index+gate.num_params],
                                    wires=wire))
                start_index += gate.num_params
            # apply single-qubit error (for more accurate behavior this could also be done after each individual gate)
            if error1q[1] > 0.0:
                op_list.append(error1q[0]([wire], error1q[1]))
        # consecutive blocks, each layer contains one (randomly-placed) 2-qubit gate, followed by two single-qubit gates
        for block in range(blocks):
            # randomly select target and control qubit
            target = rng.choice(wires)
            potential_controls = connectivity[target]  # connected qubits
            control = rng.choice(potential_controls)

            # apply two-qubit gate, potentially followed by two-qubit error
            start_index = 0
            op_list.append(gate2q(*parameters_block[block, start_index:start_index+gate2q.num_params],
                                  wires=[control, target]))
            if error2q[1] > 0.0:
                op_list.append(error2q[0]([control, target], error2q[1]))
            start_index += gate2q.num_params

            # apply first single-qubit gate (on control), potentially followed by single-qubit error
            for gate in gates1q:  # apply single-qubit gates to control
                op_list.append(gate(*parameters_block[block, start_index:start_index + gate.num_params],
                                    wires=control))
                start_index += gate.num_params
            if error1q[1] > 0.0:
                op_list.append(error1q[0]([control], error1q[1]))

            # apply second single-qubit gate (on target), potentially followed by single-qubit error
            for gate in gates1q:  # apply single-qubit gates to target
                op_list.append(gate(*parameters_block[block, start_index:start_index + gate.num_params],
                                    wires=target))
                start_index += gate.num_params
            if error1q[1] > 0.0:
                op_list.append(error1q[0]([target], error1q[1]))

        return op_list

    def _check_gates(self, gates1q, gate2q, wires, parameters_initial, parameters_block):
        """Validate and instantiate the gateset, and check parameter shapes.

        This method:

        * sets default 1-qubit and 2-qubit gates if not provided,
        * verifies that 1-qubit gates have ``num_wires == 1`` and the 2-qubit
          gate has ``num_wires == 2``, and
        * checks that ``parameters_initial`` and ``parameters_block`` match the
          expected shapes for the chosen gateset.

        Args:
            gates1q (list[Operation] | None): 1-qubit gate classes, or
                ``None`` to use the defaults.
            gate2q (Operation | None): 2-qubit gate class, or ``None`` to
                use the default.
            wires (Sequence[Any]): Wires on which the ansatz acts.
            parameters_initial (torch.Tensor | np.ndarray): Initial-layer
                parameters.
            parameters_block (torch.Tensor | np.ndarray): Block parameters.

        Returns:
            tuple[list[Operation], Operation]: Validated ``(gates1q, gate2q)``.

        Raises:
            ValueError: If gate arities are incompatible with their intended
                use or if parameter shapes do not match the expected sizes.
        """

        # check and set initial parametrized gates
        if gates1q is None:  # use default gates
            gates1q = self.gates1q_default
        else:
            for g in gates1q:
                if 1 != g.num_wires:
                    raise ValueError(f'The gate {g} cannot be applied to 1 wire.')
        num_parameters_per_wire = np.sum([g.num_params for g in gates1q])
        if (len(wires), num_parameters_per_wire) != parameters_initial.shape:
            raise ValueError(f'The parameters for the initial layer have to be of shape '
                             f'[{len(wires)}, {num_parameters_per_wire}] (were {list(parameters_initial.shape)}).')

        # check and set block gates
        if gate2q is None:  # use default gates
            gate2q = self.gate2q_default
        else:
            if 2 != gate2q.num_wires:
                raise ValueError(f'The gate {gate2q} cannot be applied to 2 wires.')
        num_parameters_per_block = gate2q.num_params + 2 * num_parameters_per_wire
        if parameters_block.shape[1] != num_parameters_per_block:
            raise ValueError(f'The parameters for the blocks have to be of shape '
                             f'[_,{num_parameters_per_block}] (were {list(parameters_block.shape)}).')
        return gates1q, gate2q

    def _check_errors(self, error1q, error2q):
        """Validate and instantiate error-channel specifications.

        This method converts user-friendly error specifications into tuples
        ``(channel_class, strength)``:

        * If ``error1q``/``error2q`` is ``None``, no error is applied
          (strength ``0.0``).
        * If a float is given, it is interpreted as the strength for the
          default 1- or 2-qubit error channel.
        * If a tuple is already given, it is passed through unchanged.

        Args:
            error1q (float | tuple[Operation, float] | None): 1-qubit error
                specification.
            error2q (float | tuple[Operation, float] | None): 2-qubit error
                specification.

        Returns:
            tuple[tuple[Operation | None, float], tuple[Operation | None, float]]:
            Normalized error specifications ``(error1q, error2q)``.
        """

        # check and set single-qubit errors
        if error1q is None:
            error1q = (None, 0.0)
        else:
            if isinstance(error1q, float):  # only error strength is set, i.e. use standard uniform depolarizing channel
                error1q = (self.error1q_default, error1q)

        # check and set two-qubit errors
        if error2q is None:
            error2q = (None, 0.0)
        else:
            if isinstance(error2q, float):  # only error strength is set, i.e. use standard uniform depolarizing channel
                error2q = (self.error2q_default, error2q)

        return error1q, error2q

    @staticmethod
    def _check_connectivity(connectivity, wires):
        """Validate or construct a connectivity graph for the ansatz.

        If ``connectivity`` is ``None``, a default all-to-all connectivity is
        constructed such that each wire is connected to all other wires.
        Otherwise, this method checks that:

        * all keys in ``connectivity`` are valid wires,
        * no wire is connected to itself,
        * each wire has at least one neighbor,
        * every wire in ``wires`` appears as a key, and
        * the connectivity is symmetric (``i in conn[j]`` iff
          ``j in conn[i]``).

        Args:
            connectivity (dict | None): Optional user-provided connectivity
                mapping.
            wires (Sequence[Any]): Wires for which connectivity is defined.

        Returns:
            dict: Validated connectivity dictionary mapping each wire to a
            list of connected wires.

        Raises:
            ValueError: If the connectivity structure is incomplete or
                inconsistent.
        """

        # check connectivity structure
        if connectivity is None:  # default: set up all-to-all connectivity
            connectivity = {wire: [w for w in wires if w != wire] for wire in wires}
        else:
            keys = list(connectivity.keys())
            for key in keys:
                if key not in wires:
                    raise ValueError(f'The wire {key} from the connectivity dictionary is not in the list of wires.')
                if key in connectivity[key]:
                    raise ValueError(f'The wire {key} is listed as connected to itself.')
                if len(connectivity[key]) < 1:
                    raise ValueError(f'The wire {key} is not connected to any other wire.')
            for wire in wires:
                if wire not in keys:
                    raise ValueError(f'The wire {wire} from the list of wires is not in the connectivity dictionary.')
            for key in keys:
                values = connectivity[key]
                for value in values:
                    if key not in connectivity[value]:
                        raise ValueError(
                            f'The wire {key} is connected to {value}, but {value} is not connected to {key}.')  # noqa
        return connectivity


if __name__ == '__main__':
    _wires = [0, 1, 2]
    _blocks = 10
    _connectivity = {
        0: [1, 2],
        1: [0],
        2: [0]
    }

    @qml.qnode(qml.device("default.mixed", wires=_wires), interface='torch')
    def circuit(parameters_initial, parameters_block):
        RandomizedEntanglingAnsatz(parameters_initial, parameters_block,
                                   instance=42, wires=_wires, connectivity=_connectivity)
        qml.adjoint(
            RandomizedEntanglingAnsatz(parameters_initial, parameters_block,
                                       instance=42, wires=_wires, connectivity=_connectivity)
        )
        return qml.state()

    torch.manual_seed(42)
    _parameters_initial = torch.rand(len(_wires), 3, requires_grad=True)
    _parameters_block = torch.rand(_blocks, 9, requires_grad=True)

    _drawer = qml.draw(circuit, level=1, show_matrices=False)
    # _drawer = qml.draw(circuit, level=3, show_matrices=False)
    print(_drawer(_parameters_initial, _parameters_block))
    _state = circuit(_parameters_initial, _parameters_block)
    print(_state)
