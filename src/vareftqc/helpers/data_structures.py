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

"""Data structures for codes, noise models, variational parameters, and training
configuration in the VarEFTQC module.

This module defines:

* logical operations and their parameterizations,
* encoding and recovery ansatz specifications,
* code-level properties (wire layout, operations),
* noise model configuration, and
* parameter containers used during training.
"""

from dataclasses import dataclass, field
import pennylane as qml
from typing import Optional, Any, Tuple, List
import torch
import numpy as np

from .utils import CU3, CS

# logical one and two-qubit operations to be learned (extend to allow for more)
OPERATIONS_1Q = {'X': qml.X, 'Z': qml.Z, 'H': qml.H, 'S': qml.S, 'T': qml.T}
OPERATIONS_2Q = {'CX': qml.CNOT, 'CZ': qml.CZ, 'CS': CS}

# gate sets for parameterized one and two-qubit ansätze
GATES_1Q = {'U3': qml.U3, 'Rot': qml.Rot, 'RX': qml.RX, 'RZ': qml.RZ}
GATES_2Q = {'CU3': CU3, 'CRot': qml.CRot, 'CX': qml.CNOT, 'CZ': qml.CZ}


@dataclass
class OperationProperties:
    """Specification of a logical one- or two-qubit operation.

    An operation can be provided either as a static QASM circuit or as a
    parameterized ansatz in one of several flavors:
    strictly transversal, transversal, weakly transversal, or
    non-transversal. The class determines the gate order (1q or 2q),
    validates the configuration, and computes the expected parameter shapes.
    """

    name: str
    num_wires: int
    qasm: Optional[Tuple[str, list[str], list[str], list[str], list[str]]] = None
    strictly_transversal: Optional[bool] = False
    transversal: Optional[bool] = False
    weakly_transversal: Optional[bool] = False
    repeat: Optional[int] = None
    flip: Optional[bool] = False
    blocks: Optional[int] = None
    instance: Optional[int] = None
    gate_1q: Optional[str] = 'U3'
    gate_2q: Optional[str] = 'CU3'
    gate: Any = field(init=False)
    order: int = field(init=False)
    gateset: None | Tuple[Any, Any] = field(init=False)
    parameter_shape: Optional[tuple] = field(init=False)

    def __post_init__(self):
        if self.name in OPERATIONS_1Q.keys():
            self.gate = OPERATIONS_1Q.get(self.name)
            self.order = 1
        elif self.name in OPERATIONS_2Q.keys():
            self.gate = OPERATIONS_2Q.get(self.name)
            self.order = 2
        else:
            raise ValueError(f'Operation {self.name} unknown.')
        if self.qasm is None:
            if GATES_1Q.get(self.gate_1q, None) is None:
                raise ValueError(f'The 1-qubit gate {self.gate_1q} is not supported.')
            if GATES_2Q.get(self.gate_2q, None) is None:
                raise ValueError(f'The 2-qubit gate {self.gate_2q} is not supported.')
            self.gateset = (GATES_1Q[self.gate_1q], GATES_2Q[self.gate_2q])
            if ((self.strictly_transversal and self.weakly_transversal)
                    or (self.strictly_transversal and self.transversal)
                    or (self.weakly_transversal and self.transversal)):
                raise ValueError('The flags `strictly_transversal`, `transversal`, and `weakly_transversal` are '
                                 'mutually exclusive.`')
            if self.strictly_transversal:
                self.parameter_shape = 0, 0
            elif self.transversal:
                if self.repeat is None:
                    raise ValueError('For `transversal` operation `repeat` must be set.')
                if 1 == self.order:
                    self.parameter_shape = (self.num_wires, self.repeat, GATES_1Q[self.gate_1q].num_params), 0
                elif 2 == self.order:
                    self.parameter_shape = (self.num_wires, self.repeat, GATES_2Q[self.gate_2q].num_params), 0
                else:
                    raise RuntimeError(f'Operation of order {self.order} not supported.')
            elif self.weakly_transversal:
                # universal two-qubit ansatz with 24 trainable parameters
                if 2 != self.order:
                    raise RuntimeError('Weakly transversal operation is only supported for 2-qubit operations.')
                self.parameter_shape = (self.num_wires, 8 * 3), 0
            else:
                if self.blocks is None:
                    raise ValueError(f'For `non-transversal` operation `blocks_operation` must be set.')
                if self.instance is None:
                    raise ValueError(f'For `non-transversal` operation `instance_ansatz` must be set.')
                if 1 == self.order:
                    if 0 == self.blocks:
                        print(f'Note: For operation `{self.name}` it is set `blocks_operation=0`, '
                              f'i.e. the ansatz is transversal.')
                    self.parameter_shape = ((self.num_wires, GATES_1Q[self.gate_1q].num_params),
                                            (self.blocks, 2 * GATES_1Q[self.gate_1q].num_params + GATES_2Q[self.gate_2q].num_params))
                else:
                    if 0 == self.blocks and self.order > 1:
                        raise RuntimeError(f'Tried to instantiate multi-qubit ansatz without any multi-qubit operations.')
                    self.parameter_shape = ((2 * self.num_wires, GATES_1Q[self.gate_1q].num_params),
                                            (self.blocks, 2 * GATES_1Q[self.gate_1q].num_params + GATES_2Q[self.gate_2q].num_params))
        else:
            self.transversal, self.parameter_shape = False, None
            if 0 == len(self.qasm[3]) + len(self.qasm[4]):  # target wires
                if 2 == self.order:
                    raise RuntimeError('Selected gate of order two, but no target wires were provided.')
            else:
                if 1 == self.order:
                    raise RuntimeError('Selected gate of order one, but target wires were provided.')
                if len(self.qasm[1]) != len(self.qasm[3]):
                    raise RuntimeError(f'Inconsistent number of data wires ({len(self.qasm[1])}) and data target '
                                       f'wires ({len(self.qasm[3])}).')
                if len(self.qasm[2]) != len(self.qasm[4]):
                    raise RuntimeError(f'Inconsistent number of ancilla wires ({len(self.qasm[2])}) and ancilla target '
                                       f'wires ({len(self.qasm[4])}).')
        self.print()

    def print(self, prefix: str = 'Set up'):
        """Print a human-readable description of the operation configuration.

        Args:
            prefix (str): Prefix string for the printed message, typically
                indicating at which stage the object was created (e.g.
                ``"Set up"`` or ``"Loaded"``).
        """

        if self.qasm is not None:
            print(f'{prefix} static gate: `{self.name}` ({self.order}-qubit).')
        elif self.strictly_transversal:
            print(f'{prefix} strictly-transversal gate: `{self.name}` ({self.order}-qubit).')
        elif self.transversal:
            print(f'{prefix} transversal gate: `{self.name}` ({self.order}-qubit,{' flipped,' if 2 == self.order 
                                                                                                 and self.flip else ''} '
                  f'parameters={self.parameter_shape[0]}).')
        elif self.weakly_transversal:
            print(f'{prefix} weakly-transversal gate: `{self.name}` ({self.order}-qubit, '
                  f'parameters={self.parameter_shape[0]}).')
        else:
            print(f'{prefix} non-transversal gate: `{self.name}` ({self.order}-qubit, instance={self.instance}, '
                  f'parameters={self.parameter_shape[0]}|{self.parameter_shape[1]}).')


@dataclass
class EncodingProperties:
    """Print a human-readable description of the operation configuration.

    Args:
        prefix (str): Prefix string for the printed message, typically
            indicating at which stage the object was created (e.g.
            ``"Set up"`` or ``"Loaded"``).
    """

    num_wires_data: int
    num_wires_ancilla: int
    qasm: Optional[Tuple[str, list[str], list[str]]] = None
    blocks: Optional[int] = None
    instance: Optional[int] = None
    gates_1q: Optional[str | list[str]] = 'U3'
    gate_2q: Optional[str] = 'CU3'
    connectivity: Optional[dict] = None
    num_wires: int = field(init=False)
    trainable: bool = field(init=False)
    gateset: None | Tuple[List[Any], Any] = field(init=False)
    parameter_shape: None | Tuple[Tuple[int, int], Tuple[int, int]] = field(init=False)

    def __post_init__(self):
        if self.num_wires_data < 1:
            raise ValueError(f'At least one data wire is required ({self.num_wires_data} selected).')
        if self.num_wires_ancilla < 1:
            raise ValueError(f'At least one ancilla wire is required ({self.num_wires_ancilla} selected).')
        self.num_wires = self.num_wires_data + self.num_wires_ancilla
        if self.qasm is None:
            if self.blocks is None or self.instance is None:
                raise ValueError('Either `encoding` or `blocks_encoding` and `instance_encoding` must be specified.')
            gates_1q = [self.gates_1q] if isinstance(self.gates_1q, str) else self.gates_1q
            self.trainable = True
            gateset_1q, parameters_1q = [], 0
            for gate in gates_1q:
                if GATES_1Q.get(gate, None) is None:
                    raise ValueError(f'The 1-qubit gateset {gate} is not supported.')
                gateset_1q.append(GATES_1Q[gate])
                parameters_1q += GATES_1Q[gate].num_params
            if GATES_2Q.get(self.gate_2q, None) is None:
                raise ValueError(f'The 2-qubit gateset {self.gate_2q} is not supported.')
            self.gateset = (gateset_1q, GATES_2Q[self.gate_2q])
            self.parameter_shape = (self.num_wires, parameters_1q), (self.blocks, 2 * parameters_1q + GATES_2Q[
                self.gate_2q].num_params)
        else:
            self.trainable, self.gateset, self.parameter_shape = False, None, None
        self.print()

    def print(self, prefix: str = 'Set up'):
        """Print a summary of the encoding code parameters.

        Args:
            prefix (str): Prefix string for the printed message.
        """

        print(f'{prefix} (({self.num_wires}, {2**self.num_wires_data})) code ('
              f'{'static encoding' if not self.trainable else f'trainable encoding, '
                                                              f'instance={self.instance}, '
                                                              f'parameters={self.parameter_shape[0]}|'
                                                              f'{self.parameter_shape[1]}'}).')


@dataclass
class RecoveryProperties:
    """Specification of the recovery ansatz for the quantum code.

    The recovery can be static (provided as QASM) or trainable. In the
    trainable case, a block-structured ansatz is used over data, ancilla, and
    a dedicated recovery register.
    """

    num_wires_data: int
    num_wires_ancilla: int
    num_wires_recovery: int
    qasm: Optional[Tuple[str, list[str], list[str], list[str]]] = None
    blocks: Optional[int] = None
    instance: Optional[int] = None
    gates_1q: Optional[str | list[str]] = 'U3'
    gate_2q: Optional[str] = 'CU3'
    num_wires: int = field(init=False)
    trainable: bool = field(init=False)
    gateset: None | Tuple[List[Any], Any] = field(init=False)
    parameter_shape: None | Tuple[Tuple[int, int], Tuple[int, int]] = field(init=False)

    def __post_init__(self):
        if self.num_wires_data < 1:
            raise ValueError(f'At least one data wire is required ({self.num_wires_data} selected).')
        if self.num_wires_ancilla < 1:
            raise ValueError(f'At least one ancilla wire is required ({self.num_wires_ancilla} selected).')
        if self.num_wires_recovery < 1:
            print('Note: No separate recovery register selected, only a single round of recovery is advised.')
        self.num_wires = self.num_wires_data + self.num_wires_ancilla + self.num_wires_recovery
        if self.qasm is None:
            if self.blocks is None or self.instance is None:
                raise ValueError('Either `recovery` or `blocks_recovery` and `instance_recovery` must be specified.')
            gates_1q = [self.gates_1q] if isinstance(self.gates_1q, str) else self.gates_1q
            self.trainable = True
            if 0 == self.blocks:
                raise ValueError('The number of `blocks_recovery` cannot be zero.')
            gateset_1q, parameters_1q = [], 0
            for gate in gates_1q:
                if GATES_1Q.get(gate, None) is None:
                    raise ValueError(f'The 1-qubit gateset {gate} is not supported.')
                gateset_1q.append(GATES_1Q[gate])
                parameters_1q += GATES_1Q[gate].num_params
            if GATES_2Q.get(self.gate_2q, None) is None:
                raise ValueError(f'The 2-qubit gateset {self.gate_2q} is not supported.')
            self.gateset = (gateset_1q, GATES_2Q[self.gate_2q])
            self.parameter_shape = (self.num_wires, parameters_1q), (self.blocks, 2 * parameters_1q + GATES_2Q[self.gate_2q].num_params)
        else:
            self.trainable, self.gateset, self.parameter_shape = False, None, None
        self.print()

    def print(self, prefix: str = 'Set up'):
        """Print a summary of the recovery ansatz configuration.

        Args:
            prefix (str): Prefix string for the printed message.
        """

        print(f'{prefix} recovery operation with {self.num_wires_recovery} recovery wires '
              f'for [[{self.num_wires_data + self.num_wires_ancilla}, {self.num_wires_data}]] code ('
              f'{'static recovery' if not self.trainable else f'trainable recovery, '
                                                              f'instance={self.instance}, '
                                                              f'parameters={self.parameter_shape[0]}|'
                                                              f'{self.parameter_shape[1]}'}).')


@dataclass
class CodeProperties:
    """Global code configuration: encoding, recovery, and logical operations.

    This class ties together the encoding and (optional) recovery ansätze,
    as well as optional logical operations (static, strictly transversal,
    transversal, weakly transversal, and non-transversal). It also sets up
    the wire labels used throughout the circuits.
    """

    encoding_properties: EncodingProperties
    recovery_properties: None | RecoveryProperties
    operation_static: None | dict[str, OperationProperties]
    operation_strictly_transversal: None | dict[str, OperationProperties]
    operation_transversal: None | dict[str, OperationProperties]
    operation_weakly_transversal: None | dict[str, OperationProperties]
    operation_non_transversal: None | dict[str, OperationProperties]
    num_wires_data: int = field(init=False)
    num_wires_ancilla: int = field(init=False)
    num_wires_recovery: int = field(init=False)
    num_wires_encoding: int = field(init=False)
    num_wires: int = field(init=False)
    wires_data: qml.wires.Wires = field(init=False)
    wires_ancilla: qml.wires.Wires = field(init=False)
    wires_recovery: qml.wires.Wires = field(init=False)
    train_encoding: bool = field(init=False)
    train_recovery: bool = field(init=False)
    train_operation: bool = field(init=False)

    def __post_init__(self):
        self.num_wires_data = self.encoding_properties.num_wires_data
        self.num_wires_ancilla = self.encoding_properties.num_wires_ancilla
        self.num_wires_encoding = self.num_wires_data + self.num_wires_ancilla
        self.train_encoding = self.encoding_properties.trainable
        ##################
        # handle recovery
        ##################
        if self.recovery_properties is None:
            self.train_recovery, self.num_wires_recovery = False, 0
        else:
            if self.num_wires_data != self.recovery_properties.num_wires_data:
                raise RuntimeError(f'Inconsistent number of data wires in encoding ({self.num_wires_data})'
                                   f' and recovery ({self.recovery_properties.num_wires_data}).')
            if self.num_wires_ancilla != self.recovery_properties.num_wires_ancilla:
                raise RuntimeError(f'Inconsistent number of ancilla wires in encoding ({self.num_wires_ancilla})'
                                   f' and recovery ({self.recovery_properties.num_wires_ancilla}).')
            self.train_recovery = self.recovery_properties.trainable
            self.num_wires_recovery = self.recovery_properties.num_wires_recovery
        self.num_wires = self.num_wires_data + self.num_wires_ancilla + self.num_wires_recovery
        self.wires_data = qml.wires.Wires([f'd{i}' for i in range(self.num_wires_data)])
        self.wires_ancilla = qml.wires.Wires([f'a{i}' for i in range(self.num_wires_ancilla)])
        self.wires_recovery = qml.wires.Wires([f'r{i}' for i in range(self.num_wires_recovery)])
        ####################
        # handle operations
        ####################
        # static operations
        if self.operation_static is None:
            operations_static = []
        else:
            operations_static = list(self.operation_static.keys())
            for o in self.operation_static.values():
                if o.num_wires != self.num_wires_encoding:
                    raise RuntimeError(f'Inconsistent number of wires in encoding ({self.num_wires_encoding}) and wires'
                                       f' in static operation `{o.name}` ({o.num_wires}).')
        # strictly-transversal operations
        if self.operation_strictly_transversal is None:
            operations_strictly_transversal = []
        else:
            operations_strictly_transversal = list(self.operation_strictly_transversal.keys())
            for o in self.operation_strictly_transversal.values():
                if o.num_wires != self.num_wires_encoding:
                    raise RuntimeError(f'Inconsistent number of wires in encoding ({self.num_wires_encoding}) and wires'
                                       f'in strictly-transversal operation `{o.name}` ({o.num_wires}).')
        # transversal operations
        if self.operation_transversal is None:
            operations_transversal = []
        else:
            operations_transversal = list(self.operation_transversal.keys())
            for o in self.operation_transversal.values():
                if o.num_wires != self.num_wires_encoding:
                    raise RuntimeError(f'Inconsistent number of wires in encoding ({self.num_wires_encoding}) and wires'
                                       f'in transversal operation `{o.name}` ({o.num_wires}).')
        # weakly-transversal operations
        if self.operation_weakly_transversal is None:
            operations_weakly_transversal = []
        else:
            operations_weakly_transversal = list(self.operation_weakly_transversal.keys())
            for o in self.operation_weakly_transversal.values():
                if o.num_wires != self.num_wires_encoding:
                    raise RuntimeError(f'Inconsistent number of wires in encoding ({self.num_wires_encoding}) and wires'
                                       f'in weakly-transversal operation `{o.name}` ({o.num_wires}).')
        # non-transversal operations
        if self.operation_non_transversal is None:
            operations_non_transversal = []
        else:
            operations_non_transversal = list(self.operation_non_transversal.keys())
            for o in self.operation_non_transversal.values():
                if o.num_wires != self.num_wires_encoding:
                    raise RuntimeError(f'Inconsistent number of wires in encoding ({self.num_wires_encoding}) and wires'
                                       f'in non-transversal operation `{o.name}` ({o.num_wires}).')
        # combined
        operations = (operations_static + operations_strictly_transversal + operations_transversal
                      + operations_weakly_transversal + operations_non_transversal)
        if len(operations) != len(set(operations)):
            raise ValueError('Some operations have been set multiple times.')
        self.train_operation = True if (self.num_transversal_operation + self.num_weakly_transversal_operation
                                        + self.num_non_transversal_operation > 0) \
            else False

    @property
    def num_static_operation(self):
        """int: Number of static logical operations."""
        return 0 if self.operation_static is None else len(self.operation_static)

    @property
    def num_strictly_transversal_operation(self):
        """int: Number of strictly transversal logical operations."""
        return 0 if self.operation_strictly_transversal is None else len(self.operation_strictly_transversal)

    @property
    def num_transversal_operation(self):
        """int: Number of transversal logical operations."""
        return 0 if self.operation_transversal is None else len(self.operation_transversal)

    @property
    def num_weakly_transversal_operation(self):
        """int: Number of weakly transversal logical operations."""
        return 0 if self.operation_weakly_transversal is None else len(self.operation_weakly_transversal)

    @property
    def num_non_transversal_operation(self):
        """int: Number of non-transversal logical operations."""
        return 0 if self.operation_non_transversal is None else len(self.operation_non_transversal)


@dataclass
class NoiseProperties:
    """Description of the physical noise model used in training and testing.

    The noise model can be one of several supported types (e.g. depolarizing,
    Pauli, thermal relaxation). Depending on the selected model, additional
    parameters such as T1/T2 times or Pauli weights are required.
    """

    noise: str
    noise_strength: float | list[float]
    noise_pauli_x: float | list[float] = None,
    noise_pauli_z: float | list[float] = None,
    noise_t1: Optional[float] = None
    noise_t2: Optional[float] = None
    noise_asymmetry: Optional[float] = 1.0
    train_encoding: Optional[bool] = True

    def __post_init__(self):
        if 'thermal_relaxation' == self.noise:
            if self.noise_t1 is None or self.noise_t2 is None:
                raise ValueError('Both `noise_t1` and `noise_t2` must be specified if noise is `thermal_relaxation`.')
            print(f'Set up thermal relaxation noise of duration {self.noise_strength}ms '
                  f'(t1={self.noise_t1:.0f}ms, t2={self.noise_t2:.0f}ms).')
        elif 'pauli' == self.noise:
            if self.noise_pauli_x is None or self.noise_pauli_z is None:
                raise ValueError('Both `noise_pauli_x` and `noise_pauli_z` must be specified if noise is `pauli`.')
            print(f'Set up pauli noise of strength {self.noise_strength} (X-noise weight: {self.noise_pauli_x}, '
                  f'Z-noise weight: {self.noise_pauli_z}).')
        elif self.train_encoding:  # noise must be set
            if self.noise is None:
                raise ValueError('Argument `noise` must be specified for training encoding.')
            print(f'Set up {self.noise} noise of strength {self.noise_strength}'
                  f'{f' (asymmetry c={self.noise_asymmetry}).' if 'depolarizing' == self.noise and 1.0 != self.noise_asymmetry else '.'}')
        else:  # optionally set dummy noise (zero-noise), as not required for training logical operations
            if self.noise is None:
                self.noise = 'dummy'
                self.noise_strength = 0.0
                print('Set up dummy noise for static encoding (i.e. noise-free).')
            else:
                print(f'Set up {self.noise} noise of strength {self.noise_strength}'
                      f'{f' (asymmetry c={self.noise_asymmetry}).' if 'depolarizing' == self.noise and 1.0 != self.noise_asymmetry else '.'}')


@dataclass
class ParametersEncoding:
    """Container for encoding ansatz parameters.

    Handles initialization from scratch (random in ``[0, 1)``) or from
    previously trained parameters, validates shapes, and exposes utilities
    to access parameters with or without gradient tracking.
    """

    parameter_shape: Tuple[Tuple[int, int], Tuple[int, int]]
    random_number_generator: np.random.Generator
    parameters_encoding_trained: Optional[Any] = None
    # two parameter sets for encoding
    parameters_encoding_initial: None | torch.Tensor = field(init=False)
    parameters_encoding_block: None | torch.Tensor = field(init=False)

    def __post_init__(self):
        if self.parameter_shape is None:
            self.parameters_encoding_initial, self.parameters_encoding_block = None, None
        else:
            if self.parameters_encoding_trained is None:
                # initialize uniform at random in [0, 1)
                parameters_encoding_initial = self.random_number_generator.random(self.parameter_shape[0])
                parameters_encoding_block = self.random_number_generator.random(self.parameter_shape[1])
                print('Initializing encoding parameters uniform at random from [0, 1).')
            else:
                if not isinstance(self.parameters_encoding_trained, tuple) and 2 == len(self.parameters_encoding_trained):
                    raise RuntimeError('Trained encoding parameters were provided in wrong format.')
                parameters_encoding_initial = self.parameters_encoding_trained[0]
                if parameters_encoding_initial.shape != self.parameter_shape[0]:
                    raise RuntimeError(f'Encoding initial parameters expected to be of shape {self.parameter_shape[0]}, '
                                       f'but was {parameters_encoding_initial.shape}.')
                parameters_encoding_block = self.parameters_encoding_trained[1]
                if parameters_encoding_block.shape != self.parameter_shape[1]:
                    raise RuntimeError(f'Encoding block parameters expected to be of shape {self.parameter_shape[1]}, '
                                       f'but was {parameters_encoding_block.shape}.')
                print(f'Loading pre-trained encoding parameters.')

            self.parameters_encoding_initial = torch.tensor(parameters_encoding_initial, requires_grad=True)
            self.parameters_encoding_block = torch.tensor(parameters_encoding_block, requires_grad=True)

    def normalized_parameters(self, symmetry: float = 4 * np.pi):
        """Return encoding parameters reduced modulo a given symmetry.

        This is useful when gates have periodic parameters (e.g. the ``theta``
        angle of the U3 gate). Gradients are not propagated.

        Args:
            symmetry (float): Period used to wrap the parameters. Defaults to
                ``4 * pi`` to account for U3 parameter periodicity.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tensors of initial-layer and
            block parameters wrapped into the interval ``[0, symmetry)``.
        """

        with torch.no_grad():
            return torch.remainder(self.parameters_encoding_initial, symmetry), torch.remainder(self.parameters_encoding_block, symmetry)

    def parameters(self, grad: bool = True):
        """Return encoding parameters with optional gradient tracking.

        Args:
            grad (bool): If ``True``, return tensors that require gradients.
                If ``False``, return detached tensors. Defaults to ``True``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Initial-layer and block
            parameter tensors.
        """

        if grad:
            return self.parameters_encoding_initial, self.parameters_encoding_block
        else:
            return self.parameters_encoding_initial.detach(), self.parameters_encoding_block.detach()


@dataclass
class ParametersTransversalOperation:
    """Container for parameters of a single transversal logical operation.
    """

    parameter_shape: Tuple[int, int, int]
    random_number_generator: np.random.Generator
    operation_name: str
    parameters_operation_trained: Optional[Any] = None
    # one parameter set for transversal operation
    parameters_operation: torch.Tensor = field(init=False)

    def __post_init__(self):
        if self.parameters_operation_trained is None:
            # initialize uniform at random in [0, 1)
            parameters_operation = self.random_number_generator.random(self.parameter_shape[0])
            print(f'Initializing transversal {self.operation_name} operation parameters uniform at random from [0, 1).')
        else:
            parameters_operation = self.parameters_operation_trained
            if parameters_operation.shape != self.parameter_shape[0]:
                raise RuntimeError(
                    f'Transversal {self.operation_name} operation parameters expected to be of shape '
                    f'{self.parameter_shape[0]}, but was {parameters_operation.shape}.')
            print(f'Loading pre-trained transversal {self.operation_name} operation parameters.')

        self.parameters_operation = torch.tensor(parameters_operation, requires_grad=True)

    def normalized_parameters(self, symmetry: float = 4 * np.pi):
        """Return transversal-operation parameters reduced modulo a symmetry.

        Args:
            symmetry (float): Period used to wrap the parameters. Defaults to
                ``4 * pi`` for U3-like gates.

        Returns:
            torch.Tensor: Wrapped parameter tensor in ``[0, symmetry)``.
        """

        with torch.no_grad():
            return torch.remainder(self.parameters_operation, symmetry)

    def parameters(self, grad: bool = False):
        """Return transversal-operation parameters with optional gradients.

        Args:
            grad (bool): If ``True``, return a tensor that requires gradients.
                If ``False``, return a detached tensor. Defaults to ``False``.

        Returns:
            torch.Tensor: Parameter tensor.
        """

        if grad:
            return self.parameters_operation
        else:
            return self.parameters_operation.detach()


@dataclass
class ParametersWeaklyTransversalOperation:
    """Container for parameters of a weakly-transversal logical operation.
    """

    parameter_shape: Tuple[int, int]
    random_number_generator: np.random.Generator
    operation_name: str
    parameters_operation_trained: Optional[Any] = None
    # one parameter set for transversal operation
    parameters_operation: torch.Tensor = field(init=False)

    def __post_init__(self):
        if self.parameters_operation_trained is None:
            # initialize uniform at random in [0, 1)
            parameters_operation = self.random_number_generator.random(self.parameter_shape[0])
            print(f'Initializing weakly-transversal {self.operation_name} operation parameters uniform at random from [0, 1).')
        else:
            parameters_operation = self.parameters_operation_trained
            if parameters_operation.shape != self.parameter_shape[0]:
                raise RuntimeError(
                    f'Weakly-transversal {self.operation_name} operation parameters expected to be of shape '
                    f'{self.parameter_shape[0]}, but was {parameters_operation.shape}.')
            print(f'Loading pre-trained weakly-transversal {self.operation_name} operation parameters.')

        self.parameters_operation = torch.tensor(parameters_operation, requires_grad=True)

    def normalized_parameters(self, symmetry: float = 4 * np.pi):
        """Return weakly-transversal-operation parameters modulo a symmetry.

        Args:
            symmetry (float): Period used to wrap the parameters. Defaults to
                ``4 * pi`` for U3-like gates.

        Returns:
            torch.Tensor: Wrapped parameter tensor in ``[0, symmetry)``.
        """

        with torch.no_grad():
            return torch.remainder(self.parameters_operation, symmetry)

    def parameters(self, grad: bool = False):
        """Return weakly-transversal-operation parameters with optional gradients.

        Args:
            grad (bool): If ``True``, return a tensor that requires gradients.
                If ``False``, return a detached tensor. Defaults to ``False``.

        Returns:
            torch.Tensor: Parameter tensor.
        """

        if grad:
            return self.parameters_operation
        else:
            return self.parameters_operation.detach()


@dataclass
class ParametersNonTransversalOperation:
    """Container for parameters of a non-transversal logical operation.

    A non-transversal operation has separate parameter tensors for an initial
    layer and a repeated block structure.
    """

    parameter_shape: Tuple[Tuple[int, int], Tuple[int, int]]
    random_number_generator: np.random.Generator
    operation_name: str
    parameters_operation_trained: Optional[Any] = None
    # two parameter sets for non-transversal operation
    parameters_operation_initial: torch.Tensor = field(init=False)
    parameters_operation_block: torch.Tensor = field(init=False)

    def __post_init__(self):
        if self.parameters_operation_trained is None:
            parameters_operation_initial = self.random_number_generator.random(self.parameter_shape[0])
            parameters_operation_block = self.random_number_generator.random(self.parameter_shape[1])
            print(f'Initializing non-transversal {self.operation_name} operation parameters uniform at random from [0, 1).')
        else:
            if not isinstance(self.parameters_operation_trained, tuple) and 2 == len(self.parameters_operation_trained):
                raise RuntimeError(f'Trained non-transversal {self.operation_name} operation parameters were provided in '
                                   'wrong format.')
            parameters_operation_initial = self.parameters_operation_trained[0]
            if parameters_operation_initial.shape != self.parameter_shape[0]:
                raise RuntimeError(
                    f'Non-transversal {self.operation_name} operation initial parameters expected to be of shape '
                    f'{self.parameter_shape[0]}, but was {parameters_operation_initial.shape}.')
            parameters_operation_block = self.parameters_operation_trained[1]
            if parameters_operation_block.shape != self.parameter_shape[1]:
                raise RuntimeError(f'Non-transversal {self.operation_name} operation block parameters expected to be of '
                                   f'shape {self.parameter_shape[1]}, but was {parameters_operation_block.shape}.')
            print(f'Loading pre-trained non-transversal {self.operation_name} operation parameters.')

        self.parameters_operation_initial = torch.tensor(parameters_operation_initial, requires_grad=True)
        self.parameters_operation_block = torch.tensor(parameters_operation_block, requires_grad=True)

    def normalized_parameters(self, symmetry: float = 4 * np.pi):
        """Return non-transversal-operation parameters modulo a symmetry.

        Args:
            symmetry (float): Period used to wrap the parameters. Defaults to
                ``4 * pi`` for U3-like gates.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Wrapped initial-layer and block
            parameter tensors.
        """

        with torch.no_grad():
            return torch.remainder(self.parameters_operation_initial, symmetry), torch.remainder(self.parameters_operation_block, symmetry)

    def parameters(self, grad: bool = False):
        """Return non-transversal-operation parameters with optional gradients.

        Args:
            grad (bool): If ``True``, return tensors that require gradients.
                If ``False``, return detached tensors. Defaults to ``False``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Initial-layer and block
            parameter tensors.
        """

        if grad:
            return self.parameters_operation_initial, self.parameters_operation_block
        else:
            return self.parameters_operation_initial.detach(), self.parameters_operation_block.detach()


@dataclass
class ParametersOperation:
    """Aggregate parameter containers for all logical operations.

    This class instantiates and holds parameter objects for transversal,
    weakly-transversal, and non-transversal operations defined in the
    provided :class:`CodeProperties`.
    """

    code_properties: CodeProperties
    random_number_generator: np.random.Generator
    parameters_operation_transversal_trained: Optional[dict[str, Any]] = None
    parameters_operation_weakly_transversal_trained: Optional[dict[str, Any]] = None
    parameters_operation_non_transversal_trained: Optional[dict[str, Any]] = None
    parameters_operation_transversal: None | dict[str, ParametersTransversalOperation] = field(init=False)
    parameters_operation_weakly_transversal: None | dict[str, ParametersWeaklyTransversalOperation] = field(init=False)
    parameters_operation_non_transversal: None | dict[str, ParametersNonTransversalOperation] = field(init=False)

    def __post_init__(self):
        # transversal
        if self.code_properties.operation_transversal is None:
            self.parameters_operation_transversal = None
        else:
            self.parameters_operation_transversal = {}
            for ot in self.code_properties.operation_transversal.values():
                self.parameters_operation_transversal[ot.name] = ParametersTransversalOperation(
                    parameter_shape=ot.parameter_shape, random_number_generator=self.random_number_generator,  # noqa
                    operation_name=ot.name,
                    parameters_operation_trained=None if self.parameters_operation_transversal_trained is None
                    else self.parameters_operation_transversal_trained.get(ot.name, None)
                )

        # weakly-transversal
        if self.code_properties.operation_weakly_transversal is None:
            self.parameters_operation_weakly_transversal = None
        else:
            self.parameters_operation_weakly_transversal = {}
            for owt in self.code_properties.operation_weakly_transversal.values():
                self.parameters_operation_weakly_transversal[owt.name] = ParametersWeaklyTransversalOperation(
                    parameter_shape=owt.parameter_shape, random_number_generator=self.random_number_generator,  # noqa
                    operation_name=owt.name,
                    parameters_operation_trained=None if self.parameters_operation_weakly_transversal_trained is None
                    else self.parameters_operation_weakly_transversal_trained.get(owt.name, None)
                )

        # non-transversal
        if self.code_properties.operation_non_transversal is None:
            self.parameters_operation_non_transversal = None
        else:
            self.parameters_operation_non_transversal = {}
            for ont in self.code_properties.operation_non_transversal.values():
                self.parameters_operation_non_transversal[ont.name] = ParametersNonTransversalOperation(
                    parameter_shape=ont.parameter_shape, random_number_generator=self.random_number_generator,  # noqa
                    operation_name=ont.name,
                    parameters_operation_trained=None if self.parameters_operation_non_transversal_trained is None
                    else self.parameters_operation_non_transversal_trained.get(ont.name, None)
                )

    def parameters_transversal(self):
        """Return a list of transversal-operation parameter tensors.

        Returns:
            list[torch.Tensor]: List of parameter tensors (with gradients)
            for all transversal operations, ordered by the internal dictionary
            order.
        """

        return [parameters.parameters(grad=True) for parameters in self.parameters_operation_transversal.values()]

    def parameters_weakly_transversal(self):
        """Return a list of weakly-transversal-operation parameter tensors.

        Returns:
            list[torch.Tensor]: List of parameter tensors (with gradients)
            for all weakly-transversal operations.
        """

        return [parameters.parameters(grad=True) for parameters in self.parameters_operation_weakly_transversal.values()]

    def parameters_non_transversal(self):
        """Return a flat list of non-transversal-operation parameter tensors.

        The list contains all initial-layer tensors followed by all block
        tensors for each non-transversal operation.

        Returns:
            list[torch.Tensor]: List of parameter tensors (with gradients) for
            initial and block parameters of all non-transversal operations.
        """

        return ([parameters.parameters(grad=True)[0] for parameters in self.parameters_operation_non_transversal.values()] +
                [parameters.parameters(grad=True)[1] for parameters in self.parameters_operation_non_transversal.values()])


@dataclass
class ParametersRecovery:
    """Container for recovery ansatz parameters.

    Handles initialization from scratch or from pretrained parameters,
    validates shapes, and exposes utilities to access parameters with or
    without gradient tracking.
    """

    parameter_shape: Tuple[Tuple[int, int], Tuple[int, int]]
    random_number_generator: np.random.Generator
    parameters_recovery_trained: Optional[Any] = None
    # two parameter sets for encoding
    parameters_recovery_initial: None | torch.Tensor = field(init=False)
    parameters_recovery_block: None | torch.Tensor = field(init=False)

    def __post_init__(self):
        if self.parameter_shape is None:
            self.parameters_recovery_initial, self.parameters_recovery_block = None, None
        else:
            if self.parameters_recovery_trained is None:
                # initialize uniform at random in [0, 1)
                parameters_recovery_initial = self.random_number_generator.random(self.parameter_shape[0])
                parameters_recovery_block = self.random_number_generator.random(self.parameter_shape[1])
                print('Initializing recovery parameters uniform at random from [0, 1).')
            else:
                if not isinstance(self.parameters_recovery_trained, tuple) and 2 == len(
                        self.parameters_recovery_trained):  # noqa
                    raise RuntimeError('Trained recovery parameters were provided in wrong format.')
                parameters_recovery_initial = self.parameters_recovery_trained[0]
                if parameters_recovery_initial.shape != self.parameter_shape[0]:
                    raise RuntimeError(
                        f'Recovery initial parameters expected to be of shape {self.parameter_shape[0]}, '  # noqa
                        f'but was {parameters_recovery_initial.shape}.')
                parameters_recovery_block = self.parameters_recovery_trained[1]
                if parameters_recovery_block.shape != self.parameter_shape[1]:
                    raise RuntimeError(f'Recovery block parameters expected to be of shape {self.parameter_shape[1]}, '
                                       f'but was {parameters_recovery_block.shape}.')
                print(f'Loading pre-trained recovery parameters.')
            self.parameters_recovery_initial = torch.tensor(parameters_recovery_initial, requires_grad=True)
            self.parameters_recovery_block = torch.tensor(parameters_recovery_block, requires_grad=True)

    def normalized_parameters(self, symmetry: float = 4 * np.pi):
        """Return recovery parameters reduced modulo a given symmetry.

        Args:
            symmetry (float): Period used to wrap the parameters. Defaults to
                ``4 * pi`` for U3-like gates.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Wrapped initial-layer and block
            recovery parameters.
        """

        with torch.no_grad():
            return torch.remainder(self.parameters_recovery_initial, symmetry), torch.remainder(self.parameters_recovery_block, symmetry)

    def parameters(self, grad: bool = True):
        """Return recovery parameters with optional gradient tracking.

        Args:
            grad (bool): If ``True``, return tensors that require gradients.
                If ``False``, return detached tensors. Defaults to ``True``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Initial-layer and block
            recovery parameter tensors.
        """

        if grad:
            return self.parameters_recovery_initial, self.parameters_recovery_block
        else:
            return self.parameters_recovery_initial.detach(), self.parameters_recovery_block.detach()


@dataclass
class TrainingProperties:
    """High-level training configuration and parameter initialization.

    This class decides which parts of the model are trained (encoding,
    operations, recovery), sets the number of epochs, instantiates a random
    number generator, and constructs parameter containers for encoding,
    operations, and recovery.
    """

    code_properties: CodeProperties
    seed_parameters: int
    learning_rate: float
    max_iter: int
    history_size: int
    epochs_encoding: int
    num_validation_states: None | int
    instance_validation_states: int
    num_test_states: None | int
    instance_test_states: int
    epochs_encoding_operation: Optional[int] = 0
    epochs_recovery: Optional[int] = 0
    encoding_loss: Optional[str] = 'avg'
    operation_loss: Optional[str] = 'diag'
    operation_loss_regularize: Optional[float] = 1.0
    parameters_encoding_trained: Optional[Any] = None
    parameters_recovery_trained: Optional[Any] = None
    parameters_operation_transversal_trained: Optional[dict[str, Any]] = None
    parameters_operation_weakly_transversal_trained: Optional[dict[str, Any]] = None
    parameters_operation_non_transversal_trained: Optional[dict[str, Any]] = None
    epochs: int = field(init=False)
    random_number_generator: np.random.Generator = field(init=False)
    parameters_encoding: ParametersEncoding = field(init=False)
    parameters_operation: ParametersOperation = field(init=False)
    parameters_recovery: ParametersRecovery = field(init=False)

    def __post_init__(self):
        self.epochs = self.epochs_encoding
        if self.code_properties.train_encoding:
            print('Training encoding ', end='')
            pre = ''
            if self.code_properties.num_static_operation > 0:
                print(f'with static {[o.name for o in self.code_properties.operation_static.values()]} operations ',
                      end='')
                pre = 'and '
                self.epochs = self.epochs_encoding_operation
            if self.code_properties.num_strictly_transversal_operation > 0:
                print(f'{pre}with strictly-transversal {[o.name for o in self.code_properties.
                      operation_strictly_transversal.values()]} operations ', end='')
                pre = 'and '
                self.epochs = self.epochs_encoding_operation
            if self.code_properties.num_transversal_operation > 0:
                print(f'{pre}with transversal {[o.name for o in self.code_properties.operation_transversal.values()]} '
                      f'operations ', end='')
                pre = 'and '
                self.epochs = self.epochs_encoding_operation
            if self.code_properties.num_weakly_transversal_operation > 0:
                print(f'{pre}with weakly-transversal {[o.name for o in self.code_properties.operation_weakly_transversal.values()]} '
                      f'operations ', end='')
                pre = 'and '
                self.epochs = self.epochs_encoding_operation
            if self.code_properties.num_non_transversal_operation > 0:
                print(f'{pre}with non-transversal {[o.name for o in self.code_properties.operation_non_transversal.values()]} '
                      f'operations ', end='')
                self.epochs = self.epochs_encoding_operation
            print(f'for {self.epochs} epochs (with up to {self.max_iter} internal loops per epoch).')
        else:
            if (0 == self.code_properties.num_transversal_operation + self.code_properties.num_weakly_transversal_operation
                    + self.code_properties.num_non_transversal_operation):
                print(f'Non-trainable encoding and no trainable operations, set number of '
                      f'epochs to 0, i.e. only testing (if `num_test_states` is larger than 0).')
                self.epochs = 0
            else:
                print('Training ', end='')
                pre = ''
                if self.code_properties.num_transversal_operation > 0:
                    print(f'transversal {[o.name for o in self.code_properties.operation_transversal.values()]} '
                          f'operations ', end='')
                    pre = 'and '
                    self.epochs = self.epochs_encoding_operation
                if self.code_properties.num_weakly_transversal_operation > 0:
                    print(f'{pre}weakly-transversal {[o.name for o in self.code_properties.operation_weakly_transversal.values()]} '
                          f'operations ', end='')
                    pre = 'and '
                    self.epochs = self.epochs_encoding_operation
                if self.code_properties.num_non_transversal_operation > 0:
                    print(f'{pre}non-transversal {[o.name for o in self.code_properties.operation_non_transversal.values()]} '
                          f'operations ', end='')
                    self.epochs = self.epochs_encoding_operation
                print(f'for {self.epochs} epochs (with up to {self.max_iter} internal loops per epoch).')

        if self.code_properties.recovery_properties is not None:
            if self.code_properties.recovery_properties.trainable:
                print(f'Post-training recovery operations for {self.epochs_recovery} epochs (with up to {self.max_iter} '
                      f'internal loops per epoch).')
            else:
                self.epochs_recovery = 0
                print(f'Non-trainable recovery, set number of training epochs to 0, i.e. only testing (if '
                      f'`num_test_states` is larger than 0).')
        print('-----')

        # instantiate random number generator
        self.random_number_generator = np.random.default_rng(self.seed_parameters)

        self.parameters_encoding = ParametersEncoding(
            parameter_shape=self.code_properties.encoding_properties.parameter_shape,
            random_number_generator=self.random_number_generator,
            parameters_encoding_trained=self.parameters_encoding_trained
        )

        self.parameters_operation = ParametersOperation(
            code_properties=self.code_properties,
            random_number_generator=self.random_number_generator,
            parameters_operation_transversal_trained=self.parameters_operation_transversal_trained,
            parameters_operation_weakly_transversal_trained=self.parameters_operation_weakly_transversal_trained,
            parameters_operation_non_transversal_trained=self.parameters_operation_non_transversal_trained
        )

        self.parameters_recovery = ParametersRecovery(
            parameter_shape=None if self.code_properties.recovery_properties is None else self.code_properties.recovery_properties.parameter_shape,
            random_number_generator=self.random_number_generator,
            parameters_recovery_trained=self.parameters_recovery_trained
        )

        if (not self.code_properties.train_encoding
                and not self.code_properties.train_recovery
                and 0 == self.code_properties.num_transversal_operation
                and 0 == self.code_properties.num_weakly_transversal_operation
                and 0 == self.code_properties.num_non_transversal_operation):
            print('No parameters to be initialized.')
