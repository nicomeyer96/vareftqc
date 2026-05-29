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

"""Abstract base module and utilities for VarEFTQC circuit construction.

This module defines :class:`BaseModule`, an abstract base class for all
VarEFTQC circuit modules (physical model, encoding, recovery, operations).
It also contains helper methods to:

* apply variational or static QASM ansätze,
* apply noise channels according to :class:`NoiseProperties`, and
* export circuits to OpenQASM 3 format.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import pennylane as qml
import torch
import numpy as np

from .helpers.data_structures import NoiseProperties
from .components.error_channels import (BitFlipChannel, PhaseFlipChannel, AmplitudeDampingChannel, PhaseDampingChannel,
                                        AmplitudePhaseDampingChannel, UniformDepolarizingChannel, PauliChannel,
                                        AsymmetricDepolarizingChannel, ThermalRelaxationChannel)
from .components import RandomizedEntanglingAnsatz


class BaseModule(ABC):
    """Abstract base class for VarEFTQC circuit modules.

    Subclasses implement specific circuits (e.g. physical noise models,
    encoding, recovery, or logical operations) by providing their own
    :meth:`_circuit` method and user-facing :meth:`run` / :meth:`draw`
    wrappers.

    Attributes:
        wires (qml.wires.Wires): Wires on which the module acts.
        noise_properties (NoiseProperties | None): Noise model used to apply
            stochastic channels, or ``None`` for noiseless modules.
        device (qml.Device): PennyLane device on which the QNode is executed.
        qnode (qml.QNode): QNode wrapping the module's :meth:`_circuit`.
    """

    def __init__(self, wires: qml.wires.Wires, device_name: str, noise_properties: NoiseProperties = None):
        """Initialize a base module with a PennyLane device and QNode.

                Args:
                    wires (qml.wires.Wires): Wires on which this module's circuit
                        operates.
                    device_name (str): Name of the PennyLane device to use (e.g.
                        ``"default.qubit"`` or ``"default.mixed"`).
                    noise_properties (NoiseProperties | None): Optional noise model
                        used by :meth:`_apply_error_channel`. If ``None``, no noise
                        channels are applied.
                """

        self.wires = wires
        self.noise_properties = noise_properties

        self.device = qml.device(device_name, wires=wires)
        self.qnode = qml.QNode(self._circuit, self.device)

    @abstractmethod
    def run(self, *args, **kwargs) -> torch.Tensor:
        """Execute the module's circuit and return its outputs.

                Subclasses should provide a user-friendly signature and internally
                call :meth:`_run` with the appropriate keyword arguments.

                Returns:
                    torch.Tensor: Output of the underlying :class:`qml.QNode`,
                    typically a state, density matrix, or set of expectation values.
                """

        pass

    @abstractmethod
    def draw(self, *args, **kwargs):
        """Draw the module's circuit.

        Subclasses should expose relevant arguments (e.g. parameters, number
        of states) and internally call :meth:`_draw` with the appropriate
        keyword arguments.

        Returns:
            None
        """

        pass

    def _run(self, **kwargs):
        """Execute the underlying QNode with the given keyword arguments.

        Args:
            **kwargs: Keyword arguments forwarded to the :class:`qml.QNode`
                wrapping :meth:`_circuit`. Subclasses determine the expected
                keys.

        Returns:
            torch.Tensor: Output produced by the QNode.
        """

        return self.qnode(**kwargs)

    def _draw(self, level: int = 0, **kwargs):
        """Print an ASCII representation of the circuit.

        This uses :func:`qml.draw` on the module's QNode.

        Args:
            level (int): Detail level passed to :func:`qml.draw`. Defaults to
                ``0``.
            **kwargs: Keyword arguments forwarded to the QNode, typically
                including parameter objects.

        Returns:
            None
        """

        drawer = qml.draw(self.qnode, level=level, show_matrices=False)
        print(drawer(**kwargs))

    @abstractmethod
    def _circuit(self):
        """Define the quantum circuit for this module.

        This method is wrapped by :attr:`qnode` and must be implemented by
        subclasses. It should use PennyLane operations and return one or more
        measurements (e.g. states, expectation values, density matrices).
        """

        pass

    @staticmethod
    def _apply_ansatz(wires: qml.wires.Wires,
                      qasm: Tuple[str, list[str], list[str]] | Tuple[str, list[str], list[str], list[str] | Tuple[str, list[str], list[str], list[str], list[str]]] = None,
                      instance: int = None, connectivity: dict = None, gateset: Tuple = (None, None),
                      parameters_initial: torch.Tensor = None, parameters_block: torch.Tensor = None,
                      adjoint: bool = False):
        """Apply either a variational or static ansatz on the given wires.

        This utility supports two modes:

        * **Trainable mode** (``qasm is None``): a
          :class:`RandomizedEntanglingAnsatz` is applied using the provided
          parameters, gateset, instance index, and connectivity.
        * **Static mode** (``qasm is not None``): a QASM3 circuit is loaded
          via :func:`qml.from_qasm3` and applied with an explicit mapping
          from QASM wire labels to the given ``wires``.

        Args:
            wires (qml.wires.Wires): Wires on which to apply the ansatz.
            qasm (tuple | None): Optional QASM description and wire label
                lists. If ``None``, a trainable ansatz is used.
            instance (int | None): Instance index used to fix the layout of
                the randomized entangling ansatz.
            connectivity (dict | None): Optional connectivity specification
                for the variational ansatz.
            gateset (tuple): Tuple ``(gates_1q, gate_2q)`` specifying the
                parameterized 1-qubit and 2-qubit gates used in the
                variational ansatz.
            parameters_initial (torch.Tensor | None): Parameters for the
                initial layer of the variational ansatz (required in
                trainable mode).
            parameters_block (torch.Tensor | None): Parameters for the
                repeated block of the variational ansatz (required in
                trainable mode).
            adjoint (bool): If ``True``, apply the adjoint (inverse) of the
                ansatz. Defaults to ``False``.

        Raises:
            RuntimeError: If trainable mode is selected without providing both
                ``parameters_initial`` and ``parameters_block``, or if an
                unsupported QASM configuration is encountered.
        """

        if qasm is None:  # trainable parameters, use RandomizedEntanglingAnsatz
            if parameters_initial is None or parameters_block is None:
                raise RuntimeError('For trainable ansatz, `parameters_initial` and `parameters_block` '
                                   'must be specified.')
            if adjoint:
                qml.adjoint(
                    RandomizedEntanglingAnsatz(
                        parameters_initial=parameters_initial, parameters_block=parameters_block,
                        instance=instance, wires=wires, connectivity=connectivity, gates1q=gateset[0], gate2q=gateset[1]
                    )
                )
            else:
                RandomizedEntanglingAnsatz(
                    parameters_initial=parameters_initial, parameters_block=parameters_block,
                    instance=instance, wires=wires, connectivity=connectivity, gates1q=gateset[0], gate2q=gateset[1]
                )
        else:  # no trainable parameters
            # allocate qubits
            if 3 == len(qasm):  # data followed ancilla qubits
                wires_qasm = qasm[1] + qasm[2]
            elif 4 == len(qasm):  # data followed by ancilla followed by recovery qubits
                wires_qasm = qasm[1] + qasm[2] + qasm[3]
            elif 5 == len(qasm):  # data followed by ancilla followed by data_target followed by ancilla_target qubits
                wires_qasm = qasm[1] + qasm[2] + qasm[3] + qasm[4]
            else:
                raise RuntimeError('Configuration not supported.')
            if len(wires) != len(wires_qasm):
                raise RuntimeError(f'Inconsistent number of wires in QASM ansatz ({len(wires_qasm)}) and wires on '
                                   f'simulator ({len(wires)}).')
            wires_allocation = {wq: w for wq, w in zip(wires_qasm, wires)}
            if adjoint:
                qml.adjoint(qml.from_qasm3(qasm[0], wires_allocation))()
            else:
                qml.from_qasm3(qasm[0], wires_allocation)()

    def _apply_error_channel(self, wires: qml.wires.Wires):
        """Apply the configured noise channel(s) to the given wires.

        The specific channel depends on :attr:`noise_properties.noise` and
        associated parameters. Supported noise types include:

        * ``"dummy"`` (no noise),
        * ``"bitflip"``, ``"phaseflip"``,
        * ``"amplitude_damping"``, ``"phase_damping"``,
        * ``"amplitude_phase_damping"``,
        * ``"depolarizing"`` (uniform or asymmetric),
        * ``"pauli"`` (possibly with qubit-dependent weights), and
        * ``"thermal_relaxation"``.

        Args:
            wires (qml.wires.Wires): Wires on which to apply the noise
                channel.

        Raises:
            RuntimeError: If :attr:`noise_properties` is ``None``.
            ValueError: For invalid parameter combinations (e.g. missing
                ``noise_t1``/``noise_t2`` for thermal relaxation).
            NotImplementedError: If an unknown noise type is requested.
        """

        if self.noise_properties is None:
            raise RuntimeError('To apply the error channel `noise_properties` is required.')
        match self.noise_properties.noise:
            case 'dummy':
                pass
            case 'bitflip':
                BitFlipChannel(wires=wires, p=self.noise_properties.noise_strength)
            case 'phaseflip':
                PhaseFlipChannel(wires=wires, p=self.noise_properties.noise_strength)
            case 'amplitude_damping':
                AmplitudeDampingChannel(wires=wires, gamma=self.noise_properties.noise_strength)
            case 'phase_damping':
                PhaseDampingChannel(wires=wires, gamma=self.noise_properties.noise_strength)
            case 'amplitude_phase_damping':
                AmplitudePhaseDampingChannel(wires=wires, gamma=self.noise_properties.noise_strength)
            case 'depolarizing':
                if 1.0 == self.noise_properties.noise_asymmetry:
                    UniformDepolarizingChannel(wires=wires, p=self.noise_properties.noise_strength)
                else:
                    AsymmetricDepolarizingChannel(wires=wires, p=self.noise_properties.noise_strength,
                                                  c=self.noise_properties.noise_asymmetry)
            case 'pauli':
                if isinstance(self.noise_properties.noise_pauli_x, float) and isinstance(self.noise_properties.noise_pauli_z, float):
                    # determine Y-noise
                    noise_pauli_y = 1 - self.noise_properties.noise_pauli_x - self.noise_properties.noise_pauli_z
                    # if p_x is close to 1, then we can model it as bitflip noise
                    if np.isclose(self.noise_properties.noise_pauli_x, 1.0, atol=0.0005):
                        BitFlipChannel(wires=wires, p=self.noise_properties.noise_strength)
                    # if p_z is close to 1, then we can model it as phaseflip noise
                    elif np.isclose(self.noise_properties.noise_pauli_z, 1.0, atol=0.0005):
                        PhaseFlipChannel(wires=wires, p=self.noise_properties.noise_strength)
                    # if p_x and p_z both are close to 1/3, then also p_y is close to 1/3 and the noise is uniform
                    elif (np.isclose(self.noise_properties.noise_pauli_x, 1/3, atol=0.0005)
                            and np.isclose(self.noise_properties.noise_pauli_z, 1/3, atol=0.0005)):
                        UniformDepolarizingChannel(wires=wires, p=self.noise_properties.noise_strength)
                    # if p_x is close to p_y (1 - p_x - p_z), then we can model it as asymmetric depolarizing noise
                    elif np.isclose(self.noise_properties.noise_pauli_x, noise_pauli_y, atol=0.0005):
                        # compute asymmetry factor c
                        noise_asymmetry = (np.log(self.noise_properties.noise_strength * self.noise_properties.noise_pauli_z + qml.math.eps)
                                           / np.log(self.noise_properties.noise_strength * noise_pauli_y + qml.math.eps))
                        AsymmetricDepolarizingChannel(wires=wires, p=self.noise_properties.noise_strength,
                                                      c=noise_asymmetry)
                    else:
                        PauliChannel(wires=wires, p=self.noise_properties.noise_strength,
                                     px=self.noise_properties.noise_pauli_x, pz=self.noise_properties.noise_pauli_z)
                else:  # different Pauli-weights for different qubits
                    PauliChannel(wires=wires, p=self.noise_properties.noise_strength,
                                 px=self.noise_properties.noise_pauli_x, pz=self.noise_properties.noise_pauli_z)
            case 'thermal_relaxation':
                if self.noise_properties.noise_t1 is None or self.noise_properties.noise_t2 is None:
                    raise ValueError('For `thermal_relaxation` noise `noise_t1` and `noise_t2` [in ms] are required.')
                ThermalRelaxationChannel(wires=wires, idle=self.noise_properties.noise_strength,
                                         t1=self.noise_properties.noise_t1, t2=self.noise_properties.noise_t2)
            case _:
                raise NotImplementedError(f'Noise type {self.noise_properties.noise} is not known.')

    def _get_qasm_ansatz(self, wires_data: qml.wires.Wires, wires_ancilla: qml.wires.Wires,
                         wires_recovery: qml.wires.Wires = None,
                         wires_data_target: qml.wires.Wires = None, wires_ancilla_target: qml.wires.Wires = None,
                         qasm: Tuple[str, list[str], list[str]] | Tuple[str, list[str], list[str], list[str]] = None,
                         instance: int = None, connectivity: dict = None, gateset: Tuple = (None, None),
                         parameters_initial: torch.Tensor = None, parameters_block: torch.Tensor = None,
                         only_header: bool = False):
        """Return an OpenQASM 3 representation of an ansatz with metadata.

        This constructs a QASM3 string consisting of:

        * a header with wire metadata (data, ancilla, optional recovery and
          target wires), and
        * the body of a circuit that either:
          - replays a static QASM ansatz, or
          - applies a variational ansatz using the provided parameters.

        The circuit is built inside a temporary QNode and then serialized
        using :meth:`_tape_to_openqasm3_simplified`.

        Args:
            wires_data (qml.wires.Wires): Data wires of the code.
            wires_ancilla (qml.wires.Wires): Ancilla wires of the code.
            wires_recovery (qml.wires.Wires | None): Optional recovery wires.
            wires_data_target (qml.wires.Wires | None): Optional data target
                wires for logical operations.
            wires_ancilla_target (qml.wires.Wires | None): Optional ancilla
                target wires for logical operations.
            qasm (tuple | None): Static QASM ansatz and wire labels. If
                ``None``, a variational ansatz is used.
            instance (int | None): Instance index used for variational ansätze.
            connectivity (dict | None): Optional connectivity for the
                variational ansatz.
            gateset (tuple): Tuple ``(gates_1q, gate_2q)`` specifying the
                parameterized gates for the variational ansatz.
            parameters_initial (torch.Tensor | None): Initial-layer parameters
                for the variational ansatz.
            parameters_block (torch.Tensor | None): Block parameters for the
                variational ansatz.
            only_header (bool): If ``True``, return only the header with wire
                metadata and skip the circuit body.

        Returns:
            str: QASM3 string containing the header and circuit body.

        Raises:
            RuntimeError: If variational mode is selected without providing
                both ``parameters_initial`` and ``parameters_block``, or if an
                invalid wire configuration is given.
        """

        # compose metadata string
        qasm_output = ''
        qasm_output += f'//{','.join(wires_data)}\n'
        qasm_output += f'//{','.join(wires_ancilla)}\n'
        if wires_recovery is not None:
            if 0 == len(wires_recovery):
                qasm_output += f'//\n'
            else:
                qasm_output += f'//{','.join(wires_recovery)}\n'
        if (wires_data_target is not None and len(wires_data_target) > 0
                and wires_ancilla_target is not None and len(wires_ancilla_target) > 0):
            qasm_output += f'//{','.join(wires_data_target)}\n'
            qasm_output += f'//{','.join(wires_ancilla_target)}\n'

        # dummy recovery wires if working with encoding
        if wires_recovery is None and wires_data_target is None and wires_ancilla_target is None:
            wires = wires_data + wires_ancilla
        elif wires_recovery is not None and wires_data_target is None and wires_ancilla_target is None:
            wires = wires_data + wires_ancilla + wires_recovery
        elif wires_recovery is None and wires_data_target is not None and wires_ancilla_target is not None:
            wires = wires_data + wires_ancilla + wires_data_target + wires_ancilla_target
        else:
            raise RuntimeError('Invalid wire configuration.')

        if only_header:
            return qasm_output

        if qasm is None:
            if parameters_initial is None or parameters_block is None:
                raise RuntimeError('For trainable ansatz, `parameters_initial` and `parameters_block` '
                                   'must be specified.')

        # construction for converting to QASM
        @qml.qnode(qml.device('default.qubit', wires=wires))
        def circuit():
            self._apply_ansatz(wires=wires, qasm=qasm,
                               instance=instance, connectivity=connectivity, gateset=gateset,
                               parameters_initial=parameters_initial,
                               parameters_block=parameters_block)
            return qml.state()  # required for conversion, will be removed again later

        circuit()  # run once to create tape
        # qasm = circuit.tape.to_openqasm(wires=self.wires, measure_all=False,  precision=8)  # convert to qasm syntax
        qasm_output += self._tape_to_openqasm3_simplified(circuit._tape, wires=wires, precision=8)  # noqa
        return qasm_output

    @staticmethod
    def _tape_to_openqasm3_simplified(tape: qml.tape.QuantumScript, wires: qml.wires, precision: int = 8):
        """Convert a PennyLane tape to a simplified OpenQASM 3 program.

        This is a modified version of ``qml.tape.to_openqasm`` that:

        * targets OpenQASM 3 syntax,
        * supports a subset of single- and multi-qubit gates defined in the
          local ``gates`` mapping, and
        * declares one named qubit per wire in ``wires``.

        Args:
            tape (qml.tape.QuantumScript): Quantum tape to serialize.
            wires (qml.wires.Wires | Sequence[Any]): Wires to declare in the
                QASM program and to use when printing operations.
            precision (int): Number of decimal places for gate parameters.
                If ``None``, use Python's default string representation.

        Returns:
            str: OpenQASM 3 program string representing the circuit.

        Raises:
            ValueError: If the tape contains an operation that is not present
                in the supported gate mapping.
        """

        gates = {
            # supported single-qubit gates
            "U3": "u3", "U2": "u2", "U1": "u1",
            "Identity": "id", "PauliX": "x", "PauliY": "y", "PauliZ": "z",
            "Hadamard": "h", "S": "s", "Adjoint(S)": "sdg", "T": "t", "Adjoint(T)": "tdg",
            "RX": "rx", "RY": "ry", "RZ": "rz",
            "PhaseShift": "u1",
            # natively supported two-qubit gates
            "CNOT": "cx", "CZ": "cz", "SWAP": "swap",
            # natively supported three-qubit gate
            "Toffoli": "ccx", "CSWAP": "cswap",
            # controlled gates with QASM3 syntax
            "C(U3)": 'ctrl @ u3'
        }

        # add the QASM headers
        qasm_str = 'OPENQASM 3;\n'

        # create the qubits
        for wire in wires:
            qasm_str += f"qubit {wire};\n"

        # get the user applied circuit operations without interface information
        [transformed_tape], _ = qml.transforms.convert_to_numpy_parameters(tape)
        operations = transformed_tape.expand(depth=10, stop_at=lambda obj: obj.name in gates).operations

        for op in operations:
            try:
                gate = gates[op.name]
            except KeyError as e:
                raise ValueError(f"Operation {op.name} not supported by the QASM serializer") from e

            wire_labels = ",".join([f"{w}" for w in op.wires.tolist()])
            params = ""

            if op.num_params > 0:
                # If the operation takes parameters, construct a string
                # with parameter values.
                if precision is not None:
                    params = "(" + ",".join([f"{p:.{precision}f}" for p in op.parameters]) + ")"
                else:
                    # use default precision
                    params = "(" + ",".join([str(p) for p in op.parameters]) + ")"

            qasm_str += f"{gate}{params} {wire_labels};\n"
        return qasm_str
