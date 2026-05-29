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

"""Utilities for loading QASM ansätze and custom PennyLane operations.

This module provides:

* mappings from short code/operation names to QASM filenames,
* a loader that parses QASM files and extracts wire metadata, and
* custom controlled-U3 (CU3) and controlled-S (CS) operations for use in
  variational ansätze.
"""

import os
from pathlib import Path
import pennylane as qml

ENCODINGS = ({
    'bitflip': 'bitflip_3_1_1',
    'approximate': 'approximate_4_1_2',
    'perfect': 'perfect_5_1_3',
    'steane': 'steane_7_1_3',
    'shor': 'shor_9_1_3',
    'shortdodeca': 'shortdodeca_10_1_4',
    'dodeca': 'dodeca_11_1_5',
    'reedmuller': 'reedmuller_15_1_3'})

RECOVERY = {
    'bitflip': 'bitflip_3_1_1',
    'perfect': 'perfect_5_1_3'
}

OPERATION = {
    'bitflip_X': 'bitflip_3_1_1_strictly_transversal_X',
    'bitflip_Z': 'bitflip_3_1_1_strictly_transversal_Z',
    'bitflip_S': 'bitflip_3_1_1_transversal_S',
    'bitflip_T': 'bitflip_3_1_1_transversal_T',
    'bitflip_CX': 'bitflip_3_1_1_strictly_transversal_CX',
    'bitflip_CZ': 'bitflip_3_1_1_strictly_transversal_CZ',
    'bitflip_CS': 'bitflip_3_1_1_transversal_CS',
    'approximate_X': 'approximate_4_1_2_transversal_X',
    'approximate_Z': 'approximate_4_1_2_transversal_Z',
    'approximate_CX': 'approximate_4_1_2_strictly_transversal_CX',
    'perfect_X': 'perfect_5_1_3_strictly_transversal_X',
    'perfect_Z': 'perfect_5_1_3_strictly_transversal_Z',
    'steane_X': 'steane_7_1_3_strictly_transversal_X',
    'steane_Z': 'steane_7_1_3_strictly_transversal_Z',
    'steane_H': 'steane_7_1_3_strictly_transversal_H',
    'steane_S': 'steane_7_1_3_transversal_S',
    'steane_CX': 'steane_7_1_3_strictly_transversal_CX',
    'steane_CZ': 'steane_7_1_3_strictly_transversal_CZ',
    'shor_X': 'shor_9_1_3_transversal_X',
    'shor_Z': 'shor_9_1_3_transversal_Z',
    'shor_CX': 'shor_9_1_3_transversal_CX',
    'shortdodeca_X': 'shortdodeca_10_1_4_transversal_X',
    'shortdodeca_Z': 'shortdodeca_10_1_4_transversal_Z',
    'dodeca_X': 'dodeca_11_1_5_transversal_X',
    'dodeca_Z': 'dodeca_11_1_5_transversal_Z',
    'reedmuller_X': 'reedmuller_15_1_3_strictly_transversal_X',
    'reedmuller_Z': 'reedmuller_15_1_3_strictly_transversal_Z',
    'reedmuller_S': 'reedmuller_15_1_3_transversal_S',
    'reedmuller_T': 'reedmuller_15_1_3_transversal_T'
}


def load_qasm_ansatz(encoding: str = None, recovery: str = None, operation: str = None):
    """Load a QASM ansatz for encoding, recovery, or a logical operation.

        Exactly one of ``encoding``, ``recovery``, or ``operation`` must be
        specified. The function resolves abbreviations to QASM filenames, reads
        the corresponding file from ``config/ansatz``, parses header comments to
        extract wire metadata, and returns the QASM body together with this
        metadata.

        The QASM files are expected to start with comment lines specifying the
        wire labels, for example:

        * Encoding:
          ``//d0,d1,...``
          ``//a0,a1,...``
        * Recovery:
          ``//d0,d1,...``
          ``//a0,a1,...``
          ``//r0,r1,...``
        * Operation:
          ``//d0,d1,...``
          ``//a0,a1,...``
          ``[//d_t0,d_t1,...]``
          ``[//a_t0,a_t1,...]``

        Args:
            encoding (str | None): Name or abbreviation of the encoding circuit.
                See ``ENCODINGS`` for available abbreviations. Mutually exclusive
                with ``recovery`` and ``operation``.
            recovery (str | None): Name or abbreviation of the recovery circuit.
                See ``RECOVERY`` for available abbreviations. Mutually exclusive
                with ``encoding`` and ``operation``.
            operation (str | None): Name of a logical operation circuit. Must be a
                key of :data:`OPERATION`. Mutually exclusive with ``encoding`` and
                ``recovery``.

        Returns:
            tuple:
                qasm_output (str): QASM string with header metadata lines removed.
                data_wires (list[str]): Labels of data wires.
                ancilla_wires (list[str]): Labels of ancilla wires.
                recovery_wires (list[str] | None): Labels of recovery wires, or
                    ``None`` if not applicable.
                data_target_wires (list[str]): Labels of data target wires for
                    two-qubit operations (empty if not set).
                ancilla_target_wires (list[str]): Labels of ancilla target wires
                    for two-qubit operations (empty if not set).
                gate (str | None): Operation identifier (e.g. ``"X"``, ``"CZ"``)
                    for logical operations; ``None`` for encoding/recovery
                    circuits.

        Raises:
            ValueError: If none or more than one of ``encoding``, ``recovery``,
                ``operation`` is specified, or if an unknown operation name is
                given.
            RuntimeError: If the QASM header does not follow the expected format.
        """

    if encoding is None and recovery is None and operation is None:
        raise ValueError('Either `encoding`, `recovery`, or `operation` must be specified`')
    if ((encoding is not None and recovery is not None) or (encoding is not None and operation is not None)
            or (recovery is not None and operation is not None)):
        raise ValueError('Only one of `encoding`, `recovery`, or `operation` can be specified`')

    # check if abbreviation is defined for encoding / recovery, otherwise just assume it is file name
    path = os.path.join(Path(__file__).resolve().parents[3], "config", "ansatz")
    if encoding is not None:
        ansatz_file = f'{encoding}.qasm' if ENCODINGS.get(encoding) is None else f'{ENCODINGS.get(encoding)}.qasm'
        gate = None
        path = os.path.join(path, 'encoding', ansatz_file)
        error_message = (f'The first lines of the QASM file needs to contain a comment specifying the code wires, '
                         f'i.e.: \n//d0,d1,...\n//a0,a1,...')
        print(f'Loading encoding circuit ({encoding}):')
    elif recovery is not None:
        ansatz_file = f'{recovery}.qasm' if RECOVERY.get(recovery) is None else f'{RECOVERY.get(recovery)}.qasm'
        gate = None
        path = os.path.join(path, 'recovery', ansatz_file)
        error_message = (f'The first lines of the QASM file needs to contain a comment specifying the code wires, '
                         f'i.e.: \n//d0,d1,...\n//a0,a1,...\n//r0,r1,...')
        print(f'Loading recovery circuit ({recovery}):')
    else:
        if OPERATION.get(operation) is None:
            raise ValueError(f'Operation `{operation}` not known.')
        ansatz_file = f'{OPERATION.get(operation)}.qasm'
        gate = OPERATION.get(operation).split('_')[-1]
        path = os.path.join(path, 'operation', ansatz_file)
        error_message = (f'The first lines of the QASM file needs to contain a comment specifying the code wires, '
                         f'i.e.: \n//d\n//a0,a1,...\n[//d_t]\n[//a0_t,a1_t,...]')
        print(f'Loading operation circuit ({operation}):')

    if not os.path.isfile(path):
        print(f'Ansatz file could not be found at {path}.')
    with open(path, 'r') as ff:
        qasm = ff.read()

    # check if first line is comment and contains metadata
    first_line = qasm.split('\n')[0]
    if first_line.startswith('//'):  # extract data qubits
        data_wires = first_line[2:].split(',')
    else:
        raise RuntimeError(error_message)
    second_line = qasm.split('\n')[1]
    if second_line.startswith('//'):  # extract ancilla wires
        ancilla_wires = second_line[2:].split(',')
    else:
        raise RuntimeError(error_message)
    print(f'Number of physical wires: n={len(data_wires) + len(ancilla_wires)}')
    print(f'Number of logical wires: k={len(data_wires)}')

    qasm_output = qasm
    if encoding:
        qasm_output = '\n'.join(qasm.split('\n')[2:])  # remove header lines

    # check if recovery wires are set (recovery also possible without designated recovery wires)
    recovery_wires = None
    if recovery:
        third_line = qasm.split('\n')[2]
        if third_line.startswith('//'):  # extract ancilla wires
            if 0 == len(third_line[2:]):  # no recovery wires
                recovery_wires = []
            else:
                recovery_wires = third_line[2:].split(',')
        else:
            raise RuntimeError(error_message)
        print(f'Number of recovery wires: r={len(recovery_wires)}')
        qasm_output = '\n'.join(qasm.split('\n')[3:])  # remove header lines

    # check if target wires are set, i.e. two-qubit operation
    data_target_wires, ancilla_target_wires = [], []
    if operation:
        third_line = qasm.split('\n')[2]
        if third_line.startswith('//'):
            data_target_wires = third_line[2:].split(',')
            fourth_line = qasm.split('\n')[3]  # if data target wires are provided, also target ancilla must be
            if fourth_line.startswith('//'):
                ancilla_target_wires = fourth_line[2:].split(',')
            else:
                raise RuntimeError(error_message)
            print('Target wires set, i.e. operation of order 2.')
            qasm_output = '\n'.join(qasm.split('\n')[4:])  # remove header lines
        else:
            qasm_output = '\n'.join(qasm.split('\n')[2:])  # remove header lines

    # load and display qaml circuit (explicit qubit mapping not required here, but done when actually applying circuit)
    print(qml.draw(qml.from_qasm3(qasm), level=0)())
    print()
    return qasm_output, data_wires, ancilla_wires, recovery_wires, data_target_wires, ancilla_target_wires, gate


class CU3(qml.operation.Operation):
    """Controlled-U3 two-qubit gate.

    This is a custom PennyLane operation representing a controlled version of
    the generic U3 rotation. It acts on two wires: a control and a target.

    The standard U3 gate is parameterized by three angles ``(phi, theta, omega)``.
    """

    num_wires = 2
    num_params = 3
    ndim_params = (0, 0, 0)
    grad_method = "A"
    name = "CU3"

    def __init__(self, phi, theta, omega, wires: list):
        """Initialize a CU3 operation.

        Args:
            phi (float): First U3 rotation angle.
            theta (float): Second U3 rotation angle.
            omega (float): Third U3 rotation angle.
            wires (list): List of two wire labels ``[control, target]`` on
                which the gate acts.
        """

        # initialize the parent class
        super().__init__(phi, theta, omega, wires=qml.wires.Wires(wires), id=f'CU3({phi},{theta},{omega})')

    @staticmethod
    def compute_decomposition(phi, theta, omega, wires):  # noqa
        """Decompose CU3 into native PennyLane operations.

        The CU3 gate is implemented as a controlled U3 rotation on the target
        wire, controlled by the first wire.

        Args:
            phi (float): First U3 rotation angle.
            theta (float): Second U3 rotation angle.
            omega (float): Third U3 rotation angle.
            wires (Sequence[Any]): Two-element sequence ``[control, target]``
                specifying the control and target wires.

        Returns:
            list[qml.operation.Operator]: List of PennyLane operations that
            implement the CU3 gate.
        """

        return [
            qml.ctrl(qml.U3(phi, theta, omega, wires=wires[1]), control=wires[0])
        ]


class CS(qml.operation.Operation):
    """Controlled-S two-qubit gate.

    This is a custom PennyLane operation representing a controlled-S phase
    gate acting on two wires: a control and a target.
    """

    num_wires = 2
    num_params = 0
    ndim_params = (0, )
    grad_method = "A"
    name = "CS"

    def __init__(self, wires: list):
        """Initialize a CS operation.

        Args:
            wires (list): List of two wire labels ``[control, target]`` on
                which the gate acts.
        """

        # initialize the parent class
        super().__init__(wires=qml.wires.Wires(wires), id=f'CS')

    @staticmethod
    def compute_decomposition(wires):  # noqa
        """Decompose CS into native PennyLane operations.

        The CS gate is implemented as a controlled-S gate on the target wire,
        controlled by the first wire.

        Args:
            wires (Sequence[Any]): Two-element sequence ``[control, target]``
                specifying the control and target wires.

        Returns:
            list[qml.operation.Operator]: List of PennyLane operations that
            implement the CS gate.
        """

        return [
            qml.ctrl(qml.S(wires=wires[1]), control=wires[0])
        ]
