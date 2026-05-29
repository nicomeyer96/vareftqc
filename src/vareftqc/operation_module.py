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

"""Operation modules for VarEFTQC.

This module defines:

* :class:`OperationBaseModule`: common infrastructure for encoded logical
  operations,
* :class:`OperationTargetModule`: the ideal target implementation of a
  logical operation, and
* :class:`OperationPredictionModule`: the encoded implementation (static,
  transversal, weakly transversal, or non-transversal), with QASM export.
"""

import pennylane as qml
import torch
from abc import abstractmethod

from .helpers.data_structures import (OperationProperties, ParametersEncoding, CodeProperties,
                                           ParametersTransversalOperation, ParametersWeaklyTransversalOperation,
                                           ParametersNonTransversalOperation)
from .components.state_preparation import SphericalTwoDesign, HaarRandomState
from .components.operations import (TargetOperation, EncodedOperationStrictlyTransversal, EncodedOperationTransversal,
                                    EncodedOperationWeaklyTransversal)
from .base_module import BaseModule


class OperationBaseModule(BaseModule):
    """Base class for encoded logical-operation modules.

    This class sets up the wire layout for logical operations acting on a
    code encoded by :class:`EncodingModule`. It arranges:

    * data wires and ancilla wires of the code, and
    * optional target wires (for two-qubit operations) labelled with ``"_t"``.

    Subclasses implement concrete circuits for target or prediction
    operations.
    """

    def __init__(self, code_properties: CodeProperties, operation_properties: OperationProperties):
        """Initialize an encoded logical-operation module.

                Args:
                    code_properties (CodeProperties): Code configuration including
                        encoding properties and wire layout.
                    operation_properties (OperationProperties): Logical operation
                        specification (order, gate, gateset, transversal flags, etc.).
                """

        self.code_properties = code_properties
        self.operation_properties = operation_properties
        self.wires_data, self.wires_ancilla, self.wires_data_target, self.wires_ancilla_target = self._compose_wires()
        self.wires = self.wires_data + self.wires_ancilla + self.wires_data_target + self.wires_ancilla_target

        super().__init__(wires=self.wires_data+self.wires_ancilla+self.wires_data_target+self.wires_ancilla_target,
                         device_name='default.qubit')

    def run(self, parameters_encoding: ParametersEncoding,
            parameters_operation: ParametersTransversalOperation | ParametersWeaklyTransversalOperation | ParametersNonTransversalOperation = None,
            number_states: int = 0, seed_states: int = None):
        """Execute the encoded operation circuit and return the output state.

        Depending on the type of ``parameters_operation``, this function
        extracts the appropriate initial/block parameter tensors and forwards
        them to the underlying QNode.

        Args:
            parameters_encoding (ParametersEncoding): Encoding parameters used
                in the encoded operation circuit.
            parameters_operation (ParametersTransversalOperation | ParametersWeaklyTransversalOperation | ParametersNonTransversalOperation | None):
                Operation parameters, whose exact type depends on whether the
                operation is transversal, weakly transversal, or
                non-transversal. For static/strictly-transversal operations,
                this can be ``None``.
            number_states (int): Number of Haar-random input states to
                prepare on data + target wires. If ``0``, a spherical
                two-design is used instead.
            seed_states (int | None): Random seed/instance index for Haar
                state sampling.

        Returns:
            torch.Tensor: Statevector of all wires (data, ancilla, target)
            as returned by :func:`qml.state`.
        """

        parameters_operation_initial, parameters_operation_block = None, None
        if isinstance(parameters_operation, ParametersTransversalOperation):
            parameters_operation_initial = parameters_operation.parameters_operation
        if isinstance(parameters_operation, ParametersWeaklyTransversalOperation):
            parameters_operation_initial = parameters_operation.parameters_operation
        if isinstance(parameters_operation, ParametersNonTransversalOperation):
            parameters_operation_initial = parameters_operation.parameters_operation_initial
            parameters_operation_block = parameters_operation.parameters_operation_block
        return self._run(parameters_encoding_initial=parameters_encoding.parameters_encoding_initial,
                         parameters_encoding_block=parameters_encoding.parameters_encoding_block,
                         parameters_operation_initial=parameters_operation_initial,
                         parameters_operation_block=parameters_operation_block,
                         number_states=number_states, seed_states=seed_states)

    def draw(self, parameters_encoding: ParametersEncoding,
             parameters_operation: ParametersTransversalOperation | ParametersWeaklyTransversalOperation | ParametersNonTransversalOperation = None,
             number_states: int = 0, seed_states: int = None, level: int = 0):
        """Print an ASCII diagram of the encoded operation circuit.

        Args:
            parameters_encoding (ParametersEncoding): Encoding parameters.
            parameters_operation (ParametersTransversalOperation | ParametersWeaklyTransversalOperation | ParametersNonTransversalOperation | None):
                Operation parameters, matching the operation type.
            number_states (int): Number of Haar-random input states to
                prepare. If ``0``, a spherical two-design is used instead.
            seed_states (int | None): Random seed/instance index for Haar
                state sampling.
            level (int): Detail level passed to :func:`qml.draw`. Defaults to
                ``0``.

        Returns:
            None
        """

        parameters_operation_initial, parameters_operation_block = None, None
        if isinstance(parameters_operation, ParametersTransversalOperation):
            parameters_operation_initial = parameters_operation.parameters_operation
        if isinstance(parameters_operation, ParametersWeaklyTransversalOperation):
            parameters_operation_initial = parameters_operation.parameters_operation
        if isinstance(parameters_operation, ParametersNonTransversalOperation):
            parameters_operation_initial = parameters_operation.parameters_operation_initial
            parameters_operation_block = parameters_operation.parameters_operation_block
        self._draw(parameters_encoding_initial=parameters_encoding.parameters_encoding_initial,
                   parameters_encoding_block=parameters_encoding.parameters_encoding_block,
                   parameters_operation_initial=parameters_operation_initial,
                   parameters_operation_block=parameters_operation_block,
                   number_states=number_states, seed_states=seed_states, level=level)

    def _compose_wires(self):
        """Compose and validate wires for encoded logical operations.

        This method:

        * checks that there is at least one ancilla wire, and
        * for 2-qubit operations, creates matching target wires for each
          data and ancilla wire with ``"_t"`` suffix.

        Returns:
            tuple[qml.wires.Wires, qml.wires.Wires, qml.wires.Wires, qml.wires.Wires]:
            ``(wires_data, wires_ancilla, wires_data_target, wires_ancilla_target)``.

        Raises:
            ValueError: If fewer than one ancilla wire is specified.
            NotImplementedError: If the operation order is greater than 2.
        """

        if self.code_properties.num_wires_ancilla < 1:
            raise ValueError(f'At least one ancilla wire is required ({self.code_properties.num_wires_ancilla} '
                             f'selected).')
        wires_data, wires_ancilla = self.code_properties.wires_data, self.code_properties.wires_ancilla
        if 1 == self.operation_properties.order:
            wires_data_target, wires_ancilla_target = qml.wires.Wires([]), qml.wires.Wires([])
        elif 2 == self.operation_properties.order:
            wires_data_target, wires_ancilla_target = (qml.wires.Wires([f'{d}_t' for d in wires_data]),
                                                       qml.wires.Wires([f'{a}_t' for a in wires_ancilla]))
        else:
            raise NotImplementedError('Higher than second order operations are currently not supported.')
        return wires_data, wires_ancilla, wires_data_target, wires_ancilla_target

    @abstractmethod
    def _circuit(self, parameters_encoding_initial: torch.Tensor = None, parameters_encoding_block: torch.Tensor = None,
                 parameters_operation_initial: torch.Tensor = None, parameters_operation_block: torch.Tensor = None,
                 number_states: int = 0, seed_states: int = None):
        """Define the encoded operation circuit.

        Subclasses implement this method to realize either:

        * the ideal target operation (for :class:`OperationTargetModule`), or
        * the encoded logical operation (for :class:`OperationPredictionModule`).

        Returns:
            qml.measurements.MeasurementProcess: Typically a statevector
            measurement via :func:`qml.state`.
        """

        pass


class OperationTargetModule(OperationBaseModule):
    """Target implementation of a logical operation in the code space.

    This module applies:

    * the ideal logical operation (e.g. X, CZ) directly on data and target
      wires, and
    * the encoding ansatz on both data+ancilla and, for two-qubit operations,
      target+ancilla_target wires,

    and returns the resulting statevector. It serves as the reference target
    when training encoded logical operations.
    """

    def __init__(self, code_properties: CodeProperties, operation_properties: OperationProperties):
        super().__init__(code_properties=code_properties, operation_properties=operation_properties)

    def _circuit(self, parameters_encoding_initial: torch.Tensor = None, parameters_encoding_block: torch.Tensor = None,
                 parameters_operation_initial: torch.Tensor = None, parameters_operation_block: torch.Tensor = None,
                 number_states: int = 0, seed_states: int = None):
        """Define the target logical-operation circuit.

        The circuit:

        * prepares two-design or Haar-random states on data + data_target
          wires,
        * applies the ideal logical operation between data and target wires,
        * applies the encoding ansatz on data+ancilla (and, for two-qubit
          operations, on target+ancilla_target), and
        * returns the statevector of all wires.

        Args:
            parameters_encoding_initial (torch.Tensor | None): Initial-layer
                encoding parameters.
            parameters_encoding_block (torch.Tensor | None): Block encoding
                parameters.
            parameters_operation_initial (torch.Tensor | None): Unused here
                (target operation is non-parameterized).
            parameters_operation_block (torch.Tensor | None): Unused here.
            number_states (int): Number of Haar-random input states to
                prepare. If ``0``, a spherical two-design is used instead.
            seed_states (int | None): Random seed/instance index for Haar
                state sampling.

        Returns:
            qml.measurements.MeasurementProcess: Statevector measurement via
            :func:`qml.state`.
        """

        # initialize state
        if 0 == number_states:  # use spherical 2-design
            SphericalTwoDesign(wires=self.wires_data + self.wires_data_target)
        else:
            HaarRandomState(wires=self.wires_data + self.wires_data_target, number=number_states, seed=seed_states)
        qml.Barrier(wires=self.wires, only_visual=False)

        if self.code_properties.encoding_properties.connectivity is not None:
            raise NotImplementedError('Setting `connectivity` is currently not supported for training operation module.')

        ###########################
        # place target operation

        TargetOperation(wire_control=self.wires_data, wire_target=self.wires_data_target,
                        operation=self.operation_properties.gate, operation_name=self.operation_properties.name)
        qml.Barrier(wires=self.wires, only_visual=False)

        #################################################
        # apply encoding (either fixed or parameterized)

        self._apply_ansatz(wires=self.wires_data + self.wires_ancilla,
                           qasm=self.code_properties.encoding_properties.qasm,
                           instance=self.code_properties.encoding_properties.instance,
                           gateset=self.code_properties.encoding_properties.gateset,
                           parameters_initial=parameters_encoding_initial,
                           parameters_block=parameters_encoding_block)
        if self.operation_properties.order > 1:  # optionally apply same ansatz on target wires
            self._apply_ansatz(wires=self.wires_data_target + self.wires_ancilla_target,
                               qasm=self.code_properties.encoding_properties.qasm,
                               instance=self.code_properties.encoding_properties.instance,
                               gateset=self.code_properties.encoding_properties.gateset,
                               parameters_initial=parameters_encoding_initial,
                               parameters_block=parameters_encoding_block)
        # return state of all qubits for computing fidelity (statevector representation for efficiency)
        return qml.state()


class OperationPredictionModule(OperationBaseModule):
    """Encoded implementation of a logical operation.

    This module applies:

    * the encoding ansatz on data+ancilla (and optionally on
      target+ancilla_target), and
    * the encoded logical operation, which can be:
      - static from QASM,
      - strictly transversal,
      - trainable transversal,
      - trainable weakly transversal, or
      - trainable non-transversal,

    and returns the resulting statevector. It also provides methods to export
    the encoded operation to OpenQASM 3.
    """

    def __init__(self, code_properties: CodeProperties, operation_properties: OperationProperties):
        super().__init__(code_properties=code_properties, operation_properties=operation_properties)

    def get_qasm_operation(self, parameters_operation: ParametersTransversalOperation | ParametersWeaklyTransversalOperation | ParametersNonTransversalOperation = None,  # noqa
                           normalize: bool = True):
        """Return an OpenQASM 3 representation of the encoded operation.

        Depending on :attr:`operation_properties`, this covers:

        * static operations (using stored QASM),
        * strictly transversal operations,
        * trainable transversal operations (requiring
          :class:`ParametersTransversalOperation`),
        * trainable weakly-transversal operations (requiring
          :class:`ParametersWeaklyTransversalOperation`), and
        * trainable non-transversal operations (requiring
          :class:`ParametersNonTransversalOperation`).

        Args:
            parameters_operation (ParametersTransversalOperation | ParametersWeaklyTransversalOperation | ParametersNonTransversalOperation | None):
                Parameter container matching the operation type. Required for
                trainable operations, ignored for static/strictly-transversal
                operations.
            normalize (bool): If ``True``, use normalized parameters (wrapped
                modulo a symmetry, e.g. ``4π``). If ``False``, use raw
                parameters without gradient tracking.

        Returns:
            str: QASM3 string describing the encoded operation circuit,
            including a metadata header and the operation body.

        Raises:
            ValueError: If a parameter container of the wrong type is
                provided for the selected operation type.
        """

        if self.operation_properties.qasm is not None:
            qasm_header = self._get_qasm_ansatz(
                wires_data=self.wires_data, wires_ancilla=self.wires_ancilla,
                wires_data_target=self.wires_data_target, wires_ancilla_target=self.wires_ancilla_target,
                instance=self.operation_properties.instance,
                only_header=True
            )
            qasm_output = (qasm_header + self.operation_properties.qasm[0])
            return qasm_output

        if (self.operation_properties.strictly_transversal or self.operation_properties.transversal
                or self.operation_properties.weakly_transversal):
            qasm_header = self._get_qasm_ansatz(
                wires_data=self.wires_data, wires_ancilla=self.wires_ancilla,
                wires_data_target=self.wires_data_target, wires_ancilla_target=self.wires_ancilla_target,
                instance=self.operation_properties.instance,
                only_header=True
            )

            if self.operation_properties.strictly_transversal:
                qasm_output = (qasm_header +
                               self._get_qasm_strictly_transversal_operation(
                                   wires=self.wires_data + self.wires_ancilla + self.wires_data_target + self.wires_ancilla_target))  # noqa
            elif self.operation_properties.transversal:
                if not isinstance(parameters_operation, ParametersTransversalOperation):
                    raise ValueError('For transversal operations a `ParametersTransversalOperation` object must be '
                                     'provided.`')
                if normalize:
                    parameters_operation_initial = parameters_operation.normalized_parameters()
                else:
                    parameters_operation_initial = parameters_operation.parameters(grad=False)
                qasm_output = (qasm_header +
                               self._get_qasm_transversal_operation(
                                   wires=self.wires_data + self.wires_ancilla + self.wires_data_target + self.wires_ancilla_target,  # noqa
                                   parameters_operation_initial=parameters_operation_initial))
            else:
                if not isinstance(parameters_operation, ParametersWeaklyTransversalOperation):
                    raise ValueError('For weakly-transversal operations a `ParametersWeaklyTransversalOperation` '
                                     'object must be provided.`')
                if normalize:
                    parameters_operation_initial = parameters_operation.normalized_parameters()
                else:
                    parameters_operation_initial = parameters_operation.parameters(grad=False)
                qasm_output = (qasm_header +
                               self._get_qasm_weakly_transversal_operation(
                                   wires=self.wires_data + self.wires_ancilla + self.wires_data_target + self.wires_ancilla_target,  # noqa
                                   parameters_operation_initial=parameters_operation_initial))
            return qasm_output

        # non-transversal case, i.e. using RandomizedEntanglingAnsatz
        if not isinstance(parameters_operation, ParametersNonTransversalOperation):
            raise ValueError('For non-transversal operations a `ParametersNonTransversalOperation` object must be '
                             'provided.`')
        if normalize:
            parameters_operation_initial, parameters_operation_block = parameters_operation.normalized_parameters()
        else:
            parameters_operation_initial, parameters_operation_block = parameters_operation.parameters(grad=False)

        return self._get_qasm_ansatz(
            wires_data=self.wires_data, wires_ancilla=self.wires_ancilla,
            wires_data_target=self.wires_data_target, wires_ancilla_target=self.wires_ancilla_target,
            instance=self.operation_properties.instance,
            gateset=([self.operation_properties.gateset[0]], self.operation_properties.gateset[1]),
            parameters_initial=parameters_operation_initial, parameters_block=parameters_operation_block
        )

    def _circuit(self, parameters_encoding_initial: torch.Tensor = None, parameters_encoding_block: torch.Tensor = None,
                 parameters_operation_initial: torch.Tensor = None, parameters_operation_block: torch.Tensor = None,
                 number_states: int = 0, seed_states: int = None):
        """Define the encoded logical-operation circuit.

        The circuit:

        * prepares two-design or Haar-random states on data + data_target
          wires,
        * applies the encoding ansatz on data+ancilla (and, for two-qubit
          operations, on target+ancilla_target), and
        * applies the encoded logical operation, which can be static,
          strictly transversal, transversal, weakly transversal, or
          non-transversal depending on :attr:`operation_properties`.

        Args:
            parameters_encoding_initial (torch.Tensor | None): Initial-layer
                encoding parameters.
            parameters_encoding_block (torch.Tensor | None): Block encoding
                parameters.
            parameters_operation_initial (torch.Tensor | None): Operation
                parameters for transversal/weakly-transversal/non-transversal
                operations.
            parameters_operation_block (torch.Tensor | None): Block operation
                parameters for non-transversal operations.
            number_states (int): Number of Haar-random input states to
                prepare. If ``0``, a spherical two-design is used instead.
            seed_states (int | None): Random seed/instance index for Haar
                state sampling.

        Returns:
            qml.measurements.MeasurementProcess: Statevector measurement via
            :func:`qml.state`.

        Raises:
            RuntimeError: If parameters are missing or inconsistent for the
                selected operation type.
        """

        # initialize state
        if 0 == number_states:  # use spherical 2-design
            SphericalTwoDesign(wires=self.wires_data + self.wires_data_target)
        else:
            HaarRandomState(wires=self.wires_data + self.wires_data_target, number=number_states, seed=seed_states)
        qml.Barrier(wires=self.wires, only_visual=False)

        if self.code_properties.encoding_properties.connectivity is not None:
            raise NotImplementedError('Setting `connectivity` is currently not supported for training operation module.')

        #################################################
        # apply encoding (either fixed or parameterized)

        self._apply_ansatz(wires=self.wires_data + self.wires_ancilla,
                           qasm=self.code_properties.encoding_properties.qasm,
                           instance=self.code_properties.encoding_properties.instance,
                           gateset=self.code_properties.encoding_properties.gateset,
                           parameters_initial=parameters_encoding_initial,
                           parameters_block=parameters_encoding_block)
        if self.operation_properties.order > 1:  # optionally apply same ansatz on target wires
            self._apply_ansatz(wires=self.wires_data_target + self.wires_ancilla_target,
                               qasm=self.code_properties.encoding_properties.qasm,
                               instance=self.code_properties.encoding_properties.instance,
                               gateset=self.code_properties.encoding_properties.gateset,
                               parameters_initial=parameters_encoding_initial,
                               parameters_block=parameters_encoding_block)
        qml.Barrier(wires=self.wires, only_visual=False)

        ######################################################################################
        # place encoded operation (either fixed or parameterized)

        if self.operation_properties.qasm is not None:  # static (non-parameterized)
            if parameters_operation_initial is not None or parameters_operation_block is not None:
                raise RuntimeError('No trainable parameters can be provided for static operations.')
            self._apply_ansatz(
                wires=self.wires_data + self.wires_ancilla + self.wires_data_target + self.wires_ancilla_target,
                qasm=self.operation_properties.qasm)  # noqa
        elif self.operation_properties.strictly_transversal:  # strictly-transversal (non-parameterized)
            if parameters_operation_initial is not None or parameters_operation_block is not None:
                raise RuntimeError('No trainable parameters can be provided for strictly transversal operations.')
            EncodedOperationStrictlyTransversal(operation=self.operation_properties.gate,
                                                wires_control=self.wires_data+self.wires_ancilla,
                                                wires_target=self.wires_data_target+self.wires_ancilla_target,
                                                operation_name=self.operation_properties.name)
        elif self.operation_properties.transversal:  # transversal (parameterized)
            if parameters_operation_initial is None:
                raise RuntimeError('Parameters `parameters_operation_initial` cannot be None.')
            if parameters_operation_block is not None:
                raise RuntimeError('Parameters `parameters_operation_block` cannot be provided for transversal '
                                   'operations.')
            EncodedOperationTransversal(parameters=parameters_operation_initial,
                                        order_operation=self.operation_properties.order,
                                        wires_control=self.wires_data+self.wires_ancilla,
                                        wires_target=self.wires_data_target+self.wires_ancilla_target,
                                        gate1q=self.operation_properties.gateset[0],
                                        gate2q=self.operation_properties.gateset[1],
                                        flip=self.operation_properties.flip)
        elif self.operation_properties.weakly_transversal:  # weakly-transversal (parameterized)
            if parameters_operation_initial is None:
                raise RuntimeError('Parameters `parameters_operation_initial` cannot be None.')
            if parameters_operation_block is not None:
                raise RuntimeError('Parameters `parameters_operation_block` cannot be provided for weakly-transversal '
                                   'operations.')
            EncodedOperationWeaklyTransversal(parameters=parameters_operation_initial,
                                              order_operation=self.operation_properties.order,
                                              wires_control=self.wires_data+self.wires_ancilla,
                                              wires_target=self.wires_data_target+self.wires_ancilla_target)
        else:
            if parameters_operation_initial is None or parameters_operation_block is None:
                raise RuntimeError('Parameters `parameters_operation_initial` and `parameters_operation_block` cannot '
                                   'be None.')
            self._apply_ansatz(wires=self.wires_data + self.wires_ancilla + self.wires_data_target + self.wires_ancilla_target,
                               instance=self.operation_properties.instance,
                               gateset=([self.operation_properties.gateset[0]], self.operation_properties.gateset[1]),
                               parameters_initial=parameters_operation_initial,
                               parameters_block=parameters_operation_block)

        # return state of all qubits for computing fidelity (statevector representation for efficiency)
        return qml.state()

    def _get_qasm_strictly_transversal_operation(self, wires: qml.wires.Wires):
        """Return QASM3 for a strictly transversal encoded operation.

        Args:
            wires (qml.wires.Wires): All wires (data, ancilla, target) on
                which the operation acts.

        Returns:
            str: QASM3 program implementing the strictly transversal encoded
            logical operation.
        """

        # construction for converting to QASM
        @qml.qnode(qml.device('default.qubit', wires=wires))
        def circuit():
            EncodedOperationStrictlyTransversal(operation=self.operation_properties.gate,
                                                wires_control=self.wires_data+self.wires_ancilla,
                                                wires_target=self.wires_data_target+self.wires_ancilla_target,
                                                operation_name=self.operation_properties.name)
            return qml.state()  # required for conversion, will be removed again later
        circuit()  # run once to create tape
        return self._tape_to_openqasm3_simplified(circuit._tape, wires=wires, precision=8)  # noqa

    def _get_qasm_transversal_operation(self, wires: qml.wires.Wires, parameters_operation_initial: torch.Tensor):
        """Return QASM3 for a trainable transversal encoded operation.

        Args:
            wires (qml.wires.Wires): All wires (data, ancilla, target).
            parameters_operation_initial (torch.Tensor): Transversal
                operation parameters.

        Returns:
            str: QASM3 program implementing the transversal encoded logical
            operation.
        """

        # construction for converting to QASM
        @qml.qnode(qml.device('default.qubit', wires=wires))
        def circuit():
            EncodedOperationTransversal(parameters=parameters_operation_initial,
                                        order_operation=self.operation_properties.order,
                                        wires_control=self.wires_data+self.wires_ancilla,
                                        wires_target=self.wires_data_target+self.wires_ancilla_target,
                                        gate1q=self.operation_properties.gateset[0],
                                        gate2q=self.operation_properties.gateset[1],
                                        flip=self.operation_properties.flip)
            return qml.state()  # required for conversion, will be removed again later
        circuit()  # run once to create tape
        return self._tape_to_openqasm3_simplified(circuit._tape, wires=wires, precision=8)  # noqa

    def _get_qasm_weakly_transversal_operation(self, wires: qml.wires.Wires, parameters_operation_initial: torch.Tensor):
        """Return QASM3 for a trainable weakly-transversal encoded operation.

        Args:
            wires (qml.wires.Wires): All wires (data, ancilla, target).
            parameters_operation_initial (torch.Tensor): Weakly-transversal
                operation parameters.

        Returns:
            str: QASM3 program implementing the weakly-transversal encoded
            logical operation.
        """

        # construction for converting to QASM
        @qml.qnode(qml.device('default.qubit', wires=wires))
        def circuit():
            EncodedOperationWeaklyTransversal(parameters=parameters_operation_initial,
                                              order_operation=self.operation_properties.order,
                                              wires_control=self.wires_data+self.wires_ancilla,
                                              wires_target=self.wires_data_target+self.wires_ancilla_target)
            return qml.state()  # required for conversion, will be removed again later
        circuit()  # run once to create tape
        return self._tape_to_openqasm3_simplified(circuit._tape, wires=wires, precision=8)  # noqa
