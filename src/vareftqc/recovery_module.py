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

"""Recovery module for VarEFTQC.

This module defines :class:`RecoveryModule`, which constructs and evaluates
a variational (or static) recovery ansatz applied after encoding and noise,
followed by a decoding step. It can also export the recovery circuit to
OpenQASM 3.
"""

import pennylane as qml
import torch

from .helpers.data_structures import CodeProperties, NoiseProperties, ParametersEncoding, ParametersRecovery
from .components.state_preparation import ProjectiveTwoDesign, HaarRandomState
from .base_module import BaseModule


class RecoveryModule(BaseModule):
    """Variational recovery module for quantum codes.

    This module:

    * prepares input states (two-design or Haar-random) on the data wires,
    * applies the encoding ansatz on data + ancilla,
    * applies the configured noise channel on data + ancilla,
    * applies a recovery ansatz on all wires (data + ancilla + recovery),
    * applies the adjoint (decoding) encoding ansatz on data + ancilla, and
    * returns the density matrix on the data wires.

    It also provides a helper to export the recovery circuit to QASM.
    """

    def __init__(self, code_properties: CodeProperties, noise_properties: NoiseProperties):
        """Initialize the recovery module.

        Args:
            code_properties (CodeProperties): Code configuration including
                encoding, recovery, and wire layout.
            noise_properties (NoiseProperties): Noise model applied between
                encoding and recovery.
        """

        # check and compose wires
        self.code_properties = code_properties
        super().__init__(wires=self._compose_wires(), device_name='default.mixed', noise_properties=noise_properties)

    def run(self, parameters_encoding: ParametersEncoding, parameters_recovery: ParametersRecovery,
            number_states: int = 0, seed_states: int = None):
        """Execute the full encode–noise–recover–decode circuit.

        Args:
            parameters_encoding (ParametersEncoding): Encoding parameter
                container providing initial-layer and block parameters for
                encoding/decoding.
            parameters_recovery (ParametersRecovery): Recovery parameter
                container providing initial-layer and block parameters for
                the recovery ansatz.
            number_states (int): Number of Haar-random input states to
                prepare. If ``0``, a spherical two-design is used instead.
            seed_states (int | None): Random seed/instance index for Haar
                state sampling.

        Returns:
            torch.Tensor: Density matrix on the data wires after encoding,
            noise, recovery, and decoding, as returned by
            :func:`qml.density_matrix`.
        """

        return self._run(parameters_encoding_initial=parameters_encoding.parameters_encoding_initial,
                         parameters_encoding_block=parameters_encoding.parameters_encoding_block,
                         parameters_recovery_initial=parameters_recovery.parameters_recovery_initial,
                         parameters_recovery_block=parameters_recovery.parameters_recovery_block,
                         number_states=number_states, seed_states=seed_states)

    def draw(self, parameters_encoding: ParametersEncoding, parameters_recovery: ParametersRecovery,
             number_states: int = 0, seed_states: int = None, level: int = 0):
        """Print an ASCII diagram of the recovery circuit.

        Args:
            parameters_encoding (ParametersEncoding): Encoding parameter
                container for the encode/decode layers.
            parameters_recovery (ParametersRecovery): Recovery parameter
                container for the recovery ansatz.
            number_states (int): Number of Haar-random input states to
                prepare. If ``0``, a spherical two-design is used instead.
            seed_states (int | None): Random seed/instance index for Haar
                state sampling.
            level (int): Detail level passed to :func:`qml.draw`. Defaults to
                ``0``.

        Returns:
            None
        """

        self._draw(parameters_encoding_initial=parameters_encoding.parameters_encoding_initial,
                   parameters_encoding_block=parameters_encoding.parameters_encoding_block,
                   parameters_recovery_initial=parameters_recovery.parameters_recovery_initial,
                   parameters_recovery_block=parameters_recovery.parameters_recovery_block,
                   number_states=number_states, seed_states=seed_states, level=level)

    def get_qasm_recovery(self, parameters_recovery: ParametersRecovery = None, normalize: bool = True):
        """Return an OpenQASM 3 representation of the recovery ansatz.

        If the recovery is trainable, the given parameters are used to fix
        the circuit; otherwise the static QASM from
        :attr:`code_properties.recovery_properties.qasm` is used. Parameters
        can optionally be wrapped modulo a symmetry (e.g. ``4π``) before
        conversion to QASM.

        Args:
            parameters_recovery (ParametersRecovery | None): Recovery
                parameters to use when the recovery is trainable. Ignored if
                the recovery is static.
            normalize (bool): If ``True``, use
                :meth:`ParametersRecovery.normalized_parameters`; if
                ``False``, use raw parameters via
                :meth:`ParametersRecovery.parameters` with ``grad=False``.

        Returns:
            str: QASM3 string describing the recovery circuit, including
            metadata header lines.
        """

        if self.code_properties.train_recovery:
            if normalize:
                parameters_recovery_initial, parameters_recovery_block = parameters_recovery.normalized_parameters()
            else:
                parameters_recovery_initial, parameters_recovery_block = parameters_recovery.parameters(grad=False)
        else:
            parameters_recovery_initial, parameters_recovery_block = None, None
        return self._get_qasm_ansatz(
            wires_data=self.code_properties.wires_data, wires_ancilla=self.code_properties.wires_ancilla,
            wires_recovery=self.code_properties.wires_recovery,
            qasm=self.code_properties.recovery_properties.qasm,
            instance=self.code_properties.recovery_properties.instance,
            gateset=self.code_properties.recovery_properties.gateset,
            parameters_initial=parameters_recovery_initial, parameters_block=parameters_recovery_block
        )

    def _circuit(self, parameters_encoding_initial: torch.Tensor = None, parameters_encoding_block: torch.Tensor = None,
                 parameters_recovery_initial: torch.Tensor = None, parameters_recovery_block: torch.Tensor = None,
                 instance_encoding: int = None, number_states: int = 0, seed_states: int = None):
        """Define the encode–noise–recover–decode circuit.

        The circuit:

        * prepares either two-design states or Haar-random states on the data
          wires,
        * applies the encoding ansatz on data + ancilla wires,
        * applies the noise channel on data + ancilla wires,
        * applies the recovery ansatz on all wires (data + ancilla + recovery),
        * applies the adjoint (decoding) encoding ansatz on data + ancilla
          wires, and
        * returns the density matrix on the data wires.

        Args:
            parameters_encoding_initial (torch.Tensor | None): Initial-layer
                encoding parameters for the encoding/decoding ansatz.
            parameters_encoding_block (torch.Tensor | None): Block encoding
                parameters for the encoding/decoding ansatz.
            parameters_recovery_initial (torch.Tensor | None): Initial-layer
                recovery parameters.
            parameters_recovery_block (torch.Tensor | None): Block recovery
                parameters.
            instance_encoding (int | None): Encoding instance index (not used
                directly here; the instance is taken from
                :attr:`code_properties.encoding_properties.instance`).
            number_states (int): Number of Haar-random input states to
                prepare. If ``0``, a spherical two-design is used instead.
            seed_states (int | None): Random seed/instance index for Haar
                state sampling.

        Returns:
            qml.measurements.MeasurementProcess: Density-matrix measurement on
            the data wires, as created by :func:`qml.density_matrix`.
        """

        # initialize state
        if 0 == number_states:  # use spherical 2-design
            ProjectiveTwoDesign(wires=self.code_properties.wires_data)
        else:
            HaarRandomState(wires=self.code_properties.wires_data, number=number_states, seed=seed_states)
        qml.Barrier(wires=self.wires, only_visual=False)

        if self.code_properties.encoding_properties.connectivity is not None:
            raise NotImplementedError('Setting `connectivity` is currently not supported for training recovery module.')

        # place encoding ansatz on data and ancilla qubits (exclude recovery qubits)
        self._apply_ansatz(wires=self.code_properties.wires_data + self.code_properties.wires_ancilla,
                           qasm=self.code_properties.encoding_properties.qasm,
                           instance=self.code_properties.encoding_properties.instance,
                           gateset=self.code_properties.encoding_properties.gateset,
                           parameters_initial=parameters_encoding_initial,
                           parameters_block=parameters_encoding_block)
        qml.Barrier(wires=self.wires, only_visual=False)

        # apply the error channel on data and ancilla qubits (exclude recovery qubits)
        self._apply_error_channel(wires=self.code_properties.wires_data + self.code_properties.wires_ancilla)
        qml.Barrier(wires=self.wires, only_visual=False)

        # place recovery ansatz on all qubits
        self._apply_ansatz(wires=self.wires,
                           qasm=self.code_properties.recovery_properties.qasm,
                           instance=self.code_properties.recovery_properties.instance,
                           gateset=self.code_properties.recovery_properties.gateset,
                           parameters_initial=parameters_recovery_initial,
                           parameters_block=parameters_recovery_block)
        qml.Barrier(wires=self.wires, only_visual=False)

        # place decoding ansatz on data and ancilla qubits (exclude recovery qubits)
        self._apply_ansatz(wires=self.code_properties.wires_data + self.code_properties.wires_ancilla,
                           qasm=self.code_properties.encoding_properties.qasm,
                           instance=self.code_properties.encoding_properties.instance,
                           gateset=self.code_properties.encoding_properties.gateset,
                           parameters_initial=parameters_encoding_initial,
                           parameters_block=parameters_encoding_block,
                           adjoint=True)

        # return density matrix of data qubits (for computing e.g. fidelity)
        return qml.density_matrix(wires=self.code_properties.wires_data)  # noqa

    def _compose_wires(self):
        """Compose and validate the full wire register for recovery.

        Ensures that at least one ancilla wire is present and returns the
        concatenation of data, ancilla, and recovery wires used by the
        recovery module.

        Returns:
            qml.wires.Wires: Combined wire list ``data + ancilla + recovery``.

        Raises:
            ValueError: If fewer than one ancilla wire is specified in
                :attr:`code_properties`.
        """

        if self.code_properties.num_wires_ancilla < 1:
            raise ValueError(f'At least one ancilla wire is required ({self.code_properties.num_wires_ancilla} '
                             f'selected).')
        return (self.code_properties.wires_data + self.code_properties.wires_ancilla
                + self.code_properties.wires_recovery)
