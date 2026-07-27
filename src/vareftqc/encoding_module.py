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

"""Encoding module for VarEFTQC.

This module defines :class:`EncodingModule`, which constructs and evaluates
a variational encoding ansatz for a quantum code, optionally under a
specified noise model. It is used to learn noise-resilient encodings and to
export trained encodings to OpenQASM 3.
"""

import pennylane as qml
import torch

from .helpers.data_structures import CodeProperties, NoiseProperties, ParametersEncoding
from .components.state_preparation import ProjectiveTwoDesign, HaarRandomState
from .base_module import BaseModule


class EncodingModule(BaseModule):
    """Variational encoding module for quantum codes.

    This module prepares input states (two-design or Haar-random), applies a
    trainable or static encoding ansatz on data and ancilla wires, applies
    the configured noise channel, and returns the resulting density matrix.
    It also provides a helper to export the encoding circuit to QASM.
    """

    def __init__(self, code_properties: CodeProperties, noise_properties: NoiseProperties):
        """Initialize the encoding module.

        Args:
            code_properties (CodeProperties): Code configuration including
                encoding properties and wire layout.
            noise_properties (NoiseProperties): Noise model applied after the
                encoding ansatz.
        """

        # check and compose wires
        self.code_properties = code_properties
        super().__init__(wires=self._compose_wires(), device_name='default.mixed', noise_properties=noise_properties)

    def run(self, parameters_encoding: ParametersEncoding, number_states: int = 0, seed_states: int = None):
        """Execute the encoding circuit and return the output density matrix.

        Args:
            parameters_encoding (ParametersEncoding): Encoding parameter
                container providing initial-layer and block parameters.
            number_states (int): Number of Haar-random input states to
                prepare. If ``0``, a spherical two-design is used instead.
            seed_states (int | None): Random seed/instance index for Haar
                state sampling.

        Returns:
            torch.Tensor: Density matrix of all encoding wires (data +
            ancilla), as returned by :func:`qml.density_matrix`.
        """

        return self._run(parameters_encoding_initial=parameters_encoding.parameters_encoding_initial,
                         parameters_encoding_block=parameters_encoding.parameters_encoding_block,
                         number_states=number_states, seed_states=seed_states)

    def draw(self, parameters_encoding: ParametersEncoding, number_states: int = 0, seed_states: int = None,
             level: int = 0):
        """Print an ASCII diagram of the encoding circuit.

        Args:
            parameters_encoding (ParametersEncoding): Encoding parameter
                container providing initial-layer and block parameters.
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
                   number_states=number_states, seed_states=seed_states, level=level)

    def get_qasm_encoding(self, parameters_encoding: ParametersEncoding = None, normalize: bool = True):
        """Return an OpenQASM 3 representation of the encoding ansatz.

             If the encoding is trainable, the given parameters are used to fix
             the circuit; otherwise the static QASM from
             :attr:`code_properties.encoding_properties.qasm` is used. Parameters
             can optionally be wrapped modulo a symmetry (e.g. ``4π``) before
             conversion to QASM.

             Args:
                 parameters_encoding (ParametersEncoding | None): Encoding
                     parameters to use when the encoding is trainable. Ignored if
                     the encoding is static.
                 normalize (bool): If ``True``, use
                     :meth:`ParametersEncoding.normalized_parameters`; if
                     ``False``, use raw parameters via
                     :meth:`ParametersEncoding.parameters` with ``grad=False``.

             Returns:
                 str: QASM3 string describing the encoding circuit, including
                 metadata header lines.
             """

        if self.code_properties.train_encoding:
            if normalize:
                parameters_encoding_initial, parameters_encoding_block = parameters_encoding.normalized_parameters()
            else:
                parameters_encoding_initial, parameters_encoding_block = parameters_encoding.parameters(grad=False)
        else:
            parameters_encoding_initial, parameters_encoding_block = None, None
        return self._get_qasm_ansatz(
            wires_data=self.code_properties.wires_data, wires_ancilla=self.code_properties.wires_ancilla,
            qasm=self.code_properties.encoding_properties.qasm,
            instance=self.code_properties.encoding_properties.instance,
            connectivity=self.code_properties.encoding_properties.connectivity,
            gateset=self.code_properties.encoding_properties.gateset,
            parameters_initial=parameters_encoding_initial, parameters_block=parameters_encoding_block
        )

    def _circuit(self, parameters_encoding_initial: torch.Tensor = None, parameters_encoding_block: torch.Tensor = None,
                 instance_encoding: int = None, number_states: int = 0, seed_states: int = None):
        """Define the encoding circuit used by the QNode.

        The circuit:

        * prepares either two-design states or Haar-random states on the data
          wires,
        * applies the encoding ansatz on data and ancilla wires (either
          static or variational, depending on encoding properties), and
        * applies the configured noise channel on all encoding wires.

        Args:
            parameters_encoding_initial (torch.Tensor | None): Initial-layer
                encoding parameters. Required for trainable encodings.
            parameters_encoding_block (torch.Tensor | None): Block encoding
                parameters. Required for trainable encodings.
            instance_encoding (int | None): Encoding instance index (not used
                directly here; the instance is taken from
                :attr:`code_properties.encoding_properties.instance`).
            number_states (int): Number of Haar-random input states to
                prepare. If ``0``, a spherical two-design is used instead.
            seed_states (int | None): Random seed/instance index for Haar
                state sampling.

        Returns:
            qml.measurements.MeasurementProcess: Density-matrix measurement on
            all encoding wires, as created by :func:`qml.density_matrix`.
        """

        # initialize state
        if 0 == number_states:  # use spherical 2-design
            ProjectiveTwoDesign(wires=self.code_properties.wires_data)
        else:
            HaarRandomState(wires=self.code_properties.wires_data, number=number_states, seed=seed_states)
        qml.Barrier(wires=self.wires, only_visual=False)

        # place encoding ansatz on data and ancilla qubits (all qubits in this case)
        self._apply_ansatz(wires=self.code_properties.wires_data + self.code_properties.wires_ancilla,
                           qasm=self.code_properties.encoding_properties.qasm,
                           instance=self.code_properties.encoding_properties.instance,
                           connectivity=self.code_properties.encoding_properties.connectivity,
                           gateset=self.code_properties.encoding_properties.gateset,
                           parameters_initial=parameters_encoding_initial,
                           parameters_block=parameters_encoding_block)
        qml.Barrier(wires=self.wires, only_visual=False)

        # apply the error channel to all qubits
        self._apply_error_channel(wires=self.wires)

        # return density matrix of all qubits (for computing e.g. trace distance)
        return qml.density_matrix(wires=self.wires)  # noqa

    def _compose_wires(self):
        """Compose and validate the full encoding wire register.

        Ensures that at least one ancilla wire is present and returns the
        concatenation of data and ancilla wires used by the encoding module.

        Returns:
            qml.wires.Wires: Combined wire list ``data + ancilla``.

        Raises:
            ValueError: If fewer than one ancilla wire is specified in
                :attr:`code_properties`.
        """

        if self.code_properties.num_wires_ancilla < 1:
            raise ValueError(f'At least one ancilla wire is required ({self.code_properties.num_wires_ancilla} '
                             f'selected).')
        return self.code_properties.wires_data + self.code_properties.wires_ancilla
