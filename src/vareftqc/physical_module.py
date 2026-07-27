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

"""Physical noise model module for VarEFTQC.

This module defines :class:`PhysicalModule`, which prepares random input
states (two-design or Haar-random), applies a configurable noise channel,
and returns the resulting density matrix on the data wires. It is used as a
reference baseline and as a noise-free/noisy target in training.
"""

import pennylane as qml
from .helpers.data_structures import NoiseProperties
from .base_module import BaseModule
from .components.state_preparation import ProjectiveTwoDesign, HaarRandomState


class PhysicalModule(BaseModule):
    """Physical noise model acting on data wires.

    This module prepares either spherical two-design states or Haar-random
    states on the data wires, applies the noise channel specified in
    :class:`NoiseProperties`, and returns the resulting density matrix on
    the data subsystem.

    Attributes:
        wires_data (qml.wires.Wires): Data wires on which states are prepared
            and noise is applied.
    """

    def __init__(self, wires_data: qml.wires.Wires, noise_properties: NoiseProperties):
        """Initialize the physical noise model.

        Args:
            wires_data (qml.wires.Wires): Wires representing the data
                subsystem.
            noise_properties (NoiseProperties): Noise model to be applied to
                the data wires.
        """

        # check and compose wires
        super().__init__(wires=wires_data, device_name='default.mixed', noise_properties=noise_properties)
        self.wires_data = wires_data

    def run(self, number_states: int = 0, seed_states: int = None):
        """Execute the physical model and return the data density matrix.

        Args:
            number_states (int): Number of Haar-random input states to
                prepare. If ``0``, a spherical two-design is used instead.
            seed_states (int | None): Random seed/instance index for state
                sampling. Ignored if ``number_states == 0``.

        Returns:
            torch.Tensor: Density matrix of the data wires as returned by
            :func:`qml.density_matrix`.
        """

        return self._run(number_states=number_states, seed_states=seed_states)

    def draw(self, number_states: int = 0, seed_states: int = None, level: int = 0):
        """Print an ASCII diagram of the physical noise model circuit.

        Args:
            number_states (int): Number of Haar-random input states to
                prepare. If ``0``, a spherical two-design is used instead.
            seed_states (int | None): Random seed/instance index for state
                sampling.
            level (int): Detail level passed to :func:`qml.draw`. Defaults to
                ``0``.

        Returns:
            None
        """

        self._draw(number_states=number_states, seed_states=seed_states, level=level)

    def _circuit(self, number_states: int = 0, seed_states: int = None):
        """Define the physical noise model circuit.

        The circuit:

        * prepares either two-design states or Haar-random states on the data
          wires,
        * applies the configured noise channel to the data wires, and
        * returns a density-matrix measurement on the data subsystem.

        Args:
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
            ProjectiveTwoDesign(wires=self.wires_data)
        else:
            HaarRandomState(wires=self.wires_data, number=number_states, seed=seed_states)
        qml.Barrier(wires=self.wires, only_visual=False)

        # apply the error channel to data qubits
        self._apply_error_channel(wires=self.wires_data)

        # return density matrix of data qubits (for computing e.g. trace distance)
        return qml.density_matrix(wires=self.wires_data)  # noqa
