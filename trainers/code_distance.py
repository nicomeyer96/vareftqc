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

"""Code-distance estimation utilities for VarEFTQC.

This module estimates the (approximate) distance of a given quantum code by
injecting Pauli fault patterns of increasing weight into the encoding circuit,
simulating their effect, and comparing the resulting states to a noise-free
reference using a distinguishability loss.
"""

import itertools
import torch

from vareftqc import PhysicalModule, EncodingModule
from vareftqc.loss_functions import distinguishability_loss
from vareftqc.helpers import CodeProperties, NoiseProperties, TrainingProperties
from vareftqc.helpers.data_structures import ParametersEncoding


def extract_code_distance(code_properties: CodeProperties, training_properties: TrainingProperties):
    """Estimate the code distance by scanning Pauli fault patterns.

    For each fault weight from 1 up to the total number of encoding wires,
    this function:

    * builds a noise-free reference using a :class:`PhysicalModule`,
    * enumerates all fault patterns of the given weight (over X, Y, Z), and
    * checks whether any such pattern violates a distinguishability
      threshold via :func:`_test_faults`.

    Once a fault weight is found for which at least one pattern fails the
    test, a message with the inferred code parameters
    ``[[n, k, d]]`` is printed, where:

    * ``n = code_properties.num_wires_encoding``,
    * ``k = code_properties.num_wires_data``, and
    * ``d`` is the smallest fault weight failing the test.

    Args:
        code_properties (CodeProperties): Code configuration containing the
            encoding layout and number of wires.
        training_properties (TrainingProperties): Training configuration and
            encoding parameters used to instantiate the encoding circuit.

    Returns:
        None
    """

    # extract target
    print('Creating noise-free target: ', end='')
    noisefree_fn = PhysicalModule(wires_data=code_properties.wires_data,
                                  noise_properties=NoiseProperties(noise='dummy', noise_strength=0.0))
    noisefree = noisefree_fn.run()

    for number_faults in range(1, code_properties.num_wires_encoding + 1):
        detectable = _test_faults(code_properties=code_properties,
                                  parameters_encoding=training_properties.parameters_encoding,
                                  noisefree=noisefree, number_faults=number_faults)
        if not detectable:
            print(f'\nAt least one fault pattern on {number_faults} wire(s) can not be detected.\nThe distance of the '
                  f'code is {number_faults}, i.e. it has code parameters '
                  f'[[{code_properties.num_wires_encoding},{code_properties.num_wires_data},{number_faults}]].')
            break


def _test_faults(code_properties: CodeProperties, parameters_encoding: ParametersEncoding, noisefree: torch.Tensor,
                 number_faults: int, strength: float = 0.5, threshold: float = 0.002):
    """Test all Pauli fault patterns of a given weight against a threshold.

    For a given number of faults (weight), this function:

    * enumerates all strings of Pauli errors of length ``number_faults`` over
      ``{'X', 'Y', 'Z'}`` (including permutations),
    * enumerates all subsets of wires of size ``number_faults``,
    * combines them into fault patterns (error string + wire subset), and
    * for each pattern, constructs a Pauli noise model and evaluates the
      encoding against a noise-free reference using
      :func:`distinguishability_loss`.

    The Pauli noise is parameterized such that only the selected wires carry
    non-zero strength (``strength``), with X/Z weights chosen according to the
    error type; Y is implicitly realized when both X and Z contributions are
    present.

    Args:
        code_properties (CodeProperties): Code configuration (used to set the
            number of encoding wires and construct the encoding module).
        parameters_encoding (ParametersEncoding): Encoding parameters used for
            the test.
        noisefree (torch.Tensor): Noise-free reference states (target).
        number_faults (int): Fault weight (number of faulty wires) to test.
        strength (float): Noise strength assigned to faulty wires in the
            Pauli noise model. Defaults to ``0.5``.
        threshold (float): Threshold on the maximum distinguishability loss
            used to decide whether a fault pattern is considered critical.

    Returns:
        bool: ``True`` if **no** fault pattern of the given weight yields a
        maximum distinguishability loss above ``threshold``; ``False`` if at
        least one pattern does.

    Raises:
        RuntimeError: If ``number_faults`` is larger than the total number of
            encoding wires in the code.
    """

    if number_faults > code_properties.num_wires_encoding:
        raise RuntimeError(f'Cannot evaluate weight-{number_faults} fault on {code_properties.num_wires_encoding}-qubit code.')
    # all combinations and orderings of Pauli errors of length `number_faults`
    faults = itertools.combinations_with_replacement(['X', 'Y', 'Z'], number_faults)
    faults = [list(set(itertools.permutations(f))) for f in faults]
    faults = [item for row in faults for item in row]
    # determine (unordered) subsets of wires that is affected by the faults
    faulty_wires = list(itertools.combinations(list(range(code_properties.num_wires_encoding)), number_faults))
    # construct all combinations
    faults_faulty_wires = list(itertools.product(faults, faulty_wires))
    fault_patterns = len(faults_faulty_wires)
    print(f'[{number_faults} faulty wire(s)] Analyzing {fault_patterns} fault patterns '
          f'({len(faults)} error string(s) on {len(faulty_wires)} wire allocation(s)).')
    counter = 0
    for fault, wire in faults_faulty_wires:
        counter += 1
        print(f'[{counter}/{fault_patterns}] ', end='')
        # encode as Pauli noise
        noise_strength = [0.0 for _ in range(code_properties.num_wires_encoding)]
        noise_pauli_x = [0.0 for _ in range(code_properties.num_wires_encoding)]
        noise_pauli_z = [0.0 for _ in range(code_properties.num_wires_encoding)]
        for f, w in zip(fault, wire):
            noise_strength[w] = strength
            if 'X' == f:
                noise_pauli_x[w] = 1.0
            elif 'Z' == f:
                noise_pauli_z[w] = 1.0
            # Y is selected automatically if X and Z are set to zero
        # set up encoding with resulting noise model
        encoding_fn = EncodingModule(code_properties=code_properties,
                                     noise_properties=NoiseProperties(noise='pauli', noise_strength=noise_strength,
                                                                      noise_pauli_x=noise_pauli_x, noise_pauli_z=noise_pauli_z))
        # # encoding = encoding_fn.draw(parameters_encoding=parameters_encoding)
        with torch.no_grad():
            encoding = encoding_fn.run(parameters_encoding=parameters_encoding)
        _, loss_max = distinguishability_loss(prediction=encoding, groundtruth=noisefree)
        if loss_max.detach().numpy() >= threshold:
            print(f'Fault pattern {fault} on wires {wire} leads to a distinguishability loss above threshold.')
            return False
    print(f'No fault pattern on {number_faults} wires leads to a distinguishability loss above threshold.')
    return True
