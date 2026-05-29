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

"""Entry point for running a VarEFTQC experiment.

This script parses configuration and command-line options, constructs the
code, noise, and training properties, and then orchestrates the full
training pipeline:

* optional code-distance evaluation,
* setup of noise-free and noisy physical baselines,
* training of the encoding and logical operations, and
* optional training of the recovery operation,

followed by storing logs and results.
"""

import warnings

from vareftqc.helpers import parse, CodeProperties, NoiseProperties, TrainingProperties
from vareftqc import PhysicalModule, EncodingModule, RecoveryModule
from trainers.logger import Logger
from trainers.encoding_operation import train_encoding_operation
from trainers.recovery import train_recovery
from trainers.helpers import initialize_operations_fn, store_logs, store_results
from trainers.code_distance import extract_code_distance


def run_vareftqc(code_properties: CodeProperties, noise_properties: NoiseProperties,
                 training_properties: TrainingProperties, evaluate_code_distance: bool, path: str, tensorboard: bool):
    """Run a complete VarEFTQC experiment for the given configuration.

    Depending on the flags, this either evaluates the code distance of the
    provided code or trains the encoding, logical operations, and optional
    recovery, and then stores logs and results.

    Args:
        code_properties (CodeProperties): Description of the code, including
            encoding, optional recovery, and logical operations.
        noise_properties (NoiseProperties): Noise model used during training
            and evaluation of the physical baseline and variational circuits.
        training_properties (TrainingProperties): Training configuration and
            initialized parameter containers.
        evaluate_code_distance (bool): If ``True``, skip training and only
            evaluate the potential code distance for the given code.
        path (str): Path to the experiment results folder where logs and
            artifacts will be stored.
        tensorboard (bool): If ``True``, enable TensorBoard logging and write
            logs into ``path``.

    Returns:
        None
    """

    if evaluate_code_distance:
        extract_code_distance(code_properties=code_properties, training_properties=training_properties)
        return None

    # set up logger
    logger = Logger(training_properties=training_properties,
                    encoding_properties=code_properties.encoding_properties,
                    recovery_properties=code_properties.recovery_properties,
                    operation_static=code_properties.operation_static,
                    operation_strictly_transversal=code_properties.operation_strictly_transversal,
                    operation_transversal=code_properties.operation_transversal,
                    operation_weakly_transversal=code_properties.operation_weakly_transversal,
                    operation_non_transversal=code_properties.operation_non_transversal,
                    tensorboard_path=path if tensorboard else None)

    # set up and simulate noise-free physical setup (required as target in loss functions)
    print('Creating noise-free target: ', end='')
    noisefree_fn = PhysicalModule(wires_data=code_properties.wires_data,
                                  noise_properties=NoiseProperties(noise='dummy', noise_strength=0.0))

    # set up and simulate noisy physical setups (as baseline)
    print(f'Creating noisy baseline.')
    baseline_fn = PhysicalModule(wires_data=code_properties.wires_data, noise_properties=noise_properties)

    # set up the encoding module
    print('Creating encoding module.')
    encoding_fn = EncodingModule(code_properties=code_properties, noise_properties=noise_properties)
    # encoding_fn.draw(parameters_encoding=training_properties.parameters_encoding, level=1)

    # optionally set up operations modules
    (operations_static_fn, operations_strictly_transversal_fn, operations_transversal_fn, operations_weakly_transversal_fn,
     operations_non_transversal_fn) = initialize_operations_fn(code_properties=code_properties,
                                                               training_properties=training_properties, draw_level=None)

    # optionally set up recovery module
    if code_properties.recovery_properties is not None:
        print('Creating recovery module.')
        recovery_fn = RecoveryModule(code_properties=code_properties, noise_properties=noise_properties)
        # recovery_fn.draw(parameters_encoding=training_properties.parameters_encoding,
        #                  parameters_recovery=training_properties.parameters_recovery, level=1)
    else:
        recovery_fn = None

    ##########################################################
    # train encoding or/and (strictly) transversal operations
    ##########################################################
    print('\n==========\n')
    with warnings.catch_warnings(action="ignore", category=UserWarning):  # filter harmless but annoying PyTorch warning
        train_encoding_operation(logger=logger, code_properties=code_properties, training_properties=training_properties,
                                 encoding_fn=encoding_fn, noisefree_fn=noisefree_fn, baseline_fn=baseline_fn,
                                 operations_static_fn=operations_static_fn,
                                 operations_strictly_transversal_fn=operations_strictly_transversal_fn,
                                 operations_transversal_fn=operations_transversal_fn,
                                 operations_weakly_transversal_fn=operations_weakly_transversal_fn,
                                 operations_non_transversal_fn=operations_non_transversal_fn)

    #################
    # train recovery
    #################
    if recovery_fn is not None:
        with warnings.catch_warnings(action="ignore", category=UserWarning):
            train_recovery(logger=logger, code_properties=code_properties, training_properties=training_properties,
                           recovery_fn=recovery_fn, noisefree_fn=noisefree_fn, baseline_fn=baseline_fn)

    ########################
    # store results and log
    ########################
    # store logging data
    store_logs(path=path, logger=logger)
    # store result data
    store_results(path=path, code_properties=code_properties, training_properties=training_properties,
                  encoding_fn=encoding_fn, recovery_fn=recovery_fn,
                  operations_static_fn=operations_static_fn,
                  operations_strictly_transversal_fn=operations_strictly_transversal_fn,
                  operations_transversal_fn=operations_transversal_fn,
                  operations_weakly_transversal_fn=operations_weakly_transversal_fn,
                  operations_non_transversal_fn=operations_non_transversal_fn)
    return None


if __name__ == '__main__':

    _code_properties, _noise_properties, _training_properties, _evaluate_code_distance, _path, _tensorboard = parse()
    run_vareftqc(code_properties=_code_properties,
                 noise_properties=_noise_properties,
                 training_properties=_training_properties,
                 evaluate_code_distance=_evaluate_code_distance,
                 path=_path, tensorboard=_tensorboard)
