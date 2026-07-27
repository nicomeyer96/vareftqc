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

"""Helper functions for initializing operation modules and storing results.

This module provides utilities to:

* construct target and prediction modules for all configured logical
  operations, and
* store logs, QASM files, and trained model properties/parameters to disk.
"""

import os
import pickle
from typing import Tuple

from vareftqc.helpers import CodeProperties, TrainingProperties
from vareftqc import EncodingModule, RecoveryModule, OperationTargetModule, OperationPredictionModule
from .logger import Logger


def initialize_operations_fn(code_properties: CodeProperties, training_properties: TrainingProperties,
                             draw_level: int = None) \
        -> Tuple[dict[str, Tuple[OperationTargetModule, OperationPredictionModule]], ...]:
    """Create target and prediction modules for all logical operations.

        For each operation type (static, strictly transversal, transversal,
        weakly transversal, non-transversal) present in ``code_properties``,
        this function instantiates:

        * an :class:`OperationTargetModule` that realizes the ideal logical
          operation, and
        * an :class:`OperationPredictionModule` that realizes the circuit to be
          compared or trained.

        If ``draw_level`` is not ``None``, the corresponding circuits are drawn
        once using the current encoding (and operation) parameters.

        Args:
            code_properties (CodeProperties): Code configuration including all
                logical operation specifications.
            training_properties (TrainingProperties): Training configuration and
                parameter containers (used for drawing parameterized circuits).
            draw_level (int | None): Optional detail level passed to the
                ``draw`` methods of the operation modules. If ``None``, no
                drawing is performed.

        Returns:
            tuple:
                operations_static_fn (dict[str, tuple[OperationTargetModule, OperationPredictionModule]]):
                    Target/prediction modules for static operations.
                operations_strictly_transversal_fn (dict[str, tuple[OperationTargetModule, OperationPredictionModule]]):
                    Modules for strictly transversal operations.
                operations_transversal_fn (dict[str, tuple[OperationTargetModule, OperationPredictionModule]]):
                    Modules for trainable transversal operations.
                operations_weakly_transversal_fn (dict[str, tuple[OperationTargetModule, OperationPredictionModule]]):
                    Modules for trainable weakly transversal operations.
                operations_non_transversal_fn (dict[str, tuple[OperationTargetModule, OperationPredictionModule]]):
                    Modules for trainable non-transversal operations.
        """

    # optionally set up static operation modules
    operations_static_fn = {}
    if code_properties.operation_static is not None:
        for operation_static in code_properties.operation_static.values():
            print(f'Creating operation module (static {operation_static.name}).')
            operation_target_fn = OperationTargetModule(code_properties=code_properties,
                                                        operation_properties=operation_static)
            operation_prediction_fn = OperationPredictionModule(code_properties=code_properties,
                                                                operation_properties=operation_static)
            if draw_level is not None:
                operation_target_fn.draw(parameters_encoding=training_properties.parameters_encoding,
                                         level=draw_level)
                operation_prediction_fn.draw(parameters_encoding=training_properties.parameters_encoding,
                                             level=draw_level)
            operations_static_fn[operation_static.name] = (operation_target_fn, operation_prediction_fn)

    # optionally set up strictly transversal operation modules
    operations_strictly_transversal_fn = {}
    if code_properties.operation_strictly_transversal is not None:
        for operation_strictly_transversal in code_properties.operation_strictly_transversal.values():
            print(f'Creating operation module (strictly transversal {operation_strictly_transversal.name}).')
            operation_target_fn = OperationTargetModule(code_properties=code_properties,
                                                        operation_properties=operation_strictly_transversal)
            operation_prediction_fn = OperationPredictionModule(code_properties=code_properties,
                                                                operation_properties=operation_strictly_transversal)
            if draw_level is not None:
                operation_target_fn.draw(parameters_encoding=training_properties.parameters_encoding,
                                         level=draw_level)
                operation_prediction_fn.draw(parameters_encoding=training_properties.parameters_encoding,
                                             level=draw_level)
            operations_strictly_transversal_fn[operation_strictly_transversal.name] = (operation_target_fn,
                                                                                       operation_prediction_fn)

    # optionally set up transversal operation modules
    operations_transversal_fn = {}
    if code_properties.operation_transversal is not None:
        for operation_transversal in code_properties.operation_transversal.values():
            print(f'Creating operation module (transversal {operation_transversal.name}).')
            operation_target_fn = OperationTargetModule(code_properties=code_properties,
                                                        operation_properties=operation_transversal)
            operation_prediction_fn = OperationPredictionModule(code_properties=code_properties,
                                                                operation_properties=operation_transversal)
            if draw_level is not None:
                operation_target_fn.draw(parameters_encoding=training_properties.parameters_encoding, level=draw_level)
                operation_prediction_fn.draw(parameters_encoding=training_properties.parameters_encoding,
                                             parameters_operation=training_properties.parameters_operation.
                                             parameters_operation_transversal.get(operation_transversal.name),
                                             level=draw_level)
            operations_transversal_fn[operation_transversal.name] = (operation_target_fn, operation_prediction_fn)

    # optionally set up weakly-transversal operation modules
    operations_weakly_transversal_fn = {}
    if code_properties.operation_weakly_transversal is not None:
        for operation_weakly_transversal in code_properties.operation_weakly_transversal.values():
            print(f'Creating operation module (weakly-transversal {operation_weakly_transversal.name}).')
            operation_target_fn = OperationTargetModule(code_properties=code_properties,
                                                        operation_properties=operation_weakly_transversal)
            operation_prediction_fn = OperationPredictionModule(code_properties=code_properties,
                                                                operation_properties=operation_weakly_transversal)
            if draw_level is not None:
                operation_target_fn.draw(parameters_encoding=training_properties.parameters_encoding, level=draw_level)
                operation_prediction_fn.draw(parameters_encoding=training_properties.parameters_encoding,
                                             parameters_operation=training_properties.parameters_operation.
                                             parameters_operation_weakly_transversal.get(operation_weakly_transversal.name),
                                             level=draw_level)
            operations_weakly_transversal_fn[operation_weakly_transversal.name] = (operation_target_fn, operation_prediction_fn)

    # optionally set up non-transversal operation modules
    operations_non_transversal_fn = {}
    if code_properties.operation_non_transversal is not None:
        for operation_non_transversal in code_properties.operation_non_transversal.values():
            print(f'Creating operation module (non-transversal {operation_non_transversal.name}).')
            operation_target_fn = OperationTargetModule(code_properties=code_properties,
                                                        operation_properties=operation_non_transversal)
            operation_prediction_fn = OperationPredictionModule(code_properties=code_properties,
                                                                operation_properties=operation_non_transversal)
            if draw_level is not None:
                operation_target_fn.draw(parameters_encoding=training_properties.parameters_encoding, level=draw_level)
                operation_prediction_fn.draw(parameters_encoding=training_properties.parameters_encoding,
                                             parameters_operation=training_properties.parameters_operation.
                                             parameters_operation_non_transversal.get(operation_non_transversal.name),
                                             level=draw_level)
            operations_non_transversal_fn[operation_non_transversal.name] = (operation_target_fn,
                                                                             operation_prediction_fn)

    return (operations_static_fn, operations_strictly_transversal_fn, operations_transversal_fn,
            operations_weakly_transversal_fn, operations_non_transversal_fn)


def store_logs(path: str, logger: Logger):
    """Serialize and store the logger object to disk.

    The TensorBoard writer is first detached from the logger (since it is not
    picklable), and then the entire :class:`Logger` instance is stored as a
    pickle file ``logs.pkl`` in the given directory.

    Args:
        path (str): Path to the experiment results folder where the log file
            will be saved.
        logger (Logger): Logger instance containing all recorded losses.

    Returns:
        None
    """

    logger.disconnect_tensorboard()  # tensorboard logger is not / cannot be stored via this function
    logger_file = os.path.join(path, 'logs.pkl')
    with open(logger_file, 'wb') as ff:
        pickle.dump(logger, ff)  # noqa
    print(f'Stored logging file to {logger_file}.')


def store_results(path: str, code_properties: CodeProperties, training_properties: TrainingProperties,
                  encoding_fn: EncodingModule = None, recovery_fn: RecoveryModule = None,
                  operations_static_fn: dict[str, Tuple[OperationTargetModule, OperationPredictionModule]] = None,
                  operations_strictly_transversal_fn: dict[str, Tuple[OperationTargetModule, OperationPredictionModule]] = None,
                  operations_transversal_fn: dict[str, Tuple[OperationTargetModule, OperationPredictionModule]] = None,
                  operations_weakly_transversal_fn: dict[str, Tuple[OperationTargetModule, OperationPredictionModule]] = None,
                  operations_non_transversal_fn: dict[str, Tuple[OperationTargetModule, OperationPredictionModule]] = None,
                  normalize: bool = True):
    """Store QASM circuits, properties, and trained parameters to disk.

    Depending on which modules are provided, this function:

    * exports the learned encoding, recovery, and operation circuits to QASM
      files under ``path`` (with suitable suffixes),
    * stores encoding/recovery/operation properties as ``*.pkl`` files, and
    * stores trained parameters (NumPy arrays) as ``*_parameters.pkl`` files
      for later reuse.

    For parameterized circuits, the ``normalize`` flag controls whether
    parameters are wrapped modulo a symmetry (e.g. ``4π`` for U3-type gates)
    before being converted to QASM, which can help avoid large equivalent
    angles.

    Args:
        path (str): Path to the experiment results folder where files will be
            written.
        code_properties (CodeProperties): Code configuration containing
            encoding, recovery, and operation properties.
        training_properties (TrainingProperties): Training configuration and
            parameter containers (encoding, recovery, operations).
        encoding_fn (EncodingModule | None): Encoding module. If not ``None``,
            its QASM circuit and, if trainable, encoding properties and
            parameters are stored.
        recovery_fn (RecoveryModule | None): Recovery module. If not
            ``None``, its QASM circuit and, if trainable, recovery properties
            and parameters are stored.
        operations_static_fn (dict[str, tuple[OperationTargetModule, OperationPredictionModule]] | None):
            Static operation modules. For each operation, the prediction
            circuit is exported as ``<op>_static.qasm``.
        operations_strictly_transversal_fn (dict[str, tuple[OperationTargetModule, OperationPredictionModule]] | None):
            Strictly transversal operation modules; each stored as
            ``<op>_strictly_transversal.qasm``.
        operations_transversal_fn (dict[str, tuple[OperationTargetModule, OperationPredictionModule]] | None):
            Transversal operation modules. For each operation, QASM is stored
            as ``<op>_transversal.qasm``, properties as
            ``<op>_transversal_properties.pkl``, and parameters as
            ``<op>_transversal_parameters.pkl``.
        operations_weakly_transversal_fn (dict[str, tuple[OperationTargetModule, OperationPredictionModule]] | None):
            Weakly transversal operation modules; similarly stored with
            ``_weakly_transversal`` suffixes.
        operations_non_transversal_fn (dict[str, tuple[OperationTargetModule, OperationPredictionModule]] | None):
            Non-transversal operation modules; stored with
            ``_non_transversal`` suffixes and separate initial/block
            parameter arrays.
        normalize (bool): If ``True``, normalize parameters before exporting
            QASM (to account for parameter periodicity). If ``False``, use
            raw parameter values.

    Returns:
        None
    """

    # store resulting fixed ansatz in QASM3 format
    # Note: minor non-significant deviations are possible due to accuracy of storing parameters; if there are large
    # deviations, check if the normalization symmetry is set correctly for the employed gates (or deactivate it)
    if encoding_fn is not None:
        qasm = encoding_fn.get_qasm_encoding(parameters_encoding=training_properties.parameters_encoding,
                                             normalize=normalize)
        file_path = os.path.join(path, f'encoding.qasm')
        with open(file_path, 'w') as ff:
            ff.write(qasm)
        print(f'Stored encoding file to {file_path}.')

    if recovery_fn is not None:
        qasm = recovery_fn.get_qasm_recovery(parameters_recovery=training_properties.parameters_recovery,
                                             normalize=normalize)
        file_path = os.path.join(path, f'recovery.qasm')
        with open(file_path, 'w') as ff:
            ff.write(qasm)
        print(f'Stored recovery file to {file_path}.')

    if operations_static_fn is not None:
        for operation_id in operations_static_fn:
            qasm = operations_static_fn.get(operation_id)[1].get_qasm_operation()
            file_path = os.path.join(path, f'{operation_id}_static.qasm')
            with open(file_path, 'w') as ff:
                ff.write(qasm)
            print(f'Stored `{operation_id}` static operation file to {file_path}.')

    if operations_strictly_transversal_fn is not None:
        for operation_id in operations_strictly_transversal_fn:
            qasm = operations_strictly_transversal_fn.get(operation_id)[1].get_qasm_operation()
            file_path = os.path.join(path, f'{operation_id}_strictly_transversal.qasm')
            with open(file_path, 'w') as ff:
                ff.write(qasm)
            print(f'Stored `{operation_id}` strictly-transversal operation file to {file_path}.')

    if operations_transversal_fn is not None:
        for operation_id in operations_transversal_fn:
            qasm = operations_transversal_fn.get(operation_id)[1].get_qasm_operation(training_properties
                                                                                     .parameters_operation
                                                                                     .parameters_operation_transversal
                                                                                     .get(operation_id))
            file_path = os.path.join(path, f'{operation_id}_transversal.qasm')
            with open(file_path, 'w') as ff:
                ff.write(qasm)
            print(f'Stored `{operation_id}` transversal operation file to {file_path}.')

    if operations_weakly_transversal_fn is not None:
        for operation_id in operations_weakly_transversal_fn:
            qasm = operations_weakly_transversal_fn.get(operation_id)[1].get_qasm_operation(training_properties
                                                                                            .parameters_operation
                                                                                            .parameters_operation_weakly_transversal
                                                                                            .get(operation_id))
            file_path = os.path.join(path, f'{operation_id}_weakly_transversal.qasm')
            with open(file_path, 'w') as ff:
                ff.write(qasm)
            print(f'Stored `{operation_id}` weakly-transversal operation file to {file_path}.')

    if operations_non_transversal_fn is not None:
        for operation_id in operations_non_transversal_fn:
            qasm = operations_non_transversal_fn.get(operation_id)[1].get_qasm_operation(training_properties
                                                                                         .parameters_operation
                                                                                         .parameters_operation_non_transversal
                                                                                         .get(operation_id))
            file_path = os.path.join(path, f'{operation_id}_non_transversal.qasm')
            with open(file_path, 'w') as ff:
                ff.write(qasm)
            print(f'Stored `{operation_id}` non-transversal operation file to {file_path}.')

    # store properties (encoding / recovery / parameterized operations) for loading trainable model
    if encoding_fn is not None:
        encoding_properties = code_properties.encoding_properties
        if encoding_properties.trainable:
            file_path = os.path.join(path, 'encoding_properties.pkl')
            with open(file_path, 'wb') as ff:
                pickle.dump(encoding_properties, ff)  # noqa
            print(f'Stored encoding properties file to {file_path}.')

            # store trained encoding parameters
            encoding_parameters = (training_properties.parameters_encoding.parameters_encoding_initial.detach().numpy(),
                                   training_properties.parameters_encoding.parameters_encoding_block.detach().numpy())
            file_path = os.path.join(path, 'encoding_parameters.pkl')
            with open(file_path, 'wb') as ff:
                pickle.dump(encoding_parameters, ff)  # noqa
            print(f'Stored encoding parameters file to {file_path}.')

    if recovery_fn is not None:
        recovery_properties = code_properties.recovery_properties
        if recovery_properties.trainable:
            file_path = os.path.join(path, 'recovery_properties.pkl')
            with open(file_path, 'wb') as ff:
                pickle.dump(recovery_properties, ff)  # noqa
            print(f'Stored recovery properties file to {file_path}.')

            # store trained recovery parameters
            file_path = os.path.join(path, 'recovery_parameters.pkl')
            recovery_parameters = (training_properties.parameters_recovery.parameters_recovery_initial.detach().numpy(),
                                   training_properties.parameters_recovery.parameters_recovery_block.detach().numpy())
            with open(file_path, 'wb') as ff:
                pickle.dump(recovery_parameters, ff)  # noqa
            print(f'Stored recovery parameters file to {file_path}.')

    if operations_transversal_fn is not None:
        for operation_id in operations_transversal_fn:
            operation_properties = code_properties.operation_transversal.get(operation_id)
            file_path = os.path.join(path, f'{operation_id}_transversal_properties.pkl')
            with open(file_path, 'wb') as ff:
                pickle.dump(operation_properties, ff)  # noqa
            print(f'Stored transversal {operation_id} operation properties file to {file_path}.')

            # store trained operation parameters
            file_path = os.path.join(path, f'{operation_id}_transversal_parameters.pkl')
            operation_parameters = (training_properties.parameters_operation.parameters_operation_transversal
                                    .get(operation_id).parameters_operation.detach().numpy())
            with open(file_path, 'wb') as ff:
                pickle.dump(operation_parameters, ff)  # noqa
            print(f'Stored transversal {operation_id} operation parameters file to {file_path}.')

    if operations_weakly_transversal_fn is not None:
        for operation_id in operations_weakly_transversal_fn:
            operation_properties = code_properties.operation_weakly_transversal.get(operation_id)
            file_path = os.path.join(path, f'{operation_id}_weakly_transversal_properties.pkl')
            with open(file_path, 'wb') as ff:
                pickle.dump(operation_properties, ff)  # noqa
            print(f'Stored weakly-transversal {operation_id} operation properties file to {file_path}.')

            # store trained operation parameters
            file_path = os.path.join(path, f'{operation_id}_weakly_transversal_parameters.pkl')
            operation_parameters = (training_properties.parameters_operation.parameters_operation_weakly_transversal
                                    .get(operation_id).parameters_operation.detach().numpy())
            with open(file_path, 'wb') as ff:
                pickle.dump(operation_parameters, ff)  # noqa
            print(f'Stored weakly-transversal {operation_id} operation parameters file to {file_path}.')

    if operations_non_transversal_fn is not None:
        for operation_id in operations_non_transversal_fn:
            operation_properties = code_properties.operation_non_transversal.get(operation_id)
            file_path = os.path.join(path, f'{operation_id}_non_transversal_properties.pkl')
            with open(file_path, 'wb') as ff:
                pickle.dump(operation_properties, ff)  # noqa
            print(f'Stored non-transversal {operation_id} operation properties file to {file_path}.')

            # store trained operation parameters
            file_path = os.path.join(path, f'{operation_id}_non_transversal_parameters.pkl')
            operation_parameters = (training_properties.parameters_operation.parameters_operation_non_transversal
                                    .get(operation_id).parameters_operation_initial.detach().numpy(),
                                    training_properties.parameters_operation.parameters_operation_non_transversal
                                    .get(operation_id).parameters_operation_block.detach().numpy())
            with open(file_path, 'wb') as ff:
                pickle.dump(operation_parameters, ff)  # noqa
            print(f'Stored non-transversal {operation_id} operation parameters file to {file_path}.')
