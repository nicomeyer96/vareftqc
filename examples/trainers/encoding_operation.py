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

"""Training routines for encoding and logical operations in VarEFTQC.

This module implements the high-level training loop for the encoding ansatz
and (optional) logical operations (static, strictly transversal, transversal,
weakly transversal, and non-transversal). It provides:

* the main training function using an L-BFGS optimizer,
* helper functions to test encoding and operations, and
* a helper to run a full test suite and log results.
"""

import torch
from typing import Tuple

from vareftqc.helpers import CodeProperties, TrainingProperties
from vareftqc import PhysicalModule, EncodingModule, OperationTargetModule, OperationPredictionModule
from vareftqc.loss_functions import distinguishability_loss, operation_loss
from .logger import Logger


def train_encoding_operation(logger: Logger, code_properties: CodeProperties, training_properties: TrainingProperties,
                             encoding_fn: EncodingModule, noisefree_fn: PhysicalModule, baseline_fn: PhysicalModule,
                             operations_static_fn: dict[str, Tuple[OperationTargetModule, OperationPredictionModule]],
                             operations_strictly_transversal_fn: dict[str, Tuple[OperationTargetModule, OperationPredictionModule]],
                             operations_transversal_fn: dict[str, Tuple[OperationTargetModule, OperationPredictionModule]],
                             operations_weakly_transversal_fn: dict[str, Tuple[OperationTargetModule, OperationPredictionModule]],
                             operations_non_transversal_fn: dict[str, Tuple[OperationTargetModule, OperationPredictionModule]]):
    """Train the encoding ansatz and (optionally) logical operations.

        This function:

        * computes a noise-free reference using a physical model,
        * performs an initial test pass (encoding + operations),
        * optionally initializes and runs an L-BFGS optimizer over all
          trainable parameters (encoding and operations),
        * performs validation during training (for encoding), and
        * performs final tests and logging after training.

        Args:
            logger (Logger): Logger instance used to record training, validation,
                test, and baseline losses.
            code_properties (CodeProperties): Code configuration including the
                encoding, operations, and flags indicating what is trainable.
            training_properties (TrainingProperties): Training configuration and
                parameter containers.
            encoding_fn (EncodingModule): Module implementing the variational
                encoding circuit.
            noisefree_fn (PhysicalModule): Noise-free physical model used as
                target for encoding loss.
            baseline_fn (PhysicalModule): Noisy physical model used as baseline
                for comparison.
            operations_static_fn (dict[str, tuple[OperationTargetModule, OperationPredictionModule]]):
                Mapping from operation IDs to (target, prediction) modules for
                static logical operations.
            operations_strictly_transversal_fn (dict[str, tuple[OperationTargetModule, OperationPredictionModule]]):
                Mapping for strictly transversal logical operations.
            operations_transversal_fn (dict[str, tuple[OperationTargetModule, OperationPredictionModule]]):
                Mapping for trainable transversal logical operations.
            operations_weakly_transversal_fn (dict[str, tuple[OperationTargetModule, OperationPredictionModule]]):
                Mapping for trainable weakly transversal logical operations.
            operations_non_transversal_fn (dict[str, tuple[OperationTargetModule, OperationPredictionModule]]):
                Mapping for trainable non-transversal logical operations.

        Returns:
            None
        """

    # pre-compute encoding static states for later use (training)
    noisefree = noisefree_fn.run()

    ##################
    # initial testing
    ##################
    run_tests(logger=logger, training_properties=training_properties, code_properties=code_properties,
              encoding_fn=encoding_fn, noisefree_fn=noisefree_fn, baseline_fn=baseline_fn,
              operations_static_fn=operations_static_fn,
              operations_strictly_transversal_fn=operations_strictly_transversal_fn,
              operations_transversal_fn=operations_transversal_fn,
              operations_weakly_transversal_fn=operations_weakly_transversal_fn,
              operations_non_transversal_fn=operations_non_transversal_fn,
              epoch=1)

    # log initial encoding on two-design states, might be overwritten by training (guarantees log if no training performed)
    # skip if encoding is static to make training operations feasible for large codes (avoids out-of-memory issues)
    if code_properties.train_encoding or 0 == len(operations_static_fn) + len(operations_strictly_transversal_fn) + len(operations_transversal_fn):
        encoding_initial = encoding_fn.run(parameters_encoding=training_properties.parameters_encoding)
        loss_avg_encoding_initial, loss_max_encoding_initial = distinguishability_loss(prediction=encoding_initial,
                                                                                       groundtruth=noisefree)
        logger.logger_encoding.log_train(loss_avg_encoding_initial, loss_max_encoding_initial, info={'epoch': 1, 'loop': 1})
        logger.logger_encoding.finish_epoch()

    ##################
    # set up training
    ##################

    # perform training if encoding is trainable or/and transversal operations are to be trained
    if training_properties.epochs > 0 and (code_properties.train_encoding
                                           or len(operations_transversal_fn) + len(operations_weakly_transversal_fn)
                                           + len(operations_non_transversal_fn) > 0):
        # initial validation
        if training_properties.num_validation_states is not None and code_properties.train_encoding:
            loss_avg_, loss_max_ = test_encoding(training_properties, encoding_fn, noisefree_fn,
                                                 ep=0, number_states=training_properties.num_validation_states,
                                                 instance_states=training_properties.instance_validation_states,
                                                 display='VALIDATE')
            logger.logger_encoding.log_validation(loss_avg_, loss_max_,
                                                  info={'epoch': 0, 'num_states': training_properties.num_validation_states})

        ###################
        # set up optimizer
        ###################
        trainable_parameters = []
        if code_properties.encoding_properties.trainable:
            trainable_parameters.extend(training_properties.parameters_encoding.parameters())
        if len(operations_transversal_fn) > 0:
            trainable_parameters.extend(training_properties.parameters_operation.parameters_transversal())
        if len(operations_weakly_transversal_fn) > 0:
            trainable_parameters.extend(training_properties.parameters_operation.parameters_weakly_transversal())
        if len(operations_non_transversal_fn) > 0:
            trainable_parameters.extend(training_properties.parameters_operation.parameters_non_transversal())
        if 0 == len(trainable_parameters):
            raise RuntimeError('No trainable parameters.')
        optimizer = torch.optim.LBFGS(trainable_parameters,
                                      lr=training_properties.learning_rate,
                                      max_iter=training_properties.max_iter,
                                      history_size=training_properties.history_size)

        # mutable counter for inner loops of L-BFGS optimizer
        loop = [0]

        #######################
        # closure for training
        #######################
        def closure():
            optimizer.zero_grad()
            loop[0] += 1

            #####################################
            # forward pass and loss for encoding
            #####################################
            # Note: If encoding not trainable, only validated above
            if code_properties.encoding_properties.trainable:
                encoding = encoding_fn.run(parameters_encoding=training_properties.parameters_encoding)
                loss_avg_encoding, loss_max_encoding = distinguishability_loss(prediction=encoding,
                                                                               groundtruth=noisefree,
                                                                               display=f'TRAIN #{epoch}/'
                                                                                       f'{training_properties.epochs} '
                                                                                       f'(loop #{loop[0]}) Encoding')
                logger.logger_encoding.log_train(loss_avg_encoding, loss_max_encoding, info={'epoch': epoch,
                                                                                             'loop': loop[0]})

            #######################################
            # forward pass and loss for operations
            #######################################
            losses_avg_operation, losses_max_operation = [], []

            # Static operations
            for operation_id in operations_static_fn:
                target = operations_static_fn.get(operation_id)[0].run(parameters_encoding=training_properties.parameters_encoding)
                prediction = operations_static_fn.get(operation_id)[1].run(parameters_encoding=training_properties.parameters_encoding)
                # compute loss
                loss_avg, loss_max, loss_log = operation_loss(prediction=prediction, target=target,
                                                              order_operation=code_properties.
                                                              operation_static.get(operation_id).order,
                                                              method=training_properties.operation_loss,
                                                              display=f'TRAIN #{epoch}/{training_properties.epochs} '
                                                                      f'(loop #{loop[0]}) Static '
                                                                      f'Operation `{operation_id}`')
                logger.logger_operation_static.get(operation_id).log_train(loss_avg, loss_max,
                                                                           info={'epoch': epoch, 'loop': loop[0],
                                                                                 'avg_diag': loss_log.get('avg_diag', None),
                                                                                 'max_diag': loss_log.get('max_diag', None)})
                losses_avg_operation.append(loss_avg)
                losses_max_operation.append(loss_max)

            # Strictly-transversal operations
            for operation_id in operations_strictly_transversal_fn:
                target = operations_strictly_transversal_fn.get(operation_id)[0].run(parameters_encoding=training_properties.parameters_encoding)
                prediction = operations_strictly_transversal_fn.get(operation_id)[1].run(parameters_encoding=training_properties.parameters_encoding)
                # compute loss
                loss_avg, loss_max, loss_log = operation_loss(prediction=prediction, target=target,
                                                              order_operation=code_properties.
                                                              operation_strictly_transversal.get(operation_id).order,
                                                              method=training_properties.operation_loss,
                                                              display=f'TRAIN #{epoch}/{training_properties.epochs} '
                                                                      f'(loop #{loop[0]}) Strictly-Transversal '
                                                                      f'Operation `{operation_id}`')
                logger.logger_operation_strictly_transversal.get(operation_id).log_train(loss_avg, loss_max,
                                                                                         info={'epoch': epoch, 'loop': loop[0],
                                                                                               'avg_diag': loss_log.get('avg_diag', None),
                                                                                               'max_diag': loss_log.get('max_diag', None)})
                losses_avg_operation.append(loss_avg)
                losses_max_operation.append(loss_max)

            # Transversal operations
            for operation_id in operations_transversal_fn:  # compute target and prediction
                target = operations_transversal_fn.get(operation_id)[0].run(parameters_encoding=training_properties.parameters_encoding)
                prediction = operations_transversal_fn.get(operation_id)[1].run(parameters_encoding=training_properties.parameters_encoding,
                                                                                parameters_operation=training_properties.parameters_operation.
                                                                                parameters_operation_transversal.get(operation_id))
                # compute loss
                loss_avg, loss_max, loss_log = operation_loss(prediction=prediction, target=target,
                                                              order_operation=code_properties.
                                                              operation_transversal.get(operation_id).order,
                                                              method=training_properties.operation_loss,
                                                              display=f'TRAIN #{epoch}/{training_properties.epochs} '
                                                                      f'(loop #{loop[0]}) Transversal '
                                                                      f'Operation `{operation_id}`')
                logger.logger_operation_transversal.get(operation_id).log_train(loss_avg, loss_max,
                                                                                info={'epoch': epoch, 'loop': loop[0],
                                                                                      'avg_diag': loss_log.get('avg_diag', None),
                                                                                      'max_diag': loss_log.get('max_diag', None)})
                losses_avg_operation.append(loss_avg)
                losses_max_operation.append(loss_max)

            # Weakly-ransversal operations
            for operation_id in operations_weakly_transversal_fn:  # compute target and prediction
                target = operations_weakly_transversal_fn.get(operation_id)[0].run(parameters_encoding=training_properties.parameters_encoding)
                prediction = operations_weakly_transversal_fn.get(operation_id)[1].run(parameters_encoding=training_properties.parameters_encoding,
                                                                                       parameters_operation=training_properties.parameters_operation.
                                                                                       parameters_operation_weakly_transversal.get(operation_id))
                # compute loss
                loss_avg, loss_max, loss_log = operation_loss(prediction=prediction, target=target,
                                                              order_operation=code_properties.
                                                              operation_weakly_transversal.get(operation_id).order,
                                                              method=training_properties.operation_loss,
                                                              display=f'TRAIN #{epoch}/{training_properties.epochs} '
                                                                      f'(loop #{loop[0]}) Weakly-Transversal '
                                                                      f'Operation `{operation_id}`')
                logger.logger_operation_weakly_transversal.get(operation_id).log_train(loss_avg, loss_max,
                                                                                       info={'epoch': epoch, 'loop': loop[0],
                                                                                             'avg_diag': loss_log.get('avg_diag', None),
                                                                                             'max_diag': loss_log.get('max_diag', None)})
                losses_avg_operation.append(loss_avg)
                losses_max_operation.append(loss_max)

            # Non-Transversal operations
            for operation_id in operations_non_transversal_fn:  # compute target and prediction
                target = operations_non_transversal_fn.get(operation_id)[0].run(parameters_encoding=training_properties.parameters_encoding)
                prediction = operations_non_transversal_fn.get(operation_id)[1].run(parameters_encoding=training_properties.parameters_encoding,
                                                                                    parameters_operation=training_properties.parameters_operation.
                                                                                    parameters_operation_non_transversal.get(operation_id))
                # compute loss
                loss_avg, loss_max, loss_log = operation_loss(prediction=prediction, target=target,
                                                              order_operation=code_properties.
                                                              operation_non_transversal.get(operation_id).order,
                                                              method=training_properties.operation_loss,
                                                              display=f'TRAIN #{epoch}/{training_properties.epochs} '
                                                                      f'(loop #{loop[0]}) Non-Transversal '
                                                                      f'Operation `{operation_id}`')
                logger.logger_operation_non_transversal.get(operation_id).log_train(loss_avg, loss_max,
                                                                                    info={'epoch': epoch, 'loop': loop[0],
                                                                                          'avg_diag': loss_log.get('avg_diag', None),
                                                                                          'max_diag': loss_log.get('max_diag', None)})
                losses_avg_operation.append(loss_avg)
                losses_max_operation.append(loss_max)

            # stack operation loss values, sum up (if operation loss values exist)
            loss_avg_operation, loss_max_operation = None, None
            if len(losses_avg_operation) > 0:
                loss_avg_operation = torch.sum(torch.stack(losses_avg_operation, dim=0), dim=0)
                loss_max_operation = torch.sum(torch.stack(losses_max_operation, dim=0), dim=0)

            ########################
            # compose loss function
            ########################
            if code_properties.encoding_properties.trainable and 0 == len(losses_avg_operation):
                # only train encoding loss
                loss_avg = loss_avg_encoding  # noqa
                loss_max = loss_max_encoding  # noqa
            elif code_properties.encoding_properties.trainable:
                # encoding loss, operation losses for static, strictly-transversal, transversal, weakly-transversal and non-transversal operations
                loss_avg = loss_avg_encoding + training_properties.operation_loss_regularize * loss_avg_operation  # noqa
                loss_max = loss_max_encoding + training_properties.operation_loss_regularize * loss_max_operation  # noqa
                logger.logger_operations_sum.log_train(loss_avg_operation, loss_max_operation,
                                                       info={'epoch': epoch, 'loop': loop[0]})
                logger.logger_encoding_operations_sum.log_train(loss_avg, loss_max, info={'epoch': epoch, 'loop': loop[0],
                                                                                          'regularize': training_properties.operation_loss_regularize})
                print(f'TRAIN #{epoch}/{training_properties.epochs} (loop #{loop[0]}) Encoding + Operations >>> '
                      f'AVG: {loss_avg.detach():.7f} | MAX: {loss_max.detach():.7f} [d-loss + {'' 
                      if 1.0 == training_properties.operation_loss_regularize 
                      else f'{training_properties.operation_loss_regularize} * '}Σ o-loss]')
            else:
                # only train operation losses (still log both)
                loss_avg = loss_avg_operation
                loss_max = loss_max_operation
                logger.logger_operations_sum.log_train(loss_avg, loss_max, info={'epoch': epoch, 'loop': loop[0]})
                if code_properties.encoding_properties.trainable:
                    logger.logger_encoding_operations_sum.log_train(loss_avg_encoding_initial + training_properties.operation_loss_regularize * loss_avg,
                                                                    loss_max_encoding_initial + training_properties.operation_loss_regularize * loss_max,
                                                                    info={'epoch': epoch, 'loop': loop[0],
                                                                          'regularize': training_properties.operation_loss_regularize})
                print(f'TRAIN #{epoch}/{training_properties.epochs} (loop #{loop[0]}) Operations >>> '
                      f'AVG: {loss_avg.detach():.7f} | MAX: {loss_max.detach():.7f} [Σ o-loss]')
            # backward pass
            if 'max' == training_properties.encoding_loss:
                loss_max.backward()
                return loss_max
            loss_avg.backward()
            return loss_avg

        ##########################
        # perform actual training
        ##########################
        for epoch in range(1, training_properties.epochs + 1):
            # perform one optimization epoch (might perform several internal steps)
            loop[0] = 0
            optimizer.step(closure)

            # validation
            if training_properties.num_validation_states is not None and code_properties.train_encoding:
                loss_avg_, loss_max_ = test_encoding(training_properties, encoding_fn, noisefree_fn,
                                                     ep=epoch, number_states=training_properties.num_validation_states,
                                                     instance_states=training_properties.instance_validation_states,
                                                     display='VALIDATE')
                logger.logger_encoding.log_validation(loss_avg_, loss_max_,
                                                      info={'epoch': epoch,
                                                            'num_states': training_properties.num_validation_states})

            # finish logging of epoch
            if code_properties.train_encoding:
                logger.logger_encoding.finish_epoch()
            for operation_id_ in operations_static_fn:
                logger.logger_operation_static.get(operation_id_).finish_epoch()
            for operation_id_ in operations_strictly_transversal_fn:
                logger.logger_operation_strictly_transversal.get(operation_id_).finish_epoch()
            for operation_id_ in operations_transversal_fn:
                logger.logger_operation_transversal.get(operation_id_).finish_epoch()
            for operation_id_ in operations_weakly_transversal_fn:
                logger.logger_operation_weakly_transversal.get(operation_id_).finish_epoch()
            for operation_id_ in operations_non_transversal_fn:
                logger.logger_operation_non_transversal.get(operation_id_).finish_epoch()
            if logger.logger_operations_sum is not None:
                logger.logger_operations_sum.finish_epoch()
            if logger.logger_encoding_operations_sum is not None:
                logger.logger_encoding_operations_sum.finish_epoch()

    ################
    # final testing
    ################
    run_tests(logger=logger, training_properties=training_properties, code_properties=code_properties,
              encoding_fn=encoding_fn, noisefree_fn=noisefree_fn, baseline_fn=baseline_fn,
              operations_static_fn=operations_static_fn,
              operations_strictly_transversal_fn=operations_strictly_transversal_fn,
              operations_transversal_fn=operations_transversal_fn,
              operations_weakly_transversal_fn=operations_weakly_transversal_fn,
              operations_non_transversal_fn=operations_non_transversal_fn,
              epoch=training_properties.epochs + 1)

    # test encoding on two-design states (guarantees log after last training step)
    # skip if encoding is static to make training operations feasible for large codes (avoids out-of-memory issues)
    if code_properties.train_encoding or 0 == len(operations_static_fn) + len(operations_strictly_transversal_fn) + len(operations_transversal_fn):
        encoding_final = encoding_fn.run(parameters_encoding=training_properties.parameters_encoding)
        loss_avg_encoding_final, loss_max_encoding_final = distinguishability_loss(prediction=encoding_final,
                                                                                   groundtruth=noisefree)
        logger.logger_encoding.log_train(loss_avg_encoding_final, loss_max_encoding_final,
                                         info={'epoch': training_properties.epochs + 1, 'loop': 1})
        logger.logger_encoding.finish_epoch()


def test_encoding(training_properties: TrainingProperties, encoding_fn: EncodingModule, noisefree_fn: PhysicalModule,
                  ep: int = None, number_states: int = 0, instance_states: int = None, display: str = None):
    """Evaluate the encoding ansatz against a noise-free reference.

    The encoding is tested either on a set of Haar-random input states
    (``number_states > 0``) or on a two-design (``number_states == 0``),
    and compared to the corresponding noise-free physical evolution.

    Args:
        training_properties (TrainingProperties): Training configuration,
            providing access to the encoding parameters.
        encoding_fn (EncodingModule): Variational encoding module to test.
        noisefree_fn (PhysicalModule): Noise-free reference model.
        ep (int | None): Optional epoch index used only for display/logging.
        number_states (int): Number of Haar-random states to test on. If
            ``0``, two-design states are used instead.
        instance_states (int | None): Seed/instance index for state sampling.
        display (str | None): Optional prefix for the display string passed
            to the loss function.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Average and maximum
        distinguishability losses (scalars).
    """

    if display is not None:
        description = f'{number_states} Haar-random states' if number_states > 0 else 'Two-design states'
        display = f'{display} {'' if ep is None else f'#{ep} '}Encoding ({description})'
    with torch.no_grad():
        encoding_test = encoding_fn.run(number_states=number_states, seed_states=instance_states,
                                        parameters_encoding=training_properties.parameters_encoding)
        noisefree_test = noisefree_fn.run(number_states=number_states, seed_states=instance_states)
    loss_avg, loss_max = distinguishability_loss(prediction=encoding_test, groundtruth=noisefree_test,
                                                 display=display)
    return loss_avg, loss_max


def baseline_encoding(baseline_fn: PhysicalModule, noisefree_fn: PhysicalModule,
                      number_states: int = 0, instance_states: int = None):
    """Compute an encoding baseline using a noisy physical model.

    This compares the noisy physical evolution to the noise-free reference
    for either Haar-random states or two-design states, and serves as a
    baseline for the variational encoding performance.

    Args:
        baseline_fn (PhysicalModule): Noisy physical model used as baseline.
        noisefree_fn (PhysicalModule): Noise-free reference model.
        number_states (int): Number of Haar-random states to test on. If
            ``0``, two-design states are used instead.
        instance_states (int | None): Seed/instance index for state sampling.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Average and maximum
        distinguishability losses (scalars).
    """

    description = f'{number_states} Haar-random states' if number_states > 0 else 'Two-design states'
    with torch.no_grad():
        baseline_test = baseline_fn.run(number_states=number_states, seed_states=instance_states)
        noisefree_test = noisefree_fn.run(number_states=number_states, seed_states=instance_states)
    loss_avg, loss_max = distinguishability_loss(prediction=baseline_test, groundtruth=noisefree_test,
                                                 display=f'TEST Baseline ({description})')
    return loss_avg, loss_max


def test_operation_static(training_properties: TrainingProperties, code_properties: CodeProperties,
                          operations_static_fn: dict[str, Tuple[OperationTargetModule, OperationPredictionModule]],
                          operation_id: str):
    """Test a static logical operation.

    The static operation is implemented by two circuits: a target circuit and
    a prediction circuit. The prediction is compared to the target using the
    selected operation loss.

    Args:
        training_properties (TrainingProperties): Training configuration,
            providing access to encoding parameters.
        code_properties (CodeProperties): Code configuration including the
            static operation metadata.
        operations_static_fn (dict[str, tuple[OperationTargetModule, OperationPredictionModule]]):
            Mapping from operation IDs to (target, prediction) modules.
        operation_id (str): Identifier of the operation to test (key in
            ``operations_static_fn`` and ``code_properties.operation_static``).

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
            Average and maximum operation losses, and (optionally) average
            and maximum diagonal losses if provided by the loss function.
    """

    with torch.no_grad():
        target = operations_static_fn.get(operation_id)[0].run(parameters_encoding=training_properties.parameters_encoding)
        prediction = operations_static_fn.get(operation_id)[1].run(parameters_encoding=training_properties.parameters_encoding)
    loss_avg, loss_max, loss_log = operation_loss(prediction=prediction, target=target,
                                                  order_operation=code_properties.operation_static.get(operation_id).order,
                                                  method=training_properties.operation_loss,
                                                  display=f'TEST Static Operation `{operation_id}`')
    return loss_avg, loss_max, loss_log.get('avg_diag'), loss_log.get('max_diag')


def test_operation_strictly_transversal(training_properties: TrainingProperties, code_properties: CodeProperties,
                                        operations_strictly_transversal_fn: dict[str, Tuple[OperationTargetModule, OperationPredictionModule]],
                                        operation_id: str):
    """Test a strictly transversal logical operation.

    Args:
        training_properties (TrainingProperties): Training configuration,
            providing access to encoding parameters.
        code_properties (CodeProperties): Code configuration including the
            strictly transversal operation metadata.
        operations_strictly_transversal_fn (dict[str, tuple[OperationTargetModule, OperationPredictionModule]]):
            Mapping from operation IDs to (target, prediction) modules.
        operation_id (str): Identifier of the operation to test.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
            Average and maximum operation losses, and (optionally) average
            and maximum diagonal losses if provided by the loss function.
    """

    with torch.no_grad():
        target = operations_strictly_transversal_fn.get(operation_id)[0].run(parameters_encoding=training_properties.parameters_encoding)
        prediction = operations_strictly_transversal_fn.get(operation_id)[1].run(parameters_encoding=training_properties.parameters_encoding)
    loss_avg, loss_max, loss_log = operation_loss(prediction=prediction, target=target,
                                                  order_operation=code_properties.operation_strictly_transversal.get(operation_id).order,
                                                  method=training_properties.operation_loss,
                                                  display=f'TEST Strictly-Transversal Operation `{operation_id}`')
    return loss_avg, loss_max, loss_log.get('avg_diag'), loss_log.get('max_diag')


def test_operation_transversal(training_properties: TrainingProperties, code_properties: CodeProperties,
                               operations_transversal_fn: dict[str, Tuple[OperationTargetModule, OperationPredictionModule]],
                               operation_id: str):
    """Test a trainable transversal logical operation.

    The prediction circuit uses trainable transversal-operation parameters,
    while the target circuit is fixed.

    Args:
        training_properties (TrainingProperties): Training configuration,
            providing access to encoding and transversal-operation parameters.
        code_properties (CodeProperties): Code configuration including the
            transversal operation metadata.
        operations_transversal_fn (dict[str, tuple[OperationTargetModule, OperationPredictionModule]]):
            Mapping from operation IDs to (target, prediction) modules.
        operation_id (str): Identifier of the operation to test.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
            Average and maximum operation losses, and (optionally) average
            and maximum diagonal losses if provided by the loss function.
    """

    with torch.no_grad():
        target = operations_transversal_fn.get(operation_id)[0].run(parameters_encoding=training_properties.parameters_encoding)
        prediction = operations_transversal_fn.get(operation_id)[1].run(parameters_encoding=training_properties.parameters_encoding,
                                                                        parameters_operation=training_properties.parameters_operation.
                                                                        parameters_operation_transversal.get(operation_id))
    loss_avg, loss_max, loss_log = operation_loss(prediction=prediction, target=target,
                                                  order_operation=code_properties.operation_transversal.get(operation_id).order,
                                                  method=training_properties.operation_loss,
                                                  display=f'TEST Transversal Operation `{operation_id}`')
    return loss_avg, loss_max, loss_log.get('avg_diag'), loss_log.get('max_diag')


def test_operation_weakly_transversal(training_properties: TrainingProperties, code_properties: CodeProperties,
                                      operations_weakly_transversal_fn: dict[str, Tuple[OperationTargetModule, OperationPredictionModule]],
                                      operation_id: str):
    """Test a trainable weakly-transversal logical operation.

    Args:
        training_properties (TrainingProperties): Training configuration,
            providing access to encoding and weakly-transversal-operation
            parameters.
        code_properties (CodeProperties): Code configuration including the
            weakly transversal operation metadata.
        operations_weakly_transversal_fn (dict[str, tuple[OperationTargetModule, OperationPredictionModule]]):
            Mapping from operation IDs to (target, prediction) modules.
        operation_id (str): Identifier of the operation to test.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
            Average and maximum operation losses, and (optionally) average
            and maximum diagonal losses if provided by the loss function.
    """

    with torch.no_grad():
        target = operations_weakly_transversal_fn.get(operation_id)[0].run(parameters_encoding=training_properties.parameters_encoding)
        prediction = operations_weakly_transversal_fn.get(operation_id)[1].run(parameters_encoding=training_properties.parameters_encoding,
                                                                               parameters_operation=training_properties.parameters_operation.
                                                                               parameters_operation_weakly_transversal.get(operation_id))
    loss_avg, loss_max, loss_log = operation_loss(prediction=prediction, target=target,
                                                  order_operation=code_properties.operation_weakly_transversal.get(operation_id).order,
                                                  method=training_properties.operation_loss,
                                                  display=f'TEST Transversal Operation `{operation_id}`')
    return loss_avg, loss_max, loss_log.get('avg_diag'), loss_log.get('max_diag')


def test_operation_non_transversal(training_properties: TrainingProperties, code_properties: CodeProperties,
                                   operations_non_transversal_fn: dict[str, Tuple[OperationTargetModule, OperationPredictionModule]],
                                   operation_id: str):
    """Test a trainable non-transversal logical operation.

    Args:
        training_properties (TrainingProperties): Training configuration,
            providing access to encoding and non-transversal-operation
            parameters.
        code_properties (CodeProperties): Code configuration including the
            non-transversal operation metadata.
        operations_non_transversal_fn (dict[str, tuple[OperationTargetModule, OperationPredictionModule]]):
            Mapping from operation IDs to (target, prediction) modules.
        operation_id (str): Identifier of the operation to test.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
            Average and maximum operation losses, and (optionally) average
            and maximum diagonal losses if provided by the loss function.
    """

    with torch.no_grad():
        target = operations_non_transversal_fn.get(operation_id)[0].run(parameters_encoding=training_properties.parameters_encoding)
        prediction = operations_non_transversal_fn.get(operation_id)[1].run(parameters_encoding=training_properties.parameters_encoding,
                                                                            parameters_operation=training_properties.parameters_operation.
                                                                            parameters_operation_non_transversal.get(operation_id))
    loss_avg, loss_max, loss_log = operation_loss(prediction=prediction, target=target,
                                                  order_operation=code_properties.operation_non_transversal.get(operation_id).order,
                                                  method=training_properties.operation_loss,
                                                  display=f'TEST Non-Transversal Operation `{operation_id}`')
    return loss_avg, loss_max, loss_log.get('avg_diag'), loss_log.get('max_diag')


def run_tests(logger: Logger, training_properties: TrainingProperties, code_properties: CodeProperties,
              encoding_fn: EncodingModule, noisefree_fn: PhysicalModule, baseline_fn: PhysicalModule,
              operations_static_fn: dict[str, Tuple[OperationTargetModule, OperationPredictionModule]],
              operations_strictly_transversal_fn: dict[str, Tuple[OperationTargetModule, OperationPredictionModule]],
              operations_transversal_fn: dict[str, Tuple[OperationTargetModule, OperationPredictionModule]],
              operations_weakly_transversal_fn: dict[str, Tuple[OperationTargetModule, OperationPredictionModule]],
              operations_non_transversal_fn: dict[str, Tuple[OperationTargetModule, OperationPredictionModule]],
              epoch: int = 1):
    """Run a full test suite for encoding and logical operations and log results.

    This function evaluates:

    * the encoding vs. noise-free reference on test states (if configured),
    * a noisy baseline for the encoding, and
    * all configured logical operations (static, strictly transversal,
      transversal, weakly transversal, non-transversal),

    then aggregates operation losses and logs combined encoding+operation
    losses if the encoding is trainable.

    Args:
        logger (Logger): Logger instance used to record test and baseline
            results.
        training_properties (TrainingProperties): Training configuration and
            parameter containers.
        code_properties (CodeProperties): Code configuration including
            operations and trainable flags.
        encoding_fn (EncodingModule): Variational encoding module.
        noisefree_fn (PhysicalModule): Noise-free physical model.
        baseline_fn (PhysicalModule): Noisy physical baseline model.
        operations_static_fn (dict[str, tuple[OperationTargetModule, OperationPredictionModule]]):
            Static operations: mapping from operation IDs to (target,
            prediction) modules.
        operations_strictly_transversal_fn (dict[str, tuple[OperationTargetModule, OperationPredictionModule]]):
            Strictly transversal operations mapping.
        operations_transversal_fn (dict[str, tuple[OperationTargetModule, OperationPredictionModule]]):
            Transversal operations mapping.
        operations_weakly_transversal_fn (dict[str, tuple[OperationTargetModule, OperationPredictionModule]]):
            Weakly transversal operations mapping.
        operations_non_transversal_fn (dict[str, tuple[OperationTargetModule, OperationPredictionModule]]):
            Non-transversal operations mapping.
        epoch (int): Epoch index associated with this test run (used for
            logging).

    Returns:
        None
    """

    # encoding (Haar-random states)
    if training_properties.num_test_states is not None:
        loss_avg_, loss_max_ = test_encoding(training_properties, encoding_fn, noisefree_fn,
                                             number_states=training_properties.num_test_states,
                                             instance_states=training_properties.instance_test_states,
                                             display='TEST')
        logger.logger_encoding.log_test(loss_avg_, loss_max_,
                                        info={'epoch': epoch, 'num_states': training_properties.num_test_states})
        loss_avg_, loss_max_ = baseline_encoding(baseline_fn, noisefree_fn,
                                                 number_states=training_properties.num_test_states,
                                                 instance_states=training_properties.instance_test_states)
        logger.logger_encoding.log_baseline(loss_avg_, loss_max_,
                                            info={'epoch': epoch, 'num_states': training_properties.num_test_states})

    # operations, log to training (either first or last entry)
    losses_avg_operation, losses_max_operation = [], []
    # static operations
    for operation_id_ in operations_static_fn:
        loss_avg_, loss_max_, loss_avg_diag_, loss_max_diag_ = test_operation_static(training_properties, code_properties,
                                                                                     operations_static_fn,
                                                                                     operation_id=operation_id_)
        logger.logger_operation_static.get(operation_id_).log_train(loss_avg_, loss_max_,
                                                                  info={'epoch': epoch, 'loop': 1,
                                                                        'avg_diag': loss_avg_diag_,
                                                                        'max_diag': loss_max_diag_})
        logger.logger_operation_static.get(operation_id_).finish_epoch()
        losses_avg_operation.append(loss_avg_)
        losses_max_operation.append(loss_max_)
    # strictly-transversal operations
    for operation_id_ in operations_strictly_transversal_fn:
        loss_avg_, loss_max_, loss_avg_diag_, loss_max_diag_ = test_operation_strictly_transversal(training_properties, code_properties,
                                                                                                   operations_strictly_transversal_fn,
                                                                                                   operation_id=operation_id_)
        logger.logger_operation_strictly_transversal.get(operation_id_).log_train(loss_avg_, loss_max_,
                                                                                  info={'epoch': epoch, 'loop': 1,
                                                                                        'avg_diag': loss_avg_diag_,
                                                                                        'max_diag': loss_max_diag_})
        logger.logger_operation_strictly_transversal.get(operation_id_).finish_epoch()
        losses_avg_operation.append(loss_avg_)
        losses_max_operation.append(loss_max_)
    # transversal operations
    for operation_id_ in operations_transversal_fn:
        loss_avg_, loss_max_, loss_avg_diag_, loss_max_diag_ = test_operation_transversal(training_properties, code_properties,
                                                                                          operations_transversal_fn,
                                                                                          operation_id=operation_id_)
        logger.logger_operation_transversal.get(operation_id_).log_train(loss_avg_, loss_max_,
                                                                         info={'epoch': epoch, 'loop': 1,
                                                                               'avg_diag': loss_avg_diag_,
                                                                               'max_diag': loss_max_diag_})
        logger.logger_operation_transversal.get(operation_id_).finish_epoch()
        losses_avg_operation.append(loss_avg_)
        losses_max_operation.append(loss_max_)
    # weakly-transversal operations
    for operation_id_ in operations_weakly_transversal_fn:
        loss_avg_, loss_max_, loss_avg_diag_, loss_max_diag_ = test_operation_weakly_transversal(training_properties,
                                                                                                 code_properties,
                                                                                                 operations_weakly_transversal_fn,
                                                                                                 operation_id=operation_id_)
        logger.logger_operation_weakly_transversal.get(operation_id_).log_train(loss_avg_, loss_max_,
                                                                                info={'epoch': epoch, 'loop': 1,
                                                                                      'avg_diag': loss_avg_diag_,
                                                                                      'max_diag': loss_max_diag_})
        logger.logger_operation_weakly_transversal.get(operation_id_).finish_epoch()
        losses_avg_operation.append(loss_avg_)
        losses_max_operation.append(loss_max_)
    # non-transversal operations
    for operation_id_ in operations_non_transversal_fn:
        loss_avg_, loss_max_, loss_avg_diag_, loss_max_diag_ = test_operation_non_transversal(training_properties, code_properties,
                                                                                              operations_non_transversal_fn,
                                                                                              operation_id=operation_id_)
        logger.logger_operation_non_transversal.get(operation_id_).log_train(loss_avg_, loss_max_,
                                                                             info={'epoch': epoch, 'loop': 1,
                                                                                   'avg_diag': loss_avg_diag_,
                                                                                   'max_diag': loss_max_diag_})
        logger.logger_operation_non_transversal.get(operation_id_).finish_epoch()
        losses_avg_operation.append(loss_avg_)
        losses_max_operation.append(loss_max_)

    # combine encoding and operation losses
    if len(losses_avg_operation) > 0:
        loss_avg_operation = torch.sum(torch.stack(losses_avg_operation, dim=0), dim=0)
        loss_max_operation = torch.sum(torch.stack(losses_max_operation, dim=0), dim=0)
        logger.logger_operations_sum.log_train(loss_avg_operation, loss_max_operation, info={'epoch': epoch, 'loop': 1})
        logger.logger_operations_sum.finish_epoch()
        if code_properties.encoding_properties.trainable:
            # encoding (two-design states)
            loss_avg_encoding, loss_max_encoding = test_encoding(training_properties, encoding_fn, noisefree_fn,
                                                                 number_states=0, instance_states=None)
            logger.logger_encoding_operations_sum.log_train(loss_avg_encoding + training_properties.operation_loss_regularize * loss_avg_operation,
                                                            loss_max_encoding + training_properties.operation_loss_regularize * loss_max_operation,
                                                            info={'epoch': epoch, 'loop': 1, 'regularize': training_properties.operation_loss_regularize})
            logger.logger_encoding_operations_sum.finish_epoch()
