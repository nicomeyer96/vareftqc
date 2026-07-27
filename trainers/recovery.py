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

"""Training routines for the recovery module in VarEFTQC.

This module implements:

* the high-level training loop for the recovery ansatz using an L-BFGS
  optimizer,
* a test function comparing recovery to a noise-free reference, and
* a baseline function comparing a noisy physical model to the noise-free
  reference.
"""

import torch

from vareftqc.helpers import CodeProperties, TrainingProperties
from vareftqc import PhysicalModule, RecoveryModule
from vareftqc.loss_functions import fidelity_loss
from .logger import Logger


def train_recovery(logger: Logger, code_properties: CodeProperties, training_properties: TrainingProperties,
                   recovery_fn: RecoveryModule, noisefree_fn: PhysicalModule, baseline_fn: PhysicalModule):
    """Train the recovery ansatz and evaluate its performance.

    This function:

    * computes a noise-free reference using a physical model,
    * performs an initial test and baseline evaluation,
    * logs an initial recovery evaluation,
    * optionally runs an L-BFGS optimizer over the recovery parameters, and
    * performs final tests and logs the final recovery performance.

    Args:
        logger (Logger): Logger used to record training, test, and baseline
            losses for recovery.
        code_properties (CodeProperties): Code configuration indicating
            whether recovery is trainable.
        training_properties (TrainingProperties): Training configuration and
            recovery parameter container.
        recovery_fn (RecoveryModule): Variational recovery module to train.
        noisefree_fn (PhysicalModule): Noise-free physical model used as the
            target in the fidelity loss.
        baseline_fn (PhysicalModule): Noisy physical model used as a baseline
            for comparison.

    Returns:
        None
    """

    # pre-compute recovery static states for later use (training)
    noisefree = noisefree_fn.run()

    ##################
    # initial testing
    ##################
    run_tests(logger=logger, training_properties=training_properties,
              recovery_fn=recovery_fn, noisefree_fn=noisefree_fn, baseline_fn=baseline_fn, epoch=1)

    # log initial recovery, might be overwritten by training (guarantees log if no training performed)
    recovery_initial = recovery_fn.run(parameters_encoding=training_properties.parameters_encoding,
                                       parameters_recovery=training_properties.parameters_recovery)
    loss_avg_recovery_initial, loss_max_recovery_initial = fidelity_loss(prediction=recovery_initial,
                                                                         groundtruth=noisefree)
    logger.logger_recovery.log_train(loss_avg_recovery_initial, loss_max_recovery_initial, info={'epoch': 1, 'loop': 1})
    logger.logger_recovery.finish_epoch()

    ##################
    # set up training
    ##################

    # perform training if recovery is trainable
    if training_properties.epochs_recovery > 0 and code_properties.train_recovery:

        ###################
        # set up optimizer
        ###################
        optimizer = torch.optim.LBFGS([*training_properties.parameters_recovery.parameters()],
                                      lr=training_properties.learning_rate, max_iter=training_properties.max_iter,
                                      history_size=training_properties.history_size)

        # mutable counter for inner loops of L-BFGS optimizer
        loop = [0]

        #######################
        # closure for training
        #######################
        def closure():
            optimizer.zero_grad()
            loop[0] += 1
            # forward pass
            recovery = recovery_fn.run(parameters_encoding=training_properties.parameters_encoding,
                                       parameters_recovery=training_properties.parameters_recovery)
            # determine fidelity loss
            loss_avg, loss_max = fidelity_loss(prediction=recovery, groundtruth=noisefree,
                                               display=f'TRAIN #{epoch}/{training_properties.epochs_recovery} '
                                                       f'(loop #{loop[0]}) Recovery')
            # logging
            logger.logger_recovery.log_train(loss_avg, loss_max, info={'epoch': epoch, 'loop': loop[0]})
            # backward pass
            loss_avg.backward()
            return loss_avg

        ##########################
        # perform actual training
        ##########################
        for epoch in range(1, training_properties.epochs_recovery + 1):
            # perform one optimization epoch (might perform several internal steps)
            loop[0] = 0
            optimizer.step(closure)
            # logging
            logger.logger_recovery.finish_epoch()

    ################
    # final testing
    ################
    run_tests(logger=logger, training_properties=training_properties,
              recovery_fn=recovery_fn, noisefree_fn=noisefree_fn, baseline_fn=baseline_fn,
              epoch=training_properties.epochs_recovery + 1)

    # log final recovery (guarantees log after last training step)
    recovery_final = recovery_fn.run(parameters_encoding=training_properties.parameters_encoding,
                                     parameters_recovery=training_properties.parameters_recovery)
    loss_avg_recovery_final, loss_max_recovery_final = fidelity_loss(prediction=recovery_final,
                                                                     groundtruth=noisefree)
    logger.logger_recovery.log_train(loss_avg_recovery_final, loss_max_recovery_final,
                                     info={'epoch': training_properties.epochs_recovery + 1, 'loop': 1})
    logger.logger_recovery.finish_epoch()


def test_recovery(training_properties: TrainingProperties, recovery_fn: RecoveryModule, noisefree_fn: PhysicalModule):
    """Evaluate the recovery ansatz against a noise-free reference.

    The recovery module is applied after the encoding, and its output is
    compared to the noise-free physical evolution using a fidelity-based
    loss.

    Args:
        training_properties (TrainingProperties): Training configuration,
            providing access to encoding and recovery parameters.
        recovery_fn (RecoveryModule): Variational recovery module to test.
        noisefree_fn (PhysicalModule): Noise-free physical model.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Average and maximum fidelity
        losses (scalars), as returned by :func:`fidelity_loss`.
    """

    with torch.no_grad():
        recovery_test = recovery_fn.run(parameters_encoding=training_properties.parameters_encoding,
                                        parameters_recovery=training_properties.parameters_recovery)
        noisefree_test = noisefree_fn.run()
    loss_avg, loss_max = fidelity_loss(prediction=recovery_test, groundtruth=noisefree_test, display='TEST Recovery')
    return loss_avg, loss_max


def baseline_recovery(baseline_fn: PhysicalModule, noisefree_fn: PhysicalModule):
    """Compute a baseline for recovery using a noisy physical model.

    This compares the output of a noisy physical model directly to the
    noise-free reference (without any learned recovery), and serves as a
    baseline for the variational recovery performance.

    Args:
        baseline_fn (PhysicalModule): Noisy physical model used as baseline.
        noisefree_fn (PhysicalModule): Noise-free physical reference model.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Average and maximum fidelity
        losses (scalars), as returned by :func:`fidelity_loss`.
    """

    with torch.no_grad():
        baseline_test = baseline_fn.run()
        noisefree_test = noisefree_fn.run()
    loss_avg, loss_max = fidelity_loss(prediction=baseline_test, groundtruth=noisefree_test, display='TEST Baseline')
    return loss_avg, loss_max


def run_tests(logger: Logger, training_properties: TrainingProperties,
              recovery_fn: RecoveryModule, noisefree_fn: PhysicalModule, baseline_fn: PhysicalModule,
              epoch: int = 1):
    """Run recovery and baseline tests and log the results.

    This function evaluates:

    * the current recovery module vs. the noise-free reference, and
    * the noisy physical baseline vs. the noise-free reference,

    and logs both test and baseline losses for the given epoch.

    Args:
        logger (Logger): Logger used to record test and baseline losses.
        training_properties (TrainingProperties): Training configuration and
            parameter containers.
        recovery_fn (RecoveryModule): Variational recovery module.
        noisefree_fn (PhysicalModule): Noise-free physical reference model.
        baseline_fn (PhysicalModule): Noisy physical baseline model.
        epoch (int): Epoch index associated with this test run (used for
            logging).

    Returns:
        None
    """

    loss_avg_, loss_max_ = test_recovery(training_properties, recovery_fn, noisefree_fn)
    logger.logger_recovery.log_test(loss_avg_, loss_max_, info={'epoch': epoch})
    loss_avg_, loss_max_ = baseline_recovery(baseline_fn, noisefree_fn)
    logger.logger_recovery.log_baseline(loss_avg_, loss_max_, info={'epoch': epoch})
