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

"""Logging utilities for VarEFTQC experiments.

This module provides:

* a small wrapper class (:class:`Loss`) that normalizes loss values and
  metadata, and
* logger classes (:class:`Logger`, :class:`LoggerBase`) that collect losses
  during training/validation/testing/baseline evaluation and optionally
  forward them to TensorBoard.
"""

import os
from dataclasses import dataclass
import torch
from tensorboardX import SummaryWriter
from typing import Optional
from vareftqc.helpers.data_structures import (EncodingProperties, OperationProperties, RecoveryProperties,
                                              TrainingProperties)


@dataclass
class Loss:
    """Container for a single loss value and its metadata.

    This class ensures that loss values and selected metadata fields are
    stored as Python floats instead of tensors, which simplifies further
    processing and serialization.

    Attributes:
        average (float | torch.Tensor): Average loss value. Tensor inputs are
            converted to floats.
        maximum (float | torch.Tensor): Maximum loss value. Tensor inputs are
            converted to floats.
        info (dict | None): Optional dictionary with additional information,
            e.g. ``{"epoch": int, "loop": int, "num_states": int, ...}``.
            Any tensor values are converted to floats.
    """

    average: float | torch.Tensor
    maximum: float | torch.Tensor
    info: Optional[dict] = None

    def __post_init__(self):
        if isinstance(self.average, torch.Tensor):
            self.average = self.average.item()
        if isinstance(self.maximum, torch.Tensor):
            self.maximum = self.maximum.item()
        if self.info is not None:
            for i in self.info:
                if isinstance(self.info[i], torch.Tensor):
                    self.info[i] = self.info[i].item()  # noqa

    def epoch(self):
        """Return the epoch index stored in :attr:`info`.

                Returns:
                    int: Epoch index.

                Raises:
                    RuntimeError: If ``"epoch"`` is not present in :attr:`info`.
                """

        if self.info.get('epoch', None) is None:
            raise RuntimeError('No `epoch` specified.')
        return self.info['epoch']

    def loop(self):
        """Return the inner-loop index stored in :attr:`info`.

                Returns:
                    int: Inner-loop index.

                Raises:
                    RuntimeError: If ``"loop"`` is not present in :attr:`info`.
                """

        if self.info.get('loop', None) is None:
            raise RuntimeError('No `loop` specified.')
        return self.info['loop']

    def num_states(self):
        """Return the number of states stored in :attr:`info`.

                Returns:
                    int: Number of states used for this loss.

                Raises:
                    RuntimeError: If ``"num_states"`` is not present in :attr:`info`.
                """

        if self.info.get('num_states', None) is None:
            raise RuntimeError('No `num_states` specified.')
        return self.info['num_states']

    def regularize(self):
        """Return the regularization factor stored in :attr:`info`.

                Returns:
                    float: Regularization factor.

                Raises:
                    RuntimeError: If ``"regularize"`` is not present in :attr:`info`.
                """

        if self.info.get('regularize', None) is None:
            raise RuntimeError('No `regularize` specified.')
        return self.info['regularize']

    def avg_diag(self):
        """Return the optional 'avg_diag' value from :attr:`info`.

                Returns:
                    float | None: Average diagonal loss or ``None`` if not present.
                """

        return self.info.get('avg_diag', None)

    def max_diag(self):
        """Return the optional 'max_diag' value from :attr:`info`.

               Returns:
                   float | None: Maximum diagonal loss or ``None`` if not present.
               """

        return self.info.get('max_diag', None)


class Logger:
    """High-level logger aggregating all component loggers.

    This class creates and holds logger instances for:

    * encoding,
    * recovery (if present),
    * static operations,
    * strictly transversal operations,
    * transversal operations,
    * weakly transversal operations,
    * non-transversal operations,
    * summed operation loss, and
    * combined encoding+operation loss.

    Optionally, it also manages a TensorBoard writer used by all sub-loggers.

    Args:
        training_properties (TrainingProperties): Global training properties,
            used to configure iteration counts and operation-loss weighting.
        encoding_properties (EncodingProperties | None): Encoding properties,
            used to decide whether encoding is trainable.
        recovery_properties (RecoveryProperties | None): Recovery properties,
            or ``None`` if no recovery module is used.
        operation_static (dict[str, OperationProperties] | None): Static
            logical operations.
        operation_strictly_transversal (dict[str, OperationProperties] | None):
            Strictly transversal logical operations.
        operation_transversal (dict[str, OperationProperties] | None):
            Trainable transversal logical operations.
        operation_weakly_transversal (dict[str, OperationProperties] | None):
            Trainable weakly transversal logical operations.
        operation_non_transversal (dict[str, OperationProperties] | None):
            Trainable non-transversal logical operations.
        tensorboard_path (str | None): If not ``None``, path to the
            experiment directory used to create a TensorBoard log subfolder.
    """

    def __init__(self, training_properties: TrainingProperties,
                 encoding_properties: EncodingProperties = None,
                 recovery_properties: RecoveryProperties = None,
                 operation_static: dict[str, OperationProperties] = None,
                 operation_strictly_transversal: dict[str, OperationProperties] = None,
                 operation_transversal: dict[str, OperationProperties] = None,
                 operation_weakly_transversal: dict[str, OperationProperties] = None,
                 operation_non_transversal: dict[str, OperationProperties] = None,
                 tensorboard_path: str = None):

        # set up tensorboard logger (optional, only if path is set)
        self.tensorboard_writer = SummaryWriter(log_dir=os.path.join(tensorboard_path, 'tensorboard')) \
            if tensorboard_path is not None else None

        # encoding logger
        self.logger_encoding = LoggerBase(logger_prefix='Encoding',
                                          max_iter=training_properties.max_iter,
                                          log_training=encoding_properties.trainable or (operation_static is None
                                                                                         and operation_strictly_transversal is None
                                                                                         and operation_transversal is None),
                                          log_validation=training_properties.num_validation_states is not None,
                                          log_test=training_properties.num_test_states is not None,
                                          log_baseline=training_properties.num_test_states is not None,
                                          tensorboard_writer=self.tensorboard_writer)

        # recovery logger
        self.logger_recovery = None if recovery_properties is None else (
            LoggerBase(logger_prefix='Recovery',
                       max_iter=training_properties.max_iter,
                       log_training=True, log_validation=False, log_test=True,
                       log_baseline=True, tensorboard_writer=self.tensorboard_writer))

        # operation loggers
        self.logger_operation_static = None if operation_static is None else \
            {os_: LoggerBase(logger_prefix='Operation', logger_suffix=f'{os}_static',
                             max_iter=training_properties.max_iter,
                             log_training=True,
                             log_validation=False, log_test=False, log_baseline=False,
                             tensorboard_writer=self.tensorboard_writer)
             for os_ in operation_static.keys()}

        self.logger_operation_strictly_transversal = None if operation_strictly_transversal is None else \
            {ost: LoggerBase(logger_prefix='Operation', logger_suffix=f'{ost}_ST',
                             max_iter=training_properties.max_iter,
                             log_training=True,
                             log_validation=False, log_test=False, log_baseline=False,
                             tensorboard_writer=self.tensorboard_writer)
             for ost in operation_strictly_transversal.keys()}

        self.logger_operation_transversal = None if operation_transversal is None else \
            {ot: LoggerBase(logger_prefix='Operation', logger_suffix=f'{ot}_T',
                            max_iter=training_properties.max_iter,
                            log_training=True, log_validation=False, log_test=False, log_baseline=False,
                            tensorboard_writer=self.tensorboard_writer)
             for ot in operation_transversal.keys()}

        self.logger_operation_weakly_transversal = None if operation_weakly_transversal is None else \
            {owt: LoggerBase(logger_prefix='Operation', logger_suffix=f'{owt}_WT',
                             max_iter=training_properties.max_iter,
                             log_training=True, log_validation=False, log_test=False, log_baseline=False,
                             tensorboard_writer=self.tensorboard_writer)
             for owt in operation_weakly_transversal.keys()}

        self.logger_operation_non_transversal = None if operation_non_transversal is None else \
            {ont: LoggerBase(logger_prefix='Operation', logger_suffix=f'{ont}_NT',
                             max_iter=training_properties.max_iter,
                             log_training=True, log_validation=False, log_test=False, log_baseline=False,
                             tensorboard_writer=self.tensorboard_writer)
             for ont in operation_non_transversal.keys()}

        # logging combined (weighted) encoding and operation losses
        self.logger_encoding_operations_sum = None if not encoding_properties.trainable or (operation_static is None
                                                                                            and operation_strictly_transversal is None
                                                                                            and operation_transversal is None
                                                                                            and operation_weakly_transversal is None
                                                                                            and operation_non_transversal is None) \
            else LoggerBase(logger_prefix=f'Encoding',
                            logger_suffix=f'{'' if 1.0 == training_properties.operation_loss_regularize else f'{training_properties.operation_loss_regularize}'}ΣOperations',
                            max_iter=training_properties.max_iter,
                            log_training=True,
                            log_validation=False, log_test=False, log_baseline=False,
                            tensorboard_writer=self.tensorboard_writer)

        # logging combined operation losses
        self.logger_operations_sum = None if (operation_static is None and operation_strictly_transversal is None
                                              and operation_transversal is None and operation_weakly_transversal is None
                                              and operation_non_transversal is None) \
            else LoggerBase(logger_prefix=f'Operation', logger_suffix='ΣOperations',
                            max_iter=training_properties.max_iter,
                            log_training=True, log_validation=False, log_test=False, log_baseline=False,
                            tensorboard_writer=self.tensorboard_writer)

    def disconnect_tensorboard(self):
        """Detach all loggers from the TensorBoard writer.

                This is useful before serializing the :class:`Logger` instance, as
                the underlying :class:`SummaryWriter` object is not picklable.
                After calling this method, no further TensorBoard logging will occur.
                """

        self.tensorboard_writer = None
        if self.logger_encoding is not None:
            self.logger_encoding.tensorboard_writer = None
        if self.logger_recovery is not None:
            self.logger_recovery.tensorboard_writer = None
        if self.logger_operation_static is not None:
            for logger_id in self.logger_operation_static:
                self.logger_operation_static[logger_id].tensorboard_writer = None
        if self.logger_operation_strictly_transversal is not None:
            for logger_id in self.logger_operation_strictly_transversal:
                self.logger_operation_strictly_transversal[logger_id].tensorboard_writer = None
        if self.logger_operation_transversal is not None:
            for logger_id in self.logger_operation_transversal:
                self.logger_operation_transversal[logger_id].tensorboard_writer = None
        if self.logger_operation_weakly_transversal is not None:
            for logger_id in self.logger_operation_weakly_transversal:
                self.logger_operation_weakly_transversal[logger_id].tensorboard_writer = None
        if self.logger_operation_non_transversal is not None:
            for logger_id in self.logger_operation_non_transversal:
                self.logger_operation_non_transversal[logger_id].tensorboard_writer = None
        if self.logger_encoding_operations_sum is not None:
            self.logger_encoding_operations_sum.tensorboard_writer = None
        if self.logger_operations_sum is not None:
            self.logger_operations_sum.tensorboard_writer = None


class LoggerBase:
    """Base logger for a single component (e.g. encoding or one operation).

    A :class:`LoggerBase` instance collects:

    * training losses (grouped by epoch),
    * validation losses (per epoch),
    * test losses (per epoch), and
    * baseline losses (per epoch),

    and optionally forwards them to a TensorBoard writer.

    Args:
        max_iter (int): Maximum number of training iterations (inner loops)
            per epoch, used to index TensorBoard steps and to enforce
            consistency.
        logger_prefix (str): Prefix used for TensorBoard tag names
            (e.g. ``"Encoding"`` or ``"Operation"``).
        logger_suffix (str | None): Optional suffix appended to the prefix to
            distinguish different operations (e.g. ``"X_T"``).
        log_training (bool): If ``True``, record training losses.
        log_validation (bool): If ``True``, record validation losses.
        log_test (bool): If ``True``, record test losses.
        log_baseline (bool): If ``True``, record baseline losses.
        tensorboard_writer (SummaryWriter | None): Optional TensorBoard writer
            used to log scalars; if ``None``, no TensorBoard logging occurs.
    """

    def __init__(self, max_iter: int, logger_prefix: str, logger_suffix: str = None,
                 log_training: bool = True, log_validation: bool = False, log_test: bool = False, log_baseline: bool = False,
                 tensorboard_writer: SummaryWriter = None):
        self.logger_prefix, self.logger_suffix = logger_prefix, logger_suffix
        self.max_iter = max_iter
        self.train = [] if log_training else None
        self._train = [] if log_training else None
        self.validation = [] if log_validation else None
        self.test = [] if log_test else None
        self.baseline = [] if log_baseline else None
        self.tensorboard_writer = tensorboard_writer

    def log_train(self, average: float | torch.Tensor, maximum: float | torch.Tensor, info: dict = None):
        """Log a training loss for the current epoch.

        The loss is stored internally (grouped by epoch) and, if a
        TensorBoard writer is available, written as scalars under a tag
        derived from ``logger_prefix`` and ``logger_suffix``.

        Args:
            average (float | torch.Tensor): Average loss value for this
                training step.
            maximum (float | torch.Tensor): Maximum loss value for this
                training step.
            info (dict | None): Additional metadata, typically including
                ``"epoch"`` and ``"loop"`` indices and, optionally,
                diagonal loss values.
        """

        if self.train is None:
            raise RuntimeError('Logging training deactivated.')
        if len(self._train) >= self.max_iter:
            raise RuntimeError(f'Reached {len(self._train)+1} iterations, but `max_iter` is set to {self.max_iter} '
                               f'\(probably missing call to `finish_epoch`).')
        loss = Loss(average=average, maximum=maximum, info=info)
        if self.tensorboard_writer is not None:
            # self.tensorboard_writer.add_scalar(f'{self.logger_prefix}/{'Train' if self.logger_suffix is None
            # else f'{self.logger_suffix}'}', loss.average, (loss.epoch() - 1) * self.max_iter + (loss.loop() - 1))
            if loss.avg_diag() is not None and loss.max_diag() is not None:
                self.tensorboard_writer.add_scalars(f'{self.logger_prefix}/{'Train' if self.logger_suffix is None
                else f'{self.logger_suffix}'}', {'Avg': loss.average, 'Max': loss.maximum,
                                                 'DiagAvg': loss.avg_diag(), 'MaxDiag': loss.max_diag()},
                                                    (loss.epoch() - 1) * self.max_iter + (loss.loop() - 1))
            else:
                self.tensorboard_writer.add_scalars(f'{self.logger_prefix}/{'Train' if self.logger_suffix is None
                else f'{self.logger_suffix}'}', {'Avg': loss.average, 'Max': loss.maximum},
                                                    (loss.epoch() - 1) * self.max_iter + (loss.loop() - 1))
        self._train.append(loss)

    def log_validation(self, average: float | torch.Tensor, maximum: float | torch.Tensor, info: dict = None):
        """Log a validation loss.

                Args:
                    average (float | torch.Tensor): Average validation loss.
                    maximum (float | torch.Tensor): Maximum validation loss.
                    info (dict | None): Additional metadata, usually including the
                        epoch index and optionally the number of validation states.
                """

        if self.validation is None:
            raise RuntimeError('Logging validation deactivated.')
        loss = Loss(average=average, maximum=maximum, info=info)
        if self.tensorboard_writer is not None:
            # self.tensorboard_writer.add_scalar(f'{self.logger_prefix}/Validation', loss.average, loss.epoch() * self.max_iter)
            self.tensorboard_writer.add_scalars(f'{self.logger_prefix}/Validation{'' if 0 == loss.num_states() 
            else f'{loss.num_states()}'}', {'Avg': loss.average, 'Max': loss.maximum}, loss.epoch() * self.max_iter)
        self.validation.append(loss)

    def log_test(self, average: float | torch.Tensor, maximum: float | torch.Tensor, info: dict = None):
        """Log a test loss.

                Args:
                    average (float | torch.Tensor): Average test loss.
                    maximum (float | torch.Tensor): Maximum test loss.
                    info (dict | None): Additional metadata, usually including the
                        epoch index and optionally the number of test states.
                """

        if self.test is None:
            raise RuntimeError('Logging test deactivated.')
        loss = Loss(average=average, maximum=maximum, info=info)
        if self.tensorboard_writer is not None:
            # self.tensorboard_writer.add_scalar(f'{self.logger_prefix}/Test', loss.average, (loss.epoch() - 1) * self.max_iter)
            self.tensorboard_writer.add_scalars(f'{self.logger_prefix}/Test{'' if 'Encoding' != self.logger_prefix or 0 == loss.num_states() 
            else f'{loss.num_states()}'}', {'Avg': loss.average, 'Max': loss.maximum}, (loss.epoch() - 1) * self.max_iter)
        self.test.append(loss)

    def log_baseline(self, average: float | torch.Tensor, maximum: float | torch.Tensor, info: dict = None):
        """Log a baseline loss.

                Baseline losses typically correspond to noisy physical models
                without variational encoding or recovery.

                Args:
                    average (float | torch.Tensor): Average baseline loss.
                    maximum (float | torch.Tensor): Maximum baseline loss.
                    info (dict | None): Additional metadata, usually including the
                        epoch index and optionally the number of states.
                """

        if self.baseline is None:
            raise RuntimeError('Logging baseline deactivated.')
        loss = Loss(average=average, maximum=maximum, info=info)
        if self.tensorboard_writer is not None:
            # self.tensorboard_writer.add_scalar(f'{self.logger_prefix}/Baseline', loss.average, (loss.epoch() - 1) * self.max_iter)
            self.tensorboard_writer.add_scalars(f'{self.logger_prefix}/Test{'' if 'Encoding' != self.logger_prefix or 0 == loss.num_states() 
            else f'{loss.num_states()}'}', {'BaselineAvg': loss.average, 'BaselineMax': loss.maximum},
                                                (loss.epoch() - 1) * self.max_iter)
        self.baseline.append(loss)

    def finish_epoch(self):
        """Mark the end of the current training epoch.

                All training losses accumulated in the internal buffer for this
                epoch are moved to the main ``train`` list, and the buffer is reset.

                Raises:
                    RuntimeError: If training logging is deactivated or if no
                        training entries have been logged for the current epoch.
                """

        if self.train is None:
            raise RuntimeError('Logging training deactivated.')
        if 0 == len(self._train):
            raise RuntimeError('Cannot finish empty epoch.')
        self.train.append(self._train)
        self._train = []

    def get_train(self):
        """Return all recorded training losses.

                Returns:
                    tuple[list[list[float]] | None, list[list[float]] | None, list[list[dict]] | None]:
                        Nested lists of averages, maxima, and info dictionaries per
                        epoch, or ``(None, None, None)`` if training logging is
                        disabled.
                """

        if self.train is None:
            return None, None, None
        else:
            return ([[t.average for t in tr] for tr in self.train], [[t.maximum for t in tr] for tr in self.train],
                    [[t.info for t in tr] for tr in self.train])

    def get_validation(self):
        """Return all recorded validation losses.

                Returns:
                    tuple[list[float] | None, list[float] | None, list[dict] | None]:
                        Lists of averages, maxima, and info dictionaries per epoch,
                        or ``(None, None, None)`` if validation logging is disabled.
                """

        if self.validation is None:
            return None, None, None
        else:
            return ([v.average for v in self.validation], [v.maximum for v in self.validation],
                    [v.info for v in self.validation])

    def get_test(self):
        """Return all recorded test losses.

                Returns:
                    tuple[list[float] | None, list[float] | None, list[dict] | None]:
                        Lists of averages, maxima, and info dictionaries, or
                        ``(None, None, None)`` if test logging is disabled.
                """

        if self.test is None:
            return None, None, None
        else:
            return [t.average for t in self.test], [t.maximum for t in self.test], [t.info for t in self.test]

    def get_baseline(self):
        """Return all recorded baseline losses.

                Returns:
                    tuple[list[float] | None, list[float] | None, list[dict] | None]:
                        Lists of averages, maxima, and info dictionaries, or
                        ``(None, None, None)`` if baseline logging is disabled.
                """

        if self.baseline is None:
            return None, None, None
        else:
            return ([b.average for b in self.baseline], [b.maximum for b in self.baseline],
                    [b.info for b in self.baseline])
