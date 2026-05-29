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

"""Argument parsing and experiment configuration for VarEFTQC.

This module reads experiment settings from a configuration file and from
command-line arguments, constructs the corresponding
:class:`CodeProperties`, :class:`NoiseProperties`, and
:class:`TrainingProperties` objects, and prepares a results folder for the
current experiment.

The default configuration is taken from ``config/config.txt`` (relative to
the project root) and can be overridden by command-line flags.
"""

import os
from pathlib import Path
import pickle
import configargparse
import numpy as np

from .utils import load_qasm_ansatz
from .data_structures import CodeProperties, NoiseProperties, TrainingProperties, OPERATIONS_1Q, OPERATIONS_2Q, \
    EncodingProperties, RecoveryProperties, OperationProperties


def parse():
    """Parse configuration and command-line arguments for a VarEFTQC experiment.

       This function sets up a :class:`configargparse.ArgParser` with all
       available options, reads defaults from ``config/config.txt``, and applies
       any command-line overrides.

       It also creates (or reuses) a results folder under ``results/`` and
       returns all objects required to run a VarEFTQC experiment.

       Returns:
           tuple:
               code_properties (CodeProperties): Description of the code, including
                   encoding, optional recovery, and (optional) logical operations.
               noise_properties (NoiseProperties): Noise model and parameters used
                   during training and evaluation.
               training_properties (TrainingProperties): Training configuration,
                   including optimizer settings, losses, and any loaded
                   pre-trained parameters.
               evaluate_code_distance (bool): Flag indicating whether only the code
                   distance should be evaluated (other functionality is disabled
                   in that case).
               folder_path (str): Path to the directory where results will be
                   written.
               use_tensorboard (bool): Flag indicating whether TensorBoard logging
                   is enabled.

       Raises:
           ValueError: If mutually exclusive options are used together (e.g.
               ``encoding`` with ``load_encoding``), or if lists of operations
               contain duplicates.
           RuntimeError: If required property/parameter files cannot be found, or
               if the experiment folder already exists and the ``--overwrite``
               flag is not set.
       """

    config_file = os.path.join(Path(__file__).resolve().parents[3], "config", "config.txt")
    parser = configargparse.ArgParser('Variational Early Fault-Tolerant Quantum Computing (VarEFTQC).',
                                      default_config_files=[config_file])

    # General arguments
    parser.add_argument('--seed', type=int, default=None,
                        help='Set seed for reproducibility of ansatz structure and initial parameters '
                             '(overwritten by `instance_encoding`, `instance_operation_non_transversal`,'
                             ' and `seed_parameters`).')
    parser.add_argument('--seed_parameters', type=int, default=None,
                        help='Set seed for reproducibility of initial parameters (overwrites general `seed`).')
    parser.add_argument('--path', type=str, default=None,
                        help='Create custom folder for storing experiment results (inside `results` folder).')
    parser.add_argument('--overwrite', action='store_true',
                        help='Skip the check if a experiment with this name already exists.')

    # Set system size
    parser.add_argument('--num_wires_data', type=int, default=1)
    parser.add_argument('--num_wires_ancilla', type=int, default=4)
    parser.add_argument('--num_wires_recovery', type=int, default=0)

    # Properties of variational ansätze
    parser.add_argument('--blocks_encoding', type=int, default=50,
                        help='Blocks with two-qubit interaction in encoding ansatz.')
    parser.add_argument('--instance_encoding', type=int, default=None,
                        help='Instance of encoding ansatz to use.')

    parser.add_argument('--blocks_recovery', type=int, default=0,
                        help='Blocks with two-qubit interaction in recovery ansatz (for `0` no recovery is trained).')
    parser.add_argument('--instance_recovery', type=int, default=None,
                        help='Instance of recovery ansatz to use.')

    parser.add_argument('--repeat_operation_transversal', type=int, default=[3], nargs='+',
                        help='Overparameterize transversal operation ansatz by repeating it.')
    parser.add_argument('--flip_operation_transversal', action='store_true',
                        help='Flip target and control wire for transversal tep-qubit ansatz.')
    parser.add_argument('--blocks_operation_non_transversal', type=int, default=[10], nargs='+',
                        help='Blocks with two-qubit interaction in non-transversal operation ansatz.')
    parser.add_argument('--instance_operation_non_transversal', type=int, default=None, nargs='+',
                        help='Instance of non-transversal operation ansatz to use.')

    parser.add_argument('--gates_1q_encoding', type=str, nargs='+', required=True,
                        help='One-qubit gate set for encoding ansatz.')
    parser.add_argument('--gate_2q_encoding', type=str, required=True,
                        help='Two-qubit gate for encoding ansatz.')
    parser.add_argument('--gates_1q_recovery', type=str, nargs='+', required=True,
                        help='One-qubit gate set for recovery ansatz.')
    parser.add_argument('--gate_2q_recovery', type=str, required=True,
                        help='Two-qubit gate for recovery ansatz.')
    parser.add_argument('--gate_1q_operation_transversal', type=str, required=True,
                        help='One-qubit gate for transversal operation ansatz.')
    parser.add_argument('--gate_2q_operation_transversal', type=str, required=True,
                        help='Two-qubit gate for transversal operation ansatz.')
    parser.add_argument('--gate_1q_operation_non_transversal', type=str, required=True,
                        help='One-qubit gate for non-transversal operation ansatz.')
    parser.add_argument('--gate_2q_operation_non_transversal', type=str, required=True,
                        help='Two-qubit gate for non-transversal operation ansatz.')

    # Training settings
    parser.add_argument('--epochs_encoding', type=int, default=50,
                        help='Number of epochs for training the encoding ansatz '
                             '(overwritten by `epochs_operation` if operations set).')
    parser.add_argument('--epochs_operation', type=int, default=100,
                        help='Number of epochs for training the encoding ansatz and operations '
                             '(overwrites `epochs_encoding` if operations set).')
    parser.add_argument('--epochs_recovery', type=int, default=0,
                        help='Number of epochs for post-training recovery operation.')

    parser.add_argument('--num_validation_states', type=int, default=None,
                        help='Number of Haar-random states to perform validation on (for None validation is skipped).')
    parser.add_argument('--num_test_states', type=int, default=None,
                        help='Number of Haar-random states to perform test on (for None testing is skipped).')
    parser.add_argument('--encoding_loss', type=str, default='avg',
                        choices=['avg', 'max'], help='Whether to use average or maximum formulation of '
                                                     'distinguishability loss for training encoding.')
    parser.add_argument('--operation_loss', type=str, default='block',
                        choices=['diag', 'block', 'block_ext', 'full'],
                        help='Type of loss function to use for training of logical operation '
                             '(`block_ext` only available for two-qubit operation).')
    parser.add_argument('--operation_loss_regularize', type=float, default=1.0,
                        help='Weighting factor for operation loss (w.r.t. encoding loss).')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate for L-BFGS optimizer.')
    parser.add_argument('--max_iter', type=int, default=20,
                        help='Number of internal iterations for L-BFGS optimizer.')
    parser.add_argument('--history_size', type=int, default=100,
                        help='Length of history for L-BFGS optimizer.')

    # Noise structure to train for, will be read from config.txt file
    parser.add_argument('--noise', type=str, default='depolarizing',
                        choices=['bitflip', 'phaseflip', 'amplitude_damping', 'phase_damping',
                                 'amplitude_phase_damping', 'depolarizing', 'pauli', 'thermal_relaxation'],
                        help='Type of noise, provided as string. Can also be set in the config.txt file.')
    parser.add_argument('--noise_strength', type=float, default=0.1,
                        help='Strength of noise (for `thermal_relaxation` duration in ms).')
    parser.add_argument('--noise_asymmetry', type=float, default=1.0,
                        help='Asymmetry parameter for asymmetric depolarizing noise (with 1.0 being symmetric).')
    parser.add_argument('--noise_pauli_x', type=float, default=None,
                        help='Noise weight parameter for pauli X-noise '
                             '(set Z-noise noise via `noise_pauli_z`, Y-noise will be determined automatically`).')
    parser.add_argument('--noise_pauli_z', type=float, default=None,
                        help='Noise weight parameter for pauli Z-noise '
                             '(set X-noise noise via `noise_pauli_x`, Y-noise will be determined automatically`).')
    parser.add_argument('--noise_t1', type=float, default=None,
                        help='T1 relaxation times for `thermal relaxation` noise, should be provided in ms.')
    parser.add_argument('--noise_t2', type=float, default=None,
                        help='T2 relaxation times for `thermal relaxation` noise, should be provided in ms.')

    # training logical operations
    parser.add_argument('--encoding', type=str, default=None,
                        help='Use static encoding (no trainable parameters, '
                             'provided as `ENCODING.qasm` file relative to `configuration/ansatz/encoding`).')
    parser.add_argument('--load_encoding', type=str, default=None,
                        help='Load trainable encoding (with trained parameters, '
                             'provided as `LOAD_ENCODING/encoding_properties.pkl` and '
                             '`LOAD_ENCODING/encoding_parameters.pkl` files relative to `results`).')
    parser.add_argument('--load_encoding_ansatz', type=str, default=None,
                        help='Load trainable encoding (without trained parameters, '
                             'provided as `LOAD_ENCODING_ANSATZ/encoding_properties.pkl` file relative to `results`).')

    parser.add_argument('--recovery', type=str, default=None,
                        help='Use static recovery (no trainable parameters, '
                             'provided as `RECOVERY.qasm` file relative to `configuration/ansatz/recovery`).')
    parser.add_argument('--load_recovery', type=str, default=None,
                        help='Load trainable recovery (with trained parameters, '
                             'provided as `LOAD_RECOVERY/recovery_properties.pkl` and '
                             'LOAD_RECOVERY/recovery_parameters.pkl` files relative to `results`).')
    parser.add_argument('--load_recovery_ansatz', type=str, default=None,
                        help='Load trainable recovery (without trained parameters, '
                             'provided as `LOAD_RECOVERY_ANSATZ/recovery_properties.pkl` file relative to `results`).')

    # static
    parser.add_argument('--operation', type=str, nargs='+', default=None,
                        help='Use static operation (no trainable parameters, '
                             'provided as `OPERATION.qasm` file relative to `configuration/ansatz/recovery`).')
    # strictly-transversal
    parser.add_argument('--operation_strictly_transversal', type=str, nargs='+', default=None,
                        choices=list(OPERATIONS_1Q.keys()) + list(OPERATIONS_2Q.keys()),
                        help='Operation, optionally multiple ones, that should be realized in a strictly transversal '
                             'fashion (no trainable parameters).')
    # transversal
    parser.add_argument('--operation_transversal', type=str, nargs='+', default=None,
                        choices=list(OPERATIONS_1Q.keys()) + list(OPERATIONS_2Q.keys()),
                        help='Operation, optionally multiple ones, that should be realized in a transversal fashion '
                             '(trainable ansatz with `repeat_operation_transversal` repetitions).')
    parser.add_argument('--load_operation_transversal', type=str, nargs='+', default=None,
                        help='Load trainable transversal operations (with trained parameters, '
                             'provided as `LOAD_TRANSVERSAL/OPERATION_transversal_properties.pkl` and '
                             'LOAD_TRANSVERSAL/OPERATION_transversal_parameters.pkl` files relative to `results`).')
    # weakly-transversal
    parser.add_argument('--operation_weakly_transversal', type=str, nargs='+', default=None,
                        choices=list(OPERATIONS_1Q.keys()) + list(OPERATIONS_2Q.keys()),
                        help='Operation, optionally multiple ones, that should be realized in a weakly-transversal fashion '
                             '(trainable ansatz with fixed layout that is universal on 2 qubits).')
    parser.add_argument('--load_operation_weakly_transversal', type=str, nargs='+', default=None,
                        help='Load trainable weakly-transversal operations (with trained parameters, '
                             'provided as `LOAD_WEAKLY_TRANSVERSAL/OPERATION_weakly_transversal_properties.pkl` and '
                             'LOAD_WEAKLY_TRANSVERSAL/OPERATION_weakly_transversal_parameters.pkl` files relative to `results`).')
    # non-transversal
    parser.add_argument('--operation_non_transversal', type=str, nargs='+', default=None,
                        choices=list(OPERATIONS_1Q.keys()) + list(OPERATIONS_2Q.keys()),
                        help='Operation, optionally multiple ones, that should be realized in a non-transversal '
                             'fashion (trainable ansatz with `blocks_operation_non_transversal` two-qubit blocks).')
    parser.add_argument('--load_operation_non_transversal', type=str, nargs='+', default=None,
                        help='Load trainable non-transversal operations (with trained parameters, '
                             'provided as `LOAD_NON_TRANSVERSAL/OPERATION_non_transversal_properties.pkl` and '
                             'LOAD_NON_TRANSVERSAL/OPERATION_non_transversal_parameters.pkl` files relative to `results`).')

    parser.add_argument('--evaluate_code_distance', action='store_true',
                        help='Evaluates potential code distance of provided code '
                             '(other functions are de-activated for this call).')

    parser.add_argument('--tensorboard', action='store_true',
                        help='Log via TensorBoard.')

    args = parser.parse_args()

    # determine a fixed encoding / recovery circuit configuration if not explicitly set; this is important to use the
    # same en-/decoding ansatz, as well as the same ansätze during the successive training steps;
    rng = np.random.default_rng(args.seed)
    instance_encoding_fallback = int(rng.integers(low=0, high=1000, size=1)[0])
    instance_operation_fallback = [int(iof) for iof in rng.integers(low=0, high=1000,
                                                                    size=0 if args.operation_non_transversal is None
                                                                    else len(args.operation_non_transversal))]
    instance_recovery_fallback = int(rng.integers(low=0, high=1000, size=1)[0])
    seed_parameters_fallback = int(rng.integers(low=0, high=1000, size=1)[0])
    instance_validation_states = int(rng.integers(low=0, high=1000, size=1)[0])
    instance_test_states = int(rng.integers(low=0, high=1000, size=1)[0])

    ##################
    # set up encoding
    ##################

    if ((args.encoding is not None and args.load_encoding is not None)
            or (args.encoding is not None and args.load_encoding_ansatz is not None)
            or (args.load_encoding is not None and args.load_encoding_ansatz is not None)):
        raise ValueError('Arguments `encoding`, `load_encoding`, and `load_encoding_ansatz` are mutually exclusive.`')

    if args.encoding is not None:  # load static encoding
        qasm_encoding, qasm_data_wires, qasm_ancilla_wires, _, _, _, _ = load_qasm_ansatz(encoding=args.encoding)
        encoding_properties = EncodingProperties(
            num_wires_data=len(qasm_data_wires),
            num_wires_ancilla=len(qasm_ancilla_wires),
            qasm=(qasm_encoding, qasm_data_wires, qasm_ancilla_wires)
        )
    elif args.load_encoding is not None or args.load_encoding_ansatz is not None:  # load trainable encoding
        path = os.path.join(Path(__file__).resolve().parents[3], 'results',
                            args.load_encoding if args.load_encoding is not None else args.load_encoding_ansatz,
                            'encoding_properties.pkl')
        if not os.path.isfile(path):
            raise RuntimeError(f'Encoding properties could not be found at {path}.')
        with open(path, 'rb') as ff:
            encoding_properties = pickle.load(ff)
        encoding_properties.print(prefix='Loaded')
    else:  # set up new trainable encoding
        encoding_properties = EncodingProperties(
            num_wires_data=args.num_wires_data,
            num_wires_ancilla=args.num_wires_ancilla,
            blocks=args.blocks_encoding,
            gates_1q=args.gates_1q_encoding, gate_2q=args.gate_2q_encoding,
            instance=instance_encoding_fallback if args.instance_encoding is None else args.instance_encoding
        )

    #############################
    # set up recovery (optional)
    #############################

    if ((args.recovery is not None and args.load_recovery is not None)
            or (args.recovery is not None and args.load_recovery_ansatz is not None)
            or (args.load_recovery is not None and args.load_recovery_ansatz is not None)):
        raise ValueError(
            'Arguments `recovery`, `load_recovery`, and `load_recovery_ansatz` are mutually exclusive.`')

    if ((args.epochs_recovery > 0) or (args.blocks_recovery > 0) or (args.recovery is not None) or
            (args.load_recovery is not None) or (args.load_recovery_ansatz is not None)):
        if args.recovery is not None:  # load static encoding
            qasm_recovery, qasm_data_wires, qasm_ancilla_wires, qasm_recovery_wires, _, _, _ \
                = load_qasm_ansatz(recovery=args.recovery)
            recovery_properties = RecoveryProperties(
                num_wires_data=len(qasm_data_wires),
                num_wires_ancilla=len(qasm_ancilla_wires),
                num_wires_recovery=len(qasm_recovery_wires),
                qasm=(qasm_recovery, qasm_data_wires, qasm_ancilla_wires, qasm_recovery_wires)
            )
        elif args.load_recovery is not None or args.load_recovery_ansatz is not None:  # load trainable recovery
            path = os.path.join(Path(__file__).resolve().parents[3], 'results',
                                args.load_recovery if args.load_recovery is not None else args.load_recovery_ansatz,
                                'recovery_properties.pkl')
            if not os.path.isfile(path):
                raise RuntimeError(f'Recovery properties could not be found at {path}.')
            with open(path, 'rb') as ff:
                recovery_properties = pickle.load(ff)
            recovery_properties.print(prefix='Loaded')
        else:  # set up new trainable recovery
            recovery_properties = RecoveryProperties(
                num_wires_data=encoding_properties.num_wires_data,  # ensures consistency if loaded encoding
                num_wires_ancilla=encoding_properties.num_wires_ancilla,   # ensures consistency if loaded encoding
                num_wires_recovery=args.num_wires_recovery,
                blocks=args.blocks_recovery,
                gates_1q=args.gates_1q_recovery, gate_2q=args.gate_2q_recovery,
                instance=instance_recovery_fallback if args.instance_recovery is None else args.instance_recovery
            )
    else:
        recovery_properties = None

    ###############################
    # set up operations (optional)
    ###############################

    # static operations provided as qasm file
    if args.operation is not None:
        if len(args.operation) > len(set(args.operation)):
            raise ValueError('List of static operations is not unique.')
        operation_static = {}
        for o in args.operation:
            qasm_operation, qasm_data_wires, qasm_ancilla_wires, _, qasm_data_target_wires, qasm_ancilla_target_wires, qasm_gate \
                = load_qasm_ansatz(operation=o)
            operation_static[qasm_gate] = OperationProperties(name=qasm_gate,
                                                              num_wires=len(qasm_data_wires) + len(qasm_ancilla_wires),
                                                              qasm=(qasm_operation, qasm_data_wires, qasm_ancilla_wires,
                                                                    qasm_data_target_wires, qasm_ancilla_target_wires))
    else:
        operation_static = None

    # strictly transversal operations
    if args.operation_strictly_transversal is not None:
        if len(args.operation_strictly_transversal) > len(set(args.operation_strictly_transversal)):
            raise ValueError('List of strictly-transversal operations is not unique.')
        operation_strictly_transversal = {o: OperationProperties(name=o, num_wires=encoding_properties.num_wires,
                                                                 strictly_transversal=True)
                                          for o in args.operation_strictly_transversal}
    else:
        operation_strictly_transversal = None

    # transversal operations
    if args.operation_transversal is not None:
        if len(args.operation_transversal) > len(set(args.operation_transversal)):
            raise ValueError('List of transversal operations is not unique.')
        repeat_operation = len(args.operation_transversal) * args.repeat_operation_transversal \
            if 1 == len(args.repeat_operation_transversal) \
            else args.repeat_operation_transversal   # same number of repetitions for all operation
        operation_transversal = {o: OperationProperties(name=o, num_wires=encoding_properties.num_wires,
                                                        transversal=True, repeat=r,
                                                        flip=args.flip_operation_transversal,
                                                        gate_1q=args.gate_1q_operation_transversal,
                                                        gate_2q=args.gate_2q_operation_transversal)
                                 for o, r in zip(args.operation_transversal, repeat_operation)}
    else:
        operation_transversal = None
    # load pre-trained transversal operation
    if args.load_operation_transversal is not None:
        if len(args.load_operation_transversal) > len(set(args.load_operation_transversal)):
            raise ValueError('List of transversal operations to load is not unique.')
        if operation_transversal is None:
            operation_transversal = {}
        for lot in args.load_operation_transversal:
            operation_id = lot.split('/')[-1]
            if operation_id in operation_transversal.keys():
                raise RuntimeError(f'Loading transversal {operation_id} operation aborted, already selected via '
                                   f'`--operation_transversal`.')
            path = os.path.join(Path(__file__).resolve().parents[3], 'results', f'{lot}_transversal_properties.pkl')
            if not os.path.isfile(path):
                raise RuntimeError(f'Transversal {operation_id} operation properties could not be found at {path}.')
            with open(path, 'rb') as ff:
                operation_transversal_properties = pickle.load(ff)
            operation_transversal_properties.print(prefix='Loaded')
            operation_transversal[operation_id] = operation_transversal_properties

    # weakly-transversal operations
    if args.operation_weakly_transversal is not None:
        if len(args.operation_weakly_transversal) > len(set(args.operation_weakly_transversal)):
            raise ValueError('List of weakly-transversal operations is not unique.')
        operation_weakly_transversal = {o: OperationProperties(name=o, num_wires=encoding_properties.num_wires,
                                                               weakly_transversal=True)
                                        for o in args.operation_weakly_transversal}
    else:
        operation_weakly_transversal = None
    # load pre-trained weakly-transversal operation
    if args.load_operation_weakly_transversal is not None:
        if len(args.load_operation_weakly_transversal) > len(set(args.load_operation_weakly_transversal)):
            raise ValueError('List of weakly-transversal operations to load is not unique.')
        if operation_weakly_transversal is None:
            operation_weakly_transversal = {}
        for lowt in args.load_operation_weakly_transversal:
            operation_id = lowt.split('/')[-1]
            if operation_id in operation_weakly_transversal.keys():
                raise RuntimeError(f'Loading weakly-transversal {operation_id} operation aborted, already selected via '
                                   f'`--operation_weakly_transversal`.')
            path = os.path.join(Path(__file__).resolve().parents[3], 'results', f'{lowt}_weakly_transversal_properties.pkl')
            if not os.path.isfile(path):
                raise RuntimeError(f'Weakly-transversal {operation_id} operation properties could not be found at {path}.')
            with open(path, 'rb') as ff:
                operation_weakly_transversal_properties = pickle.load(ff)
            operation_weakly_transversal_properties.print(prefix='Loaded')
            operation_weakly_transversal[operation_id] = operation_weakly_transversal_properties

    # non-transversal operations
    if args.operation_non_transversal is not None:
        if len(args.operation_non_transversal) > len(set(args.operation_non_transversal)):
            raise ValueError('List of non-transversal operations is not unique.')
        blocks_operation = len(args.operation_non_transversal) * args.blocks_operation_non_transversal \
            if 1 == len(args.blocks_operation_non_transversal) \
            else args.blocks_operation_non_transversal   # same number of blocks for all operations
        instance_operation = instance_operation_fallback if args.instance_operation_non_transversal is None \
            else args.instance_operation_non_transversal
        instance_operation = len(args.operation_non_transversal) * instance_operation \
            if 1 == len(instance_operation) \
            else instance_operation   # same instance for all operations
        operation_non_transversal = {o: OperationProperties(name=o, num_wires=encoding_properties.num_wires,
                                                            blocks=bo, instance=ia,
                                                            gate_1q=args.gate_1q_operation_non_transversal,
                                                            gate_2q=args.gate_2q_operation_non_transversal)
                                     for o, bo, ia in zip(args.operation_non_transversal, blocks_operation,
                                                          instance_operation)}
    else:
        operation_non_transversal = None
    # load pre-trained non-transversal operation
    if args.load_operation_non_transversal is not None:
        if len(args.load_operation_non_transversal) > len(set(args.load_operation_non_transversal)):
            raise ValueError('List of non-transversal operations to load is not unique.')
        if operation_non_transversal is None:
            operation_non_transversal = {}
        for lont in args.load_operation_non_transversal:
            operation_id = lont.split('/')[-1]
            if operation_id in operation_non_transversal.keys():
                raise RuntimeError(f'Loading non-transversal {operation_id} operation aborted, already selected via '
                                   f'`--operation_non_transversal`.')
            path = os.path.join(Path(__file__).resolve().parents[3], 'results', f'{lont}_non_transversal_properties.pkl')
            if not os.path.isfile(path):
                raise RuntimeError(f'Non-transversal {operation_id} operation properties could not be found at {path}.')
            with open(path, 'rb') as ff:
                operation_non_transversal_properties = pickle.load(ff)
            operation_non_transversal_properties.print(prefix='Loaded')
            operation_non_transversal[operation_id] = operation_non_transversal_properties

    ####################################
    # compose code and noise components
    ####################################

    code_properties = CodeProperties(
        encoding_properties=encoding_properties,
        recovery_properties=recovery_properties,
        operation_static=operation_static,
        operation_strictly_transversal=operation_strictly_transversal,
        operation_transversal=operation_transversal,
        operation_weakly_transversal=operation_weakly_transversal,
        operation_non_transversal=operation_non_transversal
    )

    print('\n==========\n')

    noise_properties = NoiseProperties(
        noise=args.noise,
        noise_strength=args.noise_strength,
        noise_asymmetry=args.noise_asymmetry,
        noise_pauli_x=args.noise_pauli_x,
        noise_pauli_z=args.noise_pauli_z,
        noise_t1=args.noise_t1,
        noise_t2=args.noise_t2,
        train_encoding=code_properties.train_encoding
    )

    print('\n==========\n')

    ##############################################
    # load trained parameters and set up training
    ##############################################

    # load trained encoding parameters (optional)
    if args.load_encoding is not None:
        path = os.path.join(Path(__file__).resolve().parents[3], 'results', args.load_encoding, 'encoding_parameters.pkl')
        if not os.path.isfile(path):
            raise RuntimeError(f'Encoding parameters could not be found at {path}.')
        with open(path, 'rb') as ff:
            parameters_encoding_trained = pickle.load(ff)
    else:
        parameters_encoding_trained = None

    # load trained recovery parameters (optional)
    if args.load_recovery is not None:
        path = os.path.join(Path(__file__).resolve().parents[3], 'results', args.load_recovery, 'recovery_parameters.pkl')
        if not os.path.isfile(path):
            raise RuntimeError(f'Recovery parameters could not be found at {path}.')
        with open(path, 'rb') as ff:
            parameters_recovery_trained = pickle.load(ff)
    else:
        parameters_recovery_trained = None

    # load trained transversal operation parameters (optional)
    if args.load_operation_transversal is not None:
        parameters_operation_transversal_trained = {}
        for lot in args.load_operation_transversal:
            operation_id = lot.split('/')[-1]
            path = os.path.join(Path(__file__).resolve().parents[3], 'results', f'{lot}_transversal_parameters.pkl')
            if not os.path.isfile(path):
                raise RuntimeError(f'Transversal {operation_id} operation parameters could not be found at {path}.')
            with open(path, 'rb') as ff:
                parameters_operation_transversal_trained[operation_id] = pickle.load(ff)
    else:
        parameters_operation_transversal_trained = None

    # load trained weakly-transversal operation parameters (optional)
    if args.load_operation_weakly_transversal is not None:
        parameters_operation_weakly_transversal_trained = {}
        for lowt in args.load_operation_weakly_transversal:
            operation_id = lowt.split('/')[-1]
            path = os.path.join(Path(__file__).resolve().parents[3], 'results', f'{lowt}_weakly_transversal_parameters.pkl')
            if not os.path.isfile(path):
                raise RuntimeError(f'Weakly-transversal {operation_id} operation parameters could not be found at {path}.')
            with open(path, 'rb') as ff:
                parameters_operation_weakly_transversal_trained[operation_id] = pickle.load(ff)
    else:
        parameters_operation_weakly_transversal_trained = None

    # load trained non-transversal operation parameters (optional)
    if args.load_operation_non_transversal is not None:
        parameters_operation_non_transversal_trained = {}
        for lont in args.load_operation_non_transversal:
            operation_id = lont.split('/')[-1]
            path = os.path.join(Path(__file__).resolve().parents[3], 'results', f'{lont}_non_transversal_parameters.pkl')
            if not os.path.isfile(path):
                raise RuntimeError(f'Non-transversal {operation_id} operation parameters could not be found at {path}.')
            with open(path, 'rb') as ff:
                parameters_operation_non_transversal_trained[operation_id] = pickle.load(ff)
    else:
        parameters_operation_non_transversal_trained = None

    training_properties = TrainingProperties(
        code_properties=code_properties,
        seed_parameters=seed_parameters_fallback if args.seed_parameters is None else args.seed_parameters,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        history_size=args.history_size,
        epochs_encoding=args.epochs_encoding,
        epochs_encoding_operation=args.epochs_operation,
        epochs_recovery=args.epochs_recovery,
        encoding_loss=args.encoding_loss,
        operation_loss=args.operation_loss,
        operation_loss_regularize=args.operation_loss_regularize,
        parameters_encoding_trained=parameters_encoding_trained,
        parameters_recovery_trained=parameters_recovery_trained,
        parameters_operation_transversal_trained=parameters_operation_transversal_trained,
        parameters_operation_weakly_transversal_trained=parameters_operation_weakly_transversal_trained,
        parameters_operation_non_transversal_trained=parameters_operation_non_transversal_trained,
        num_validation_states=args.num_validation_states,
        instance_validation_states=instance_validation_states,
        num_test_states=args.num_test_states,
        instance_test_states=instance_test_states
    )

    print('\n==========\n')

    # set up folder structure for storing results
    folder_path = os.path.join(Path(__file__).resolve().parents[3], 'results') if args.path is None \
        else os.path.join(Path(__file__).resolve().parents[3], 'results', args.path)
    if os.path.exists(folder_path):
        if not args.overwrite:
            raise RuntimeError(f'The experiment {folder_path} already exists. Use `--overwrite` to repeat it.')
        print(f'Writing results to {folder_path} (potentially overwriting).')
    else:
        os.makedirs(folder_path)
        print(f'Writing results to {folder_path}.')

    return code_properties, noise_properties, training_properties, args.evaluate_code_distance, folder_path, args.tensorboard
