#!/usr/bin/env python3

import glob, os, sys
import pathlib
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser('Tangler', usage='python3 tangler.py [RUNTIME_FLAGS]')

    parser.add_argument('--debug', action='store_true', help='Enable TensorFlow debug mode')
    parser.add_argument('--cpu', action='store_true', help='Disable GPU compute')

    subparsers = parser.add_subparsers(dest='mode', title='Mode', description='Which action to perform.')

    prep_parser = subparsers.add_parser("prep", help='Convert the dataset into .tfrecord format for training')
    prep_parser.add_argument('--res', '-r', type=int, default=600,
        help="Crop/scale images to squares with RES pixels per side (default: 600)")
    prep_parser.add_argument('--num-shards', '-n', type=int, default=10,
        help="Split dataset into multiple files (default: 10)")
    prep_parser.add_argument('--num-procs', '-J', type=int, default=10)
    prep_parser.add_argument('--path-length', '-N', type=int, default=6000)
    prep_parser.add_argument('--num-pins', '-k', type=int, default=300)
    prep_parser.add_argument('--num-cons', '-c', type=int, default=-1, help='Defaults to 2*path_len//n_pins')
    prep_parser.add_argument('input', help='Root directory for dataset (e.g. ./train)')
    prep_parser.add_argument('output', help='Directory in which to save tfrecord files')

    train_parser = subparsers.add_parser("train", help='Train the model')
    train_parser.add_argument('--optimizer', type=str, default='adam_amsgrad', help='Optimizer to use in model.fit. Default: adam_amsgrad')
    train_parser.add_argument('--learning-rate', '-lr', type=float, default=1e-4, help='Learning rate for optimizer. Default: 1e-4')
    train_parser.add_argument('--loss', type=str, default='mse', help='Loss function for optimizer. Default: mse')
    train_parser.add_argument('--cache', action='store_true', help='Cache examples in RAM. Default: false')
    train_parser.add_argument('--vis', action='store_true', help='Generate a graphical representation of the model architecture. Saves to `output_dir/models/{...}.png`')
    train_parser.add_argument('--dry-run', action='store_true', help='Compile and summarize model, then exit')
    train_parser.add_argument('--format', type=str, default='h5', choices=['h5', 'tf'], help='Format to save model. Default: h5')
    train_parser.add_argument('--fp16', action='store_true', help='Use mixed precision fp16/fp32 training mode. Default: false')
    train_parser.add_argument('--batch', '-b', type=int, default=100, help='Number of examples per batch. Default: 100')
    train_parser.add_argument('--epochs', '-e', type=int, default=100, help='How many epochs to run before terminating. Default: 100')
    train_parser.add_argument('--patience', type=int, default=1, help='Terminate early if validation loss does not improve in this many epochs. Default: 1')
    train_parser.add_argument('--train-steps', '-ts', type=int, help='How many batches of training data to process per epoch. Default: all of them')
    train_parser.add_argument('--val-steps', '-vs', type=int, help='How many batches to run on validation data. Default: all of them')
    train_parser.add_argument('--checkpoint-period', type=int, default=1, help='Epochs between checkpoint outpubs. Ignored if --train-steps-per-epoch is not specified. Default: 1')
    train_parser.add_argument('--name', type=str, default=None, help='Arbitrary model name. If omitted will default to a name descriptive of the model settings')
    train_parser.add_argument('--checkpoint', type=pathlib.Path, default='/tmp/latest', help='Uses the same output format as --save-format. Default: /tmp/latest')
    train_parser.add_argument('--output', '-o', type=pathlib.Path, required=True, help='Root directory for logs and model results.')
    train_parser.add_argument('--train-data', '-td', type=pathlib.Path, required=True)
    train_parser.add_argument('--val-data', '-vd', type=pathlib.Path, required=True)

    predict_parser = subparsers.add_parser("predict", help='Run inference on arbitrary image(s)')
    predict_parser.add_argument('--res', '-r', type=int, default=600)
    predict_parser.add_argument('--num-pins', '-k', type=int, default=300)
    predict_parser.add_argument('--model', '-m', help='Path to saved model', required=True)
    predict_parser.add_argument('fname', nargs='+', help='Image(s) to convert into thread pattern')

    demo_parser = subparsers.add_parser("demo", help='Run inferences in demonstration mode')
    demo_parser.add_argument('--source', '-s', default='webcam', choices=['webcam', 'files'], help='Inference source. If "files" source is selected, then the --input option must be specified. Default: webcam')
    demo_parser.add_argument('--input', '-i', nargs='+', help='Images or directories containing images for inferencing. Ignored unless `--source files` is specified')
    demo_parser.add_argument('--cycle', action='store_true', help='Repeat input images indefinitely. Ignored unless --source files is specified. Default: false')

    demo_parser.add_argument('--mirror', '-m', action='store_true', help='Flip visualization output Left/Right. Default: false')
    demo_parser.add_argument('--delay', '-d', default=0, type=int, help='Time delay in milliseconds between frames. Default: 0')

    demo_parser.add_argument('model', help='Saved model path')

    args = parser.parse_args()

    if (args.debug or args.cpu):
        import tensorflow as tf
        if (args.debug):
            tf.config.run_functions_eagerly(True)
            tf.data.experimental.enable_debug_mode()

        if (args.cpu):
            tf.config.set_visible_devices([], 'GPU')

    if args.mode == "prep":
        from prep import do_prep
        do_prep(input_dir=args.input, output_dir=args.output,
            n_cons=args.num_cons, num_shards=args.num_shards, res=args.res,
            path_len=args.path_length, n_pins=args.num_pins, n_procs=args.num_procs)

    elif args.mode == "train":
        from train import do_train

        assert os.path.isdir(args.train_data)
        assert os.path.isdir(args.val_data)
        for d in ('', 'logs', 'models'):
            D = os.path.join(args.output, d)
            if not os.path.exists(D):
                os.makedirs(D)
            assert os.path.isdir(D)

        train_records = glob.glob(os.path.join(args.train_data, '*.tfrecord'))
        val_records = glob.glob(os.path.join(args.val_data, '*.tfrecord'))

        if args.debug:
            train_records = train_records[:1]
            val_records = val_records[-1:]

        #train_records = [f's3://storage-9iudgkuqwurq6/tangler/tfrecords/train/tangle_{i:05d}-of-00016.tfrecord' for i in range(16)]
        #val_records = [f's3://storage-9iudgkuqwurq6/tangler/tfrecords/val/tangle_{i:05d}-of-00016.tfrecord' for i in range(16)]

        do_train(train_records, val_records, args.output, model_name=args.name,
            checkpoint_path=args.checkpoint, checkpoint_period=args.checkpoint_period,
            loss_function=args.loss, optimizer=args.optimizer, learning_rate=args.learning_rate,
            data_cache=args.cache, vis_model=args.vis, batch_size=args.batch, save_format=args.format,
            epochs=args.epochs, patience=args.patience, use_mixed_precision=args.fp16,
            train_steps=args.train_steps, val_steps=args.val_steps, dry_run=args.dry_run)

    elif args.mode == "predict":
        from predict import do_predict
        do_predict(paths=args.fname, model_path=args.model, res=args.res, n_pins=args.num_pins)

    elif args.mode == "demo":
        source = args.source
        if source == "files":
            source = []
            for src in args.input:
                if os.path.isdir(src):
                    source += glob.glob(os.path.join(src, '*'))
                else:
                    source.append(src)
            for src in source:
                assert os.path.isfile(src), \
                    f'Not a valid input source: {src}'

        from opengl_demo import do_demo
        do_demo(args.model, source, args.mirror, args.cycle, args.delay)
