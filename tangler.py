#!/usr/bin/env python3

import glob, os
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
    train_parser.add_argument('--optimizer', type=str, default='adam_amsgrad')
    train_parser.add_argument('--learning-rate', '-lr', type=float, default=1e-4)
    train_parser.add_argument('--loss', type=str, default='mse')
    train_parser.add_argument('--weighted-loss', action='store_true')
    train_parser.add_argument('--mixed-precision', action='store_true')
    train_parser.add_argument('--batch', '-b', type=int, default=100)
    train_parser.add_argument('--epochs', '-e', type=int, default=100)
    train_parser.add_argument('--train-steps-per-epoch', '-ts', type=int)
    train_parser.add_argument('--val-steps', '-vs', type=int)
    train_parser.add_argument('--patience', type=int, default=30)
    train_parser.add_argument('--checkpoint-period', type=int, default=1)
    train_parser.add_argument('--name', type=str, default=None)
    train_parser.add_argument('--checkpoint-path', type=str, default='/tmp/latest.tf')
    train_parser.add_argument('--output', '-o', default='results')
    train_parser.add_argument('--train-data', type=str)
    train_parser.add_argument('--val-data', type=str)

    predict_parser = subparsers.add_parser("predict", help='Run inference on arbitrary image(s)')
    predict_parser.add_argument('--res', '-r', type=int, default=600)
    predict_parser.add_argument('--num-pins', '-k', type=int, default=300)
    predict_parser.add_argument('--model', '-m', help='Path to saved model', required=True)
    predict_parser.add_argument('fname', nargs='+', help='Image(s) to convert into thread pattern')

    demo_parser = subparsers.add_parser("demo", help='Run inferences in demonstration mode')
    demo_parser.add_argument('--model', '-m', help='Saved model path', required=True)
    demo_parser.add_argument('--backend', default='opengl', help='Rendering backend (opengl|matplotlib)')
    demo_parser.add_argument('--cycle', action='store_true', help='Repeat input images indefinitely')
    demo_parser.add_argument('input', nargs='*', help='File names for inference. Defaults to webcam input')

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

        train_records = glob.glob(os.path.join(args.train_data, '*.tfrecord'))
        val_records = glob.glob(os.path.join(args.val_data, '*.tfrecord'))

        if args.debug:
            train_records = train_records[:1]
            val_records = val_records[-1:]

        #train_records = [f's3://storage-9iudgkuqwurq6/tangler/tfrecords/train/tangle_{i:05d}-of-00016.tfrecord' for i in range(16)]
        #val_records = [f's3://storage-9iudgkuqwurq6/tangler/tfrecords/val/tangle_{i:05d}-of-00016.tfrecord' for i in range(16)]

        do_train(train_records, val_records, args.output, model_name=args.name,
            checkpoint_path=args.checkpoint_path, checkpoint_period=args.checkpoint_period,
            loss_function=args.loss, optimizer=args.optimizer, learning_rate=args.learning_rate,
            weighted_loss=args.weighted_loss, batch_size=args.batch,
            epochs=args.epochs, patience=args.patience, use_mixed_precision=args.mixed_precision,
            train_steps_per_epoch=args.train_steps_per_epoch, val_steps=args.val_steps)

    elif args.mode == "predict":
        from predict import do_predict
        do_predict(paths=args.fname, model_path=args.model, res=args.res, n_pins=args.num_pins)

    elif args.mode == "demo":
        from opengl_demo import do_demo
        source = args.input if len(args.input) > 0 else 'webcam'
        do_demo(args.model, source)
        #from demo import do_demo
        #do_demo(model_path=args.model, source=source, backend=args.backend, cycle=args.cycle)
