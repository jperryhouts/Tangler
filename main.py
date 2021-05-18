#!/usr/bin/env python3

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--debug', action='store_true', help='Enable TensorFlow debug mode')
    parser.add_argument('--cpu', action='store_true', help='Disable GPU compute')

    subparsers = parser.add_subparsers(dest='mode')

    prep_parser = subparsers.add_parser("prep")
    prep_parser.add_argument('--res', '-r', type=int, default=600,
        help="Crop/scale images to squares with this number of pixels per side (default: 600)")
    prep_parser.add_argument('--num-shards', '-n', type=int, default=10,
        help="Split dataset into this many tfrecord files (default: 10)")
    prep_parser.add_argument('input', help='Root directory for dataset (e.g. ./train)')
    prep_parser.add_argument('output', help='Directory in which to save tfrecord files')

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument('--res', '-r', type=int, default=600)
    train_parser.add_argument('--path-length', '-N', type=int, default=6000)
    train_parser.add_argument('--num-pins', '-k', type=int, default=300)
    train_parser.add_argument('--batch', '-b', type=int, default=10)
    train_parser.add_argument('--epochs', '-e', type=int, default=1)
    train_parser.add_argument('--overshoot-epochs', type=int, default=30)
    train_parser.add_argument('--random-seed', type=int, default=42)
    train_parser.add_argument('--output', '-o', default='results', help='Path in which to save models and logs')
    train_parser.add_argument('tfrecord', nargs='+', help='Path(s) to tfrecord data')

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument('--res', '-r', type=int, default=600)
    predict_parser.add_argument('--num-pins', '-k', type=int, default=300)
    predict_parser.add_argument('--model', '-m', help='Path to saved model', required=True)
    predict_parser.add_argument('--output', '-o', help='Folder in which to save predictions')
    predict_parser.add_argument('fname', nargs='+', help='Image(s) to convert into thread pattern')

    args = parser.parse_args()

    if (args.debug or args.cpu):
        import tensorflow as tf

        if (args.debug):
            tf.config.run_functions_eagerly(True)
            tf.data.experimental.enable_debug_mode()

        if (args.cpu):
            tf.config.set_visible_devices([], 'GPU')

    if args.mode == "train":
        from train import do_train
        do_train(args.tfrecord, args.res, args.path_length, args.output,
            args.num_pins, args.batch, args.epochs, overshoot_epochs=args.overshoot_epochs,
            random_seed=args.random_seed)

    elif args.mode == "predict":
        from predict import do_predict
        do_predict(args.fname, args.model, args.output, args.res, args.num_pins)

    elif args.mode == "prep":
        from prep_dataset import create_tf_records
        create_tf_records(args.input, args.output, num_shards=args.num_shards, res=args.res)
