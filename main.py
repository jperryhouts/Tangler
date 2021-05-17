#!/usr/bin/env python3

import os
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
    train_parser.add_argument('--output', '-o', help='Save model to path')
    train_parser.add_argument('dataset', help='path to tfrecord dataset directory')

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument('--res', '-r', type=int, default=600)
    predict_parser.add_argument('--num-pins', '-k', type=int, default=300)
    predict_parser.add_argument('--model', '-m', help='path to saved model', required=True)
    predict_parser.add_argument('--output', '-o', help='Save prediction to file')
    predict_parser.add_argument('fname', help='Image to convert into thread pattern')

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
        model = do_train(args.dataset, args.res, args.path_length, args.num_pins, args.batch)
        if (args.output):
            model.save(args.output)

    elif args.mode == "predict":
        from predict import do_predict
        do_predict(args.fname, args.model, args.output, args.res, args.num_pins)

    elif args.mode == "prep":
        from prep_dataset import create_tf_records
        if not os.path.exists(args.output):
            os.makedirs(args.output)

        assert os.path.isdir(args.input)
        assert os.path.isdir(args.output)

        create_tf_records(args.input, args.output, num_shards=args.num_shards, res=args.res)
