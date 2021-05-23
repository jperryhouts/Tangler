#!/usr/bin/env python3

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--debug', action='store_true', help='Enable TensorFlow debug mode')
    parser.add_argument('--cpu', action='store_true', help='Disable GPU compute')

    subparsers = parser.add_subparsers(dest='mode')

    prep_parser = subparsers.add_parser("prep")
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

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3)
    train_parser.add_argument('--optimizer', type=str, default='adam')
    train_parser.add_argument('--loss', type=str, default='mse')
    train_parser.add_argument('--batch', '-b', type=int, default=10)
    train_parser.add_argument('--epochs', '-e', type=int, default=1)
    train_parser.add_argument('--steps-per-epoch', '-s', type=int, default=1000)
    train_parser.add_argument('--overshoot-epochs', type=int, default=30)
    train_parser.add_argument('--checkpoint-period', type=int, default=1)
    train_parser.add_argument('--random-seed', type=int, default=42)
    train_parser.add_argument('--name', type=str, default=None)
    train_parser.add_argument('--output', '-o', default='results', help='Path in which to save models and logs')
    train_parser.add_argument('tfrecord', nargs='+', help='Path(s) to tfrecord data')

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument('--res', '-r', type=int, default=600)
    predict_parser.add_argument('--num-pins', '-k', type=int, default=300)
    predict_parser.add_argument('--model', '-m', help='Path to saved model', required=True)
    predict_parser.add_argument('fname', nargs='+', help='Image(s) to convert into thread pattern')

    args = parser.parse_args()

    import tensorflow as tf
    if (args.debug or args.cpu):
        if (args.debug):
            tf.config.run_functions_eagerly(True)
            tf.data.experimental.enable_debug_mode()

        if (args.cpu):
            tf.config.set_visible_devices([], 'GPU')

    elif args.mode == "prep":
        from prep_dataset import create_tf_records
        create_tf_records(input_dir=args.input, output_dir=args.output,
            n_cons=args.num_cons, num_shards=args.num_shards, res=args.res,
            path_len=args.path_length, n_pins=args.num_pins, n_procs=args.num_procs)

    if args.mode == "train":
        from train import do_train
        do_train(train_records=args.tfrecord, output_dir=args.output, model_name=args.name,
            loss=args.loss, optimizer=args.optimizer, learning_rate=args.learning_rate,
            batch_size=args.batch, epochs=args.epochs, overshoot_epochs=args.overshoot_epochs,
            steps_per_epoch=args.steps_per_epoch, checkpoint_period=args.checkpoint_period,
            random_seed=args.random_seed)

    elif args.mode == "predict":
        from predict_compressed_mask import do_predict
        do_predict(paths=args.fname, model_path=args.model, res=args.res, n_pins=args.num_pins)
