#!/usr/bin/env python3

import os, pathlib
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser('Tangler', usage='python3 -m tangler [RUNTIME ARGUMENTS]')

    parser.add_argument('--debug', action='store_true', help='Enable TensorFlow debug mode')
    parser.add_argument('--cpu', action='store_true', help='Disable GPU compute')

    subparsers = parser.add_subparsers(dest='mode', title='Mode', description='Which action to perform.')

    train_parser = subparsers.add_parser("train", help='Train the model')
    train_parser.add_argument('--optimizer', type=str, default='adam_amsgrad', help='Optimizer to use in model.fit. Default: adam_amsgrad')
    train_parser.add_argument('--learning-rate', '-lr', type=float, default=1e-4, help='Learning rate for optimizer. Default: 1e-4')
    train_parser.add_argument('--loss', type=str, default='pooled_binary_crossentropy', help='Loss function for optimizer. Default: pooled_binary_crossentropy')
    train_parser.add_argument('--cache', action='store_true', help='Cache examples in RAM. Default: false')
    train_parser.add_argument('--vis', action='store_true', help='Generate a graphical representation of the model architecture. Saves to `output_dir/models/{...}.png`')
    train_parser.add_argument('--summarize', action='store_true', help='Compile and summarize model, then exit')
    train_parser.add_argument('--peek', action='store_true', help='Visualize each data example')
    train_parser.add_argument('--format', type=str, default='h5', choices=['h5', 'tf'], help='Format to save model. Default: h5')
    train_parser.add_argument('--fp16', action='store_true', help='Use mixed precision fp16/fp32 training mode. Default: false')
    train_parser.add_argument('--quantization-aware', '-qa', action='store_true', help='Prefer weights and biases that can be rounded to lower precision. Default: false')
    train_parser.add_argument('--batch', '-b', type=int, default=100, help='Number of examples per batch. Default: 100')
    train_parser.add_argument('--epochs', '-e', type=int, default=100, help='How many epochs to run before terminating. Default: 100')
    train_parser.add_argument('--patience', type=int, default=-1, help='Terminate early if validation loss does not improve in this many epochs. Default: none')
    train_parser.add_argument('--train-steps', '-ts', type=int, help='How many batches of training data to process per epoch. Default: all of them')
    train_parser.add_argument('--val-steps', '-vs', type=int, help='How many batches to run on validation data. Default: all of them')
    train_parser.add_argument('--name', type=str, default=None, help='Arbitrary model name. If omitted will default to a name descriptive of the model settings')
    train_parser.add_argument('--checkpoint-path', type=pathlib.Path, default='/tmp/latest', help='Uses the same output format as --save-format. Default: /tmp/latest')
    train_parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint. Default: false')
    train_parser.add_argument('--fine-tuning', action='store_true', help='Set MobileNet variables trainable. Default: false')
    train_parser.add_argument('--train-data', '-td', type=pathlib.Path, required=True)
    train_parser.add_argument('--val-data', '-vd', type=pathlib.Path, required=True)
    train_parser.add_argument('--output', '-o', type=pathlib.Path, required=True, help='Root directory for logs and model results.')

    eval_parser = subparsers.add_parser("evaluate", help='Evaluate metrics on test data')
    eval_parser.add_argument('--batch', type=int, default=25, help='Number of examples per batch. Default: 25')
    eval_parser.add_argument('--loss', type=str, default='pooled_binary_crossentropy', help='Loss function for optimizer. Default: pooled_binary_crossentropy')
    eval_parser.add_argument('model', help='Saved model path')
    eval_parser.add_argument('test_data', help='Path to test data')

    demo_parser = subparsers.add_parser("demo", help='Run inferences in demo mode')
    demo_parser.add_argument('--source', '-s', default='webcam', choices=['webcam', 'video', 'files'], help='Inference source. If "files" source is selected, then the --input option must be specified. Default: webcam')
    demo_parser.add_argument('--input', '-i', nargs='+', default=["0"], help='Images or directories containing images for inferencing. Ignored unless `--source files` is specified')
    demo_parser.add_argument('--cycle', action='store_true', help='Repeat input images indefinitely. Ignored unless --source files is specified. Default: false')
    demo_parser.add_argument('--mirror', '-m', action='store_true', help='Flip visualization output Left/Right. Default: false')
    demo_parser.add_argument('--stats', action='store_true', help='Periodically report performance stats including FPS and inference times. Default: false')
    demo_parser.add_argument('--delay', '-d', default=0, type=int, help='Time delay in milliseconds between frames. Default: 0')
    demo_parser.add_argument('--path-buffer', default=35000, type=int, help='Size of memory buffer to use for string path. Default: 35000')
    demo_parser.add_argument('--threshold', default=-2.5, type=float, help='Value to consider a positive prediction. Default: 0')
    demo_parser.add_argument('--resolution', '-r', default=512, type=int, help='Size of rendered output. Default: 512')
    demo_parser.add_argument('--aspect', '-a', default='1.0', help='Aspect ratio of output. Can be a number or aspect ratio in the format "w:h"')
    demo_parser.add_argument('--webcam', help='Output as fake webcam device using v4l2loopback. The path of a v4l2loopback device must be specified as an argument (e.g. /dev/video1)')
    demo_parser.add_argument('model', help='Saved model path')

    quant_parser = subparsers.add_parser("convert", help='Convert model for tflite or tfjs')
    quant_parser.add_argument('--format', '-f', default='tflite', choices=('tflite','tfjs'), help='Target format for model output')
    quant_parser.add_argument('input', type=pathlib.Path, help='Original model weights')
    quant_parser.add_argument('output', type=pathlib.Path, help='Resulting quantized model')

    args = parser.parse_args()

    if (args.debug or args.cpu):
        import tensorflow as tf
        if (args.debug):
            tf.config.run_functions_eagerly(True)
            tf.data.experimental.enable_debug_mode()

        if (args.cpu):
            tf.config.set_visible_devices([], 'GPU')

    if args.mode == "train":
        from .train import fit

        assert os.path.isdir(args.train_data)
        assert os.path.isdir(args.val_data)
        for d in ('', 'logs', 'models'):
            D = os.path.join(args.output, d)
            if not os.path.exists(D):
                os.makedirs(D)
            assert os.path.isdir(D)

        train_data = str(args.train_data.absolute())
        val_data = str(args.val_data.absolute())
        fit(train_data, val_data, args.output, model_name=args.name,
            resume=args.resume, checkpoint_path=args.checkpoint_path,
            loss_function=args.loss, optimizer=args.optimizer, learning_rate=args.learning_rate,
            data_cache=args.cache, vis_model=args.vis, batch_size=args.batch, save_format=args.format,
            train_steps=args.train_steps, val_steps=args.val_steps,
            epochs=args.epochs, patience=args.patience,
            do_fine_tuning=args.fine_tuning,
            use_mixed_precision=args.fp16, quantization_aware=args.quantization_aware,
            summarize=args.summarize, peek=args.peek)

    elif args.mode == "evaluate":
        from .train import evaluate

        assert os.path.isdir(args.test_data)
        evaluate(args.model, args.test_data, args.loss, args.batch)

    elif args.mode == "demo":
        from .inference import demo
        if ':' in args.aspect:
            w,h = args.aspect.strip().split(':')
            aspect = float(w)/float(h)
        else:
            aspect = float(args.aspect)

        demo(args.model, args.source, args.input, args.mirror, args.cycle, args.delay,
            args.path_buffer, args.threshold, args.resolution, args.webcam,
            aspect=aspect, show_stats=args.stats)

    elif args.mode == "convert":
        from . import convert
        if args.format == 'tflite':
            convert.model_to_tflite(args.input, args.output)
        elif args.format == 'tfjs':
            convert.model_to_tfjs(args.input, args.output)
        else:
            raise Exception("Input format not recognized:",args.format)
