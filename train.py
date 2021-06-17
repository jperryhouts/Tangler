from typing import Tuple, Callable, Optional
import datetime, os, random
import tensorflow as tf
import glob

from model import TangledModel
from simple_model import SimpleModel, get_down_stack
import utils

def get_compiler_args(loss_func_id:str, optimizer_id:Optional[str]=None,
        learning_rate:Optional[float]=None) -> Tuple[Callable, Optional[tf.keras.optimizers.Optimizer], list[str]]:

    if optimizer_id is None:
        optimizer = None
    elif optimizer_id == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate)
    elif optimizer_id == 'adam_amsgrad':
        optimizer = tf.keras.optimizers.Adam(learning_rate, amsgrad=True)
    elif optimizer_id == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(learning_rate)
    elif optimizer_id == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate)
    elif optimizer_id == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate)
    elif optimizer_id == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate)
    else:
        optimizer = tf.keras.optimizers.get(optimizer_id)

    if loss_func_id == 'pooled_binary_crossentropy':
        loss_function = utils.pooled_binary_crossentropy
        metrics=[utils.pooled_binary_crossentropy,
                tf.keras.metrics.BinaryCrossentropy(from_logits=True),
                tf.keras.metrics.BinaryAccuracy(threshold=0),
                tf.keras.metrics.AUC(from_logits=True)]
    elif loss_func_id == 'binary_crossentropy':
        loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metrics=[tf.keras.metrics.BinaryCrossentropy(from_logits=True),
                tf.keras.metrics.BinaryAccuracy(threshold=0),
                tf.keras.metrics.AUC(from_logits=True)]
    else:
        loss_function = tf.keras.losses.get(loss_func_id)
        metrics=[loss_func_id, 'mse', 'mae']

    return (loss_function, optimizer, metrics)

def load_dataset(source:str, batch_size:int, repeat:bool, cache:bool,
                shuffle:bool, peek:bool=False, rotate:str="none"):
    ## Load data
    if len(glob.glob(os.path.join(source, '*.tfrecord'))) > 0:
        FROM_TFRECORDS = True
        records = glob.glob(os.path.join(source, '*.tfrecord'))
        random.shuffle(records)

        ds = tf.data.TFRecordDataset(records, num_parallel_reads=tf.data.AUTOTUNE)

        ex = ds.take(1).map(utils.parse_example).as_numpy_iterator().next()
        record_version = ex['record/version']
        assert record_version == 6, f"Incorrect record version: {record_version} != 6"
        res, n_pins = ex['image/res'], ex['target/n_pins']
    else:
        FROM_TFRECORDS = False
        res, n_pins = 256, 256

        ds = tf.data.Dataset.from_generator(utils.example_generator,
            args=(source, n_pins, res),
            output_signature=(
                tf.TensorSpec(shape=(res, res, 1), dtype=tf.uint8),
                tf.TensorSpec(shape=(n_pins,n_pins), dtype=tf.uint8),
            ))

    if not peek:
        ds = ds.prefetch(tf.data.AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(5*batch_size, seed=42)

    if cache:
        ds = ds.cache()
    if repeat:
        ds = ds.repeat()

    if FROM_TFRECORDS:
        ds = ds.map(utils.get_decoder(res, n_pins, rotate=rotate),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False)

    if peek:
        for img, target in ds.as_numpy_iterator():
            img = img.reshape((res,res))
            utils.plot_example(img, target)
        return (None, None)

    ds = ds.batch(batch_size)

    return ds, {"res": res, "n_pins": n_pins}

def evaluate(saved_model:str, test_data:str, loss_function:str, batch_size:int) -> None:
    '''
    Evaluate loss and accuracy metrics on test data.
    '''

    ds, _ = load_dataset(test_data, batch_size=batch_size, repeat=False,
                        cache=False, shuffle=False, peek=False)
    model = TangledModel()
    model.load_weights(saved_model)

    (loss, _, metrics) = \
        get_compiler_args(loss_function, None, None)

    model.compile(optimizer='adam', loss=loss, metrics=metrics)
    print(model.summary())

    result = model.evaluate(ds, verbose=True)
    print(result)

def fit(train_data:str, val_data:str, output_dir:str, model_name:str=None,
            resume:bool=False, checkpoint_path:str='/tmp/latest', save_format:str='h5',
            loss_function:str='mse', optimizer:str='adam', learning_rate:float=1e-3, batch_size:int=100,
            epochs:int=1, patience:int=-1, train_steps:int=None, val_steps:int=None,
            do_fine_tuning:bool=False,
            use_mixed_precision:bool=False, quantization_aware:bool=False, data_cache:bool=False,
            vis_model:bool=False, summarize:bool=False, peek:bool=False) -> None:
    tf.random.set_seed(42)

    if use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    ds_train, model_info = load_dataset(train_data, batch_size, repeat=(train_steps is not None),
                                        cache=data_cache, shuffle=True, peek=peek, rotate='any')
    ds_val, _ = load_dataset(val_data, batch_size, repeat=(val_steps is not None),
                            cache=data_cache, shuffle=False, peek=peek)

    n_pins = model_info['n_pins']

    if peek:
        return

    if model_name is None:
        model_name = f"b{batch_size}_{loss_function}_{optimizer}_lr{learning_rate:g}"

    if use_mixed_precision:
        model_name += '_f16'

    checkpoint_path = f"{checkpoint_path}.{save_format}"

    # model = TangledModel(n_pins, model_name)
    down_stack = get_down_stack()
    model = SimpleModel(down_stack, model_name)

    if resume:
        # model = tf.keras.models.load_model(checkpoint_path)
        try:
            down_stack.trainable = False
            model.load_weights(checkpoint_path)
        except ValueError:
            down_stack.trainable = True
            model.load_weights(checkpoint_path)

    down_stack.trainable = do_fine_tuning

    if quantization_aware:
        import tensorflow_model_optimization as tfmot
        model = tfmot.quantization.keras.quantize_model(model)

    (loss, optimizer, metrics) = \
        get_compiler_args(loss_function, optimizer, learning_rate)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    timestamp = datetime.datetime.now().strftime(r'%Y%m%d-%H%M%S')

    save_path = os.path.join(output_dir, 'models', f'{timestamp}_{model_name}')
    if vis_model:
        tf.keras.utils.plot_model(model, to_file=save_path+'.png', show_shapes=True)
    print(model.summary())
    print("Base model trainable:",do_fine_tuning)
    if summarize:
        return

    ## Train model
    log_dir = os.path.join(output_dir, 'logs', f'{timestamp}_{model_name}')
    callbacks = [
        # tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_auc',
            save_freq='epoch', save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=500)]

    if patience > 0:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss',
            patience=patience, restore_best_weights=True))

    model.fit(ds_train, validation_data=ds_val, callbacks=callbacks, batch_size=batch_size,
        epochs=epochs, steps_per_epoch=train_steps, validation_steps=val_steps)

    # Save model
    model.save(f"{save_path}.{save_format}", include_optimizer=False)
