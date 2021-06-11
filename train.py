import datetime, os, random
import tensorflow as tf
import glob

from model import TangledModel
import utils

def do_train(train_data:str, val_data:str, output_dir:str, model_name:str=None,
            resume:bool=False, checkpoint_path:str='/tmp/latest', save_format:str='h5',
            loss_function:str='mse', optimizer:str='adam', learning_rate:float=1e-3, batch_size:int=100,
            epochs:int=1, patience:int=-1, train_steps:int=None, val_steps:int=None,
            use_mixed_precision:bool=False, data_cache:bool=False, vis_model:bool=False,
            summarize:bool=False, debug:bool=False, peek:bool=False) -> None:
    tf.random.set_seed(42)

    if use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    ## Load data
    if len(glob.glob(os.path.join(train_data, '*.tfrecord'))) > 0:
        FROM_TFRECORDS = True
        train_records = glob.glob(os.path.join(train_data, '*.tfrecord'))
        val_records = glob.glob(os.path.join(val_data, '*.tfrecord'))

        random.shuffle(train_records)
        random.shuffle(val_records)

        ds_train = tf.data.TFRecordDataset(train_records, num_parallel_reads=tf.data.AUTOTUNE)
        ds_val = tf.data.TFRecordDataset(val_records, num_parallel_reads=tf.data.AUTOTUNE)

        ex = ds_train.take(1).map(utils.parse_example).as_numpy_iterator().next()
        record_version = ex['record/version']
        assert record_version == 6, f"Incorrect record version: {record_version} != 6"
        res, n_pins = ex['image/res'], ex['target/n_pins']
    else:
        FROM_TFRECORDS = False
        res, n_pins = 256, 256

        ds_train = tf.data.Dataset.from_generator(utils.example_generator,
            args=(train_data, n_pins, res),
            output_signature=(
                tf.TensorSpec(shape=(res, res, 1), dtype=tf.uint8),
                tf.TensorSpec(shape=(n_pins,n_pins), dtype=tf.uint8),
            ))
        ds_val = tf.data.Dataset.from_generator(utils.example_generator,
            args=(val_data, n_pins, res),
            output_signature=(
                tf.TensorSpec(shape=(res,res,1), dtype=tf.uint8),
                tf.TensorSpec(shape=(n_pins,n_pins), dtype=tf.uint8),
            ))

    if not (debug or peek):
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
        ds_train = ds_train.shuffle(5*batch_size, seed=42)
        ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    if data_cache:
        ds_train = ds_train.cache()
    if train_steps is not None:
        ds_train = ds_train.repeat()
    if data_cache:
        ds_val = ds_val.cache()
    if val_steps is not None:
        ds_val = ds_val.repeat()

    if FROM_TFRECORDS:
        ds_train = ds_train.map(utils.get_decoder(res, n_pins, rotate='none'),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False)
        ds_val = ds_val.map(utils.get_decoder(res, n_pins, rotate='none'),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False)

    if peek:
        for img, target in ds_train.as_numpy_iterator():
            img = img.reshape((res,res))
            utils.plot_example(img, target)
        return

    ds_train = ds_train.batch(batch_size)
    ds_val = ds_val.batch(batch_size)

    if model_name is None:
        model_name = f"b{batch_size}_{loss_function}_{optimizer}_lr{learning_rate:g}"

    if use_mixed_precision:
        model_name += '_f16'

    checkpoint_path = f"{checkpoint_path}.{save_format}"


    model = TangledModel(n_pins, model_name)

    if resume:
        # model = tf.keras.models.load_model(checkpoint_path)
        model.load_weights(checkpoint_path)

    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate)
    elif optimizer == 'adam_amsgrad':
        opt = tf.keras.optimizers.Adam(learning_rate, amsgrad=True)
    elif optimizer == 'nadam':
        opt = tf.keras.optimizers.Nadam(learning_rate)
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate)
    elif optimizer == 'adadelta':
        opt = tf.keras.optimizers.Adadelta(learning_rate)
    elif optimizer == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate)
    else:
        opt = optimizer

    if loss_function == 'pooled_binary_crossentropy':
        loss_function = utils.pooled_binary_crossentropy
        metrics=[tf.keras.metrics.BinaryCrossentropy(from_logits=True),
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.AUC(from_logits=True)]
    elif loss_function == 'binary_crossentropy':
        loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metrics=[tf.keras.metrics.BinaryCrossentropy(from_logits=True),
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.AUC(from_logits=True)]
    elif loss_function == 'cosine_similarity':
        metrics=['cosine_similarity', 'mse', 'mae']
    else:
        metrics=['mse', 'mae']

    model.compile(optimizer=opt, loss=loss_function, metrics=metrics)

    timestamp = datetime.datetime.now().strftime(r'%Y%m%d-%H%M%S')

    save_path = os.path.join(output_dir, 'models', f'{timestamp}_{model_name}')
    if vis_model:
        tf.keras.utils.plot_model(model, to_file=save_path+'.png', show_shapes=True)
    print(model.summary())
    if summarize:
        return

    ## Train model
    log_dir = os.path.join(output_dir, 'logs', f'{timestamp}_{model_name}')
    callbacks = [
        # tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss',
            save_freq='epoch', save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=500)]

    if patience > 0:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss',
            patience=patience, restore_best_weights=True))

    model.fit(ds_train, validation_data=ds_val, callbacks=callbacks, batch_size=batch_size,
        epochs=epochs, steps_per_epoch=train_steps, validation_steps=val_steps)

    # Save model
    model.save(f"{save_path}.{save_format}", include_optimizer=False)
