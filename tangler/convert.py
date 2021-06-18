# from model import TangledModel
from .simple_model import SimpleModel

def model_to_tflite(input_model_path:str, output_model_path:str) -> None:
    print(f"Converting {input_model_path} to {output_model_path}")
    import tensorflow as tf

    # model = TangledModel()
    model = SimpleModel()
    model.load_weights(input_model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()

    with open(output_model_path, 'wb') as tflite_model_output:
        tflite_model_output.write(tflite_quant_model)

def model_to_tfjs(input_model_path:str, output_model_path:str) -> None:
    print(f"Converting {input_model_path} to {output_model_path}")
    import tensorflowjs as tfjs

    model = SimpleModel()
    model.load_weights(input_model_path)

    tfjs.converters.save_keras_model(model, output_model_path)
