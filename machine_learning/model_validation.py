import onnx


def validate_model():
    print('validating model using onnx model-checking')
    onnx_model_deep = onnx.load("torch_model.onnx")
    onnx.checker.check_model(onnx_model_deep)
    print('models validated')


if __name__ == '__main__':
    validate_model()