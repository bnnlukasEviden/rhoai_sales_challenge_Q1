import onnx


def validate_model():
    print('validating model using onnx model-checking')
    onnx_model = onnx.load("./models/best_deep_model.onnx")
    onnx.checker.check_model(onnx_model)
    print('model validated')


if __name__ == '__main__':
    validate_model()