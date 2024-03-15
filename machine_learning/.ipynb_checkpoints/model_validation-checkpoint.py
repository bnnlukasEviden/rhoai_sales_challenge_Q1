import onnx


def validate_model():
    print('validating model using onnx model-checking')
    onnx_model_deep = onnx.load("best_deep_model.onnx")
    onnx_model_wide = onnx.load("best_wide_model.onnx")
    onnx.checker.check_model(onnx_model_deep)
    onnx.checker.check_model(onnx_model_wide)
    print('models validated')


if __name__ == '__main__':
    validate_model()