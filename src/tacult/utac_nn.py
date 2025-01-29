from .network import UtacNNet
from .base.nn_wrapper import NNetWrapper


class UtacNN(NNetWrapper):
    def __init__(self, args):
        super().__init__(UtacNNet(onnx_export=args.onnx_export, **args.model_args), args)
        
