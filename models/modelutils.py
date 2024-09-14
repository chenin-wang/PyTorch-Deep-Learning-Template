""" Model creation / weight loading / state_dict helpers
    pytorch converter to onnx
"""
import logging
import os
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Union, Tuple, List
import torch
from ..loggers.logging_colors import get_logger

logger = get_logger(__name__)

try:
    import safetensors.torch
    _has_safetensors = True
except ImportError:
    _has_safetensors = False

_logger = logging.getLogger(__name__)

__all__ = ['clean_state_dict', 'load_state_dict', 'load_checkpoint', 'remap_state_dict', 'resume_checkpoint']


def clean_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict

def add_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    add_state_dict = {}
    for k, v in state_dict.items():
        name = 'module.'+k
        add_state_dict[name] = v
    return add_state_dict

def load_state_dict(
        checkpoint_path: str,
        use_ema: bool = True,
        device: Union[str, torch.device] = 'cpu',
) -> Dict[str, Any]:
    if checkpoint_path and os.path.isfile(checkpoint_path):
        # Check if safetensors or not and load weights accordingly
        if str(checkpoint_path).endswith(".safetensors"):
            assert _has_safetensors, "`pip install safetensors` to use .safetensors"
            checkpoint = safetensors.torch.load_file(checkpoint_path, device=device)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)

        state_dict_key = ''
        if isinstance(checkpoint, dict):
            if use_ema and checkpoint.get('state_dict_ema', None) is not None:
                state_dict_key = 'state_dict_ema'
            elif use_ema and checkpoint.get('model_ema', None) is not None:
                state_dict_key = 'model_ema'
            elif 'state_dict' in checkpoint:
                state_dict_key = 'state_dict'
            elif 'model' in checkpoint:
                state_dict_key = 'model'

        state_dict = clean_state_dict(checkpoint[state_dict_key] if state_dict_key else checkpoint)
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_checkpoint(
        model: torch.nn.Module,
        checkpoint_path: str,
        use_ema: bool = True,
        device: Union[str, torch.device] = 'cpu',
        strict: bool = True,
        remap: bool = False,
        filter_fn: Optional[Callable] = None
):
    if os.path.splitext(checkpoint_path)[-1].lower() in ('.npz', '.npy'):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError('Model cannot load numpy checkpoint')
        return

    state_dict = load_state_dict(checkpoint_path, use_ema, device=device)
    if remap:
        state_dict = remap_state_dict(state_dict, model)
    elif filter_fn:
        state_dict = filter_fn(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def remap_state_dict(
        state_dict: Dict[str, Any],
        model: torch.nn.Module,
        allow_reshape: bool = True
):
    """ remap checkpoint by iterating over state dicts in order (ignoring original keys).
    This assumes models (and originating state dict) were created with params registered in same order.
    """
    out_dict = {}
    for (ka, va), (kb, vb) in zip(model.state_dict().items(), state_dict.items()):
        assert va.numel() == vb.numel(), f'Tensor size mismatch {ka}: {va.shape} vs {kb}: {vb.shape}. Remap failed.'
        if va.shape != vb.shape:
            if allow_reshape:
                vb = vb.reshape(va.shape)
            else:
                assert False,  f'Tensor shape mismatch {ka}: {va.shape} vs {kb}: {vb.shape}. Remap failed.'
        out_dict[ka] = vb
    return out_dict


def resume_checkpoint(
        model: torch.nn.Module,
        checkpoint_path: str,
        optimizer: torch.optim.Optimizer = None,
        loss_scaler: Any = None,
        log_info: bool = True,
):
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            if log_info:
                _logger.info('Restoring model state from checkpoint...')
            state_dict = clean_state_dict(checkpoint['state_dict'])
            model.load_state_dict(state_dict)

            if optimizer is not None and 'optimizer' in checkpoint:
                if log_info:
                    _logger.info('Restoring optimizer state from checkpoint...')
                optimizer.load_state_dict(checkpoint['optimizer'])

            if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
                if log_info:
                    _logger.info('Restoring AMP loss scaler state from checkpoint...')
                loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save

                if log_info:
                    _logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            if log_info:
                _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return resume_epoch
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()

def get_node_output_shape(node):
    return [x.dim_value for x in node.type.tensor_type.shape.dim]

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def onnx_forward(onnx_file, example_input):
    import onnxruntime

    sess_options = onnxruntime.SessionOptions()
    session = onnxruntime.InferenceSession(onnx_file, sess_options)
    input_name = session.get_inputs()[0].name
    output = session.run([], {input_name: example_input.numpy()})
    output = output[0]
    return output

def onnx_export(
        model: torch.nn.Module,
        output_file: str,
        example_input: Optional[torch.Tensor] = None,
        training: bool = False,
        verbose: bool = False,
        check: bool = True,
        check_forward: bool = False,
        batch_size: int = 64,
        input_size: Tuple[int, int, int] = None,
        opset: Optional[int] = None,
        dynamic_size: bool = False,
        keep_initializers: Optional[bool] = None,
        input_names: List[str] = None,
        output_names: List[str] = None,
        export_params: bool = False,
        device: Union[str, torch.device] = 'cpu',
):
    import onnx

    if training:
        training_mode = torch.onnx.TrainingMode.TRAINING
        model.train()
    else:
        training_mode = torch.onnx.TrainingMode.EVAL
        model.eval()

    if example_input is None:
        if not input_size:
            assert hasattr(model, 'default_cfg')
            input_size = model.default_cfg.get('input_size')
        example_input = torch.randn((batch_size,) + input_size, requires_grad=training).to(device=device)

    # Run model once before export trace, sets padding for models with Conv2dSameExport. This means
    # that the padding for models with Conv2dSameExport (most models with tf_ prefix) is fixed for
    # the input img_size specified in this script.

    # Opset >= 11 should allow for dynamic padding, however I cannot get it to work due to
    # issues in the tracing of the dynamic padding or errors attempting to export the model after jit
    # scripting it (an approach that should work). Perhaps in a future PyTorch or ONNX versions...
    original_out = model(example_input)
    original_out = model(example_input)
    print('\n=== pytorch model info ===')
    print(f"input_shape:{example_input.shape}")
    print(f"out_shape:{original_out.shape}")

    input_names = input_names or ["input"]
    output_names = output_names or ["output"]

    if dynamic_size:
        dynamic_axes = {'input': {0: 'batch'}, 'output': {0: 'batch'}}
        dynamic_axes['input'][2] = 'height'
        dynamic_axes['input'][3] = 'width'
    else:
        dynamic_axes= None

    export_type = torch.onnx.OperatorExportTypes.ONNX

    torch.onnx.export(
        model, # model being run
        example_input, # model input (or a tuple for multiple inputs)
        output_file, # where to save the model (can be a file or file-like object)
        training=training_mode,
        export_params= export_params, # store the trained parameter weights inside the model file
        verbose=verbose,
        input_names=input_names,  # the model's input names
        output_names=output_names, # the model's output names
        keep_initializers_as_inputs=keep_initializers,
        dynamic_axes=dynamic_axes,
        opset_version=opset,  # the ONNX version to export the model to
        operator_export_type=export_type
    )

    if check:
        onnx.save(onnx.shape_inference.infer_shapes(onnx.load(output_file)),output_file)
        onnx_model = onnx.load(output_file)

        net_output = [(node.name, get_node_output_shape(node)) for node in onnx_model.graph.output]

        input_initializer = [node.name for node in onnx_model.graph.initializer]
        net_feed_input = [(node.name, get_node_output_shape(node)) for node in onnx_model.graph.input
                        if node.name not in input_initializer]

        print('\n=== onnx model info ===')
        print(f'Inputs:{net_feed_input[0]}')
        print(f'Outputs:{net_output[0]}')

        onnx.checker.check_model(onnx_model, full_check=True)  # assuming throw on error
        # Print a human readable representation of the graph
        # print(onnx.helper.printable_graph(onnx_model.graph))

        if check_forward and not training:
            import numpy as np
            onnx_out = onnx_forward(output_file, example_input.cpu())
            np.testing.assert_array_almost_equal(to_numpy(original_out), onnx_out, decimal=3)
            print("Exported model has been tested with ONNXRuntime, and the result of assert_array_almost_equal looks good!")
            # compare ONNX Runtime and PyTorch results
            np.testing.assert_allclose(to_numpy(original_out), onnx_out, rtol=1e-03, atol=1e-05,verbose=True)
            
            print("Exported model has been tested with ONNXRuntime, and the result of assert_allclose looks good!")
        
