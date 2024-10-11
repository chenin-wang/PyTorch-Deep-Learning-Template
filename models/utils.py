""" Model creation / weight loading / state_dict helpers
    pytorch converter to onnx
"""
import os
from typing import Any, Callable, Dict, Optional, Union, Tuple, List, Sequence
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..loggers.logging_colors import get_logger

logger = get_logger()

try:
    import safetensors.torch

    _has_safetensors = True
except ImportError:
    _has_safetensors = False


__all__ = [
    "clean_state_dict",
    "load_state_dict",
    "load_checkpoint",
    "remap_state_dict",
    "resume_checkpoint",
    "get_node_output_shape",
    "to_numpy",
    "onnx_forward",
    "onnx_export",
    "freeze",
    "unfreeze",
    "allclose",
    "num_parameters",
    "eye_like",
    "interpolate_like",
    "expand_dim",
]


def clean_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict


def add_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    add_state_dict = {}
    for k, v in state_dict.items():
        name = "module." + k
        add_state_dict[name] = v
    return add_state_dict


def load_state_dict(
    checkpoint_path: str,
    use_ema: bool = True,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    if checkpoint_path and os.path.isfile(checkpoint_path):
        # Check if safetensors or not and load weights accordingly
        if str(checkpoint_path).endswith(".safetensors"):
            assert _has_safetensors, "`pip install safetensors` to use .safetensors"
            checkpoint = safetensors.torch.load_file(checkpoint_path, device=device)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)

        state_dict_key = ""
        if isinstance(checkpoint, dict):
            if use_ema and checkpoint.get("state_dict_ema", None) is not None:
                state_dict_key = "state_dict_ema"
            elif use_ema and checkpoint.get("model_ema", None) is not None:
                state_dict_key = "model_ema"
            elif "state_dict" in checkpoint:
                state_dict_key = "state_dict"
            elif "model" in checkpoint:
                state_dict_key = "model"

        state_dict = clean_state_dict(
            checkpoint[state_dict_key] if state_dict_key else checkpoint
        )
        logger.info(
            "Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path)
        )
        return state_dict
    else:
        logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    use_ema: bool = True,
    device: Union[str, torch.device] = "cpu",
    strict: bool = True,
    remap: bool = False,
    filter_fn: Optional[Callable] = None,
):
    if os.path.splitext(checkpoint_path)[-1].lower() in (".npz", ".npy"):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, "load_pretrained"):
            model.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError("Model cannot load numpy checkpoint")
        return

    state_dict = load_state_dict(checkpoint_path, use_ema, device=device)
    if remap:
        state_dict = remap_state_dict(state_dict, model)
    elif filter_fn:
        state_dict = filter_fn(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def remap_state_dict(
    state_dict: Dict[str, Any], model: torch.nn.Module, allow_reshape: bool = True
):
    """remap checkpoint by iterating over state dicts in order (ignoring original keys).
    This assumes models (and originating state dict) were created with params registered in same order.
    """
    out_dict = {}
    for (ka, va), (kb, vb) in zip(model.state_dict().items(), state_dict.items()):
        assert (
            va.numel() == vb.numel()
        ), f"Tensor size mismatch {ka}: {va.shape} vs {kb}: {vb.shape}. Remap failed."
        if va.shape != vb.shape:
            if allow_reshape:
                vb = vb.reshape(va.shape)
            else:
                assert False, f"Tensor shape mismatch {ka}: {va.shape} vs {kb}: {vb.shape}. Remap failed."
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
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            if log_info:
                logger.info("Restoring model state from checkpoint...")
            state_dict = clean_state_dict(checkpoint["state_dict"])
            model.load_state_dict(state_dict)

            if optimizer is not None and "optimizer" in checkpoint:
                if log_info:
                    logger.info("Restoring optimizer state from checkpoint...")
                optimizer.load_state_dict(checkpoint["optimizer"])

            if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
                if log_info:
                    logger.info("Restoring AMP loss scaler state from checkpoint...")
                loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

            if "epoch" in checkpoint:
                resume_epoch = checkpoint["epoch"]
                if "version" in checkpoint and checkpoint["version"] > 1:
                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save

                if log_info:
                    logger.info(
                        "Loaded checkpoint '{}' (epoch {})".format(
                            checkpoint_path, checkpoint["epoch"]
                        )
                    )
        else:
            model.load_state_dict(checkpoint)
            if log_info:
                logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return resume_epoch
    else:
        logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def get_node_output_shape(node):
    return [x.dim_value for x in node.type.tensor_type.shape.dim]


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


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
    device: Union[str, torch.device] = "cpu",
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
            assert hasattr(model, "default_cfg")
            input_size = model.default_cfg.get("input_size")
        example_input = torch.randn(
            (batch_size,) + input_size, requires_grad=training
        ).to(device=device)

    # Run model once before export trace, sets padding for models with Conv2dSameExport. This means
    # that the padding for models with Conv2dSameExport (most models with tf_ prefix) is fixed for
    # the input img_size specified in this script.

    # Opset >= 11 should allow for dynamic padding, however I cannot get it to work due to
    # issues in the tracing of the dynamic padding or errors attempting to export the model after jit
    # scripting it (an approach that should work). Perhaps in a future PyTorch or ONNX versions...
    original_out = model(example_input)
    original_out = model(example_input)
    print("\n=== pytorch model info ===")
    print(f"input_shape:{example_input.shape}")
    print(f"out_shape:{original_out.shape}")

    input_names = input_names or ["input"]
    output_names = output_names or ["output"]

    if dynamic_size:
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}
        dynamic_axes["input"][2] = "height"
        dynamic_axes["input"][3] = "width"
    else:
        dynamic_axes = None

    export_type = torch.onnx.OperatorExportTypes.ONNX

    torch.onnx.export(
        model,  # model being run
        example_input,  # model input (or a tuple for multiple inputs)
        output_file,  # where to save the model (can be a file or file-like object)
        training=training_mode,
        export_params=export_params,  # store the trained parameter weights inside the model file
        verbose=verbose,
        input_names=input_names,  # the model's input names
        output_names=output_names,  # the model's output names
        keep_initializers_as_inputs=keep_initializers,
        dynamic_axes=dynamic_axes,
        opset_version=opset,  # the ONNX version to export the model to
        operator_export_type=export_type,
    )

    if check:
        onnx.save(
            onnx.shape_inference.infer_shapes(onnx.load(output_file)), output_file
        )
        onnx_model = onnx.load(output_file)

        net_output = [
            (node.name, get_node_output_shape(node)) for node in onnx_model.graph.output
        ]

        input_initializer = [node.name for node in onnx_model.graph.initializer]
        net_feed_input = [
            (node.name, get_node_output_shape(node))
            for node in onnx_model.graph.input
            if node.name not in input_initializer
        ]

        print("\n=== onnx model info ===")
        print(f"Inputs:{net_feed_input[0]}")
        print(f"Outputs:{net_output[0]}")

        onnx.checker.check_model(onnx_model, full_check=True)  # assuming throw on error
        # Print a human readable representation of the graph
        # print(onnx.helper.printable_graph(onnx_model.graph))

        if check_forward and not training:
            import numpy as np

            onnx_out = onnx_forward(output_file, example_input.cpu())
            np.testing.assert_array_almost_equal(
                to_numpy(original_out), onnx_out, decimal=3
            )
            print(
                "Exported model has been tested with ONNXRuntime, and the result of assert_array_almost_equal looks good!"
            )
            # compare ONNX Runtime and PyTorch results
            np.testing.assert_allclose(
                to_numpy(original_out), onnx_out, rtol=1e-03, atol=1e-05, verbose=True
            )

            print(
                "Exported model has been tested with ONNXRuntime, and the result of assert_allclose looks good!"
            )


def freeze(net: nn.Module, /) -> nn.Module:
    """Fix all model parameters and prevent training."""
    for p in net.parameters():
        p.requires_grad = False
    return net


def unfreeze(net: nn.Module, /) -> nn.Module:
    """Make all model parameters trainable."""
    for p in net.parameters():
        p.requires_grad = True
    return net


def allclose(net1: nn.Module, net2: nn.Module, /) -> bool:
    """Check if two networks are equal."""
    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        try:
            if not p1.allclose(p2):
                return False
        except RuntimeError:  # Non-matching parameter shapes.
            return False
    return True


def num_parameters(net: nn.Module, /) -> int:
    """Get number of trainable parameters in a network."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def eye_like(x: Tensor, /) -> Tensor:
    """Create an Identity matrix of the same dtype and size as the input.

    NOTE: The input can be of any shape, expect the final two dimensions, which must be square.

    :param x: (Tensor) (*, n, n) Input reference tensor, where `*` can be any size (including zero).
    :return: (Tensor) (*, n, n) Identity matrix with the same dtype and size as the input.
    """
    ndim = x.ndim
    if ndim < 2:
        raise ValueError(f'Input must have at least two dimensions! Got "{ndim}"')

    n, n2 = x.shape[-2], x.shape[-1]
    if n != n2:
        raise ValueError(
            f'Input last two dimensions must be square (*, n, n)! Got "{x.shape}"'
        )

    view = [1] * (ndim - 2) + [n, n]  # (*, n, n)
    I = torch.eye(n, dtype=x.dtype, device=x.device).view(view).expand_as(x).clone()
    return I


def interpolate_like(
    input: Tensor, /, other: Tensor, mode: str = "nearest", align_corners: bool = False
) -> Tensor:
    """Interpolate to match the size of `other` tensor."""
    if mode == "nearest":
        align_corners = None
    return F.interpolate(
        input, size=other.shape[-2:], mode=mode, align_corners=align_corners
    )


def expand_dim(
    x: Tensor,
    /,
    num: Union[int, Sequence[int]],
    dim: Union[int, Sequence[int]] = 0,
    insert: bool = False,
) -> Tensor:
    """Expand the specified input tensor dimensions, inserting new ones if required.

    >>> expand_dim(torch.rand(1, 1, 1), num=5, dim=1, insert=False)             # (1, 1, 1) -> (1, 5, 1)
    >>> expand_dim(torch.rand(1, 1, 1), num=5, dim=1, insert=True)              # (1, 1, 1) -> (1, 5, 1, 1)
    >>> expand_dim(torch.rand(1, 1, 1), num=(5, 3), dim=(0, 1), insert=False)   # (1, 1, 1) -> (5, 3, 1)
    >>> expand_dim(torch.rand(1, 1, 1), num=(5, 3), dim=(0, 1), insert=True)    # (1, 1, 1) -> (5, 3, 1, 1, 1)

    :param x: (Tensor) (*) Input tensor of any shape.
    :param num: (int|Sequence[int]) Expansion amount for the target dimension(s).
    :param dim: (int|Sequence[int]) Dimension(s) to expand.
    :param insert: (bool) If `True`, insert a new dimension at the specified location(s).
    :return: (Tensor) (*, num, *) Expanded tensor at the given location(s).
    """
    if isinstance(num, int):
        if isinstance(dim, int):
            num, dim = [num], [dim]  # (1, 1) -> ([1], [1])
        else:
            num = [num] * len(dim)  # (1, [1, 2]) -> ([1, 1], [1, 2])
    elif len(num) != len(dim):
        raise ValueError(
            f"Non-matching expansion and dims. ({len(num)} vs. {len(dim)})"
        )

    # Add new dims to expand.
    for d in dim if insert else ():
        x = x.unsqueeze(d)

    # Create target shape, leaving other dims unchanged (-1).
    sizes = [-1] * x.ndim
    for n, d in zip(num, dim):
        sizes[d] = n

    return x.expand(sizes)
