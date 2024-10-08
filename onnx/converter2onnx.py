#!/usr/bin/env python

import torch
from ..models.modelutils import load_checkpoint, add_state_dict, onnx_export
import os
import argparse
from unet import UNet
from PIL import Image
import numpy as np
from data.Dataset import ValDataset
from pathlib import Path
from ..loggers.logging_colors import logger


def main(args):
    PATH = args.path
    # model reparameterize
    loaded_model = UNet()  # 注意这里需要对模型结构有定义
    load_checkpoint(model=loaded_model, checkpoint_path=PATH)
    state_dict_temp = loaded_model.state_dict()
    state_dict_with_name = add_state_dict(state_dict_temp)
    SAVE_NAME = os.path.splitext(PATH)[0]
    SAVE_NAME_DP = os.path.splitext(SAVE_NAME)[0] + "_dp.pth"
    torch.save(state_dict_temp, SAVE_NAME)
    torch.save(state_dict_with_name, SAVE_NAME_DP)
    print(f"save normal path:{SAVE_NAME}")
    print(f"save dp path:{SAVE_NAME_DP}")

    if torch.cuda.is_available():
        net_dp = torch.nn.DataParallel(UNet(), [0]).cuda()
        net_dp.load_state_dict(torch.load(SAVE_NAME_DP), strict=True)
        print("load DP model ok")
        torch.cuda.empty_cache()
        del net_dp

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = UNet().to(device)
    net.load_state_dict(torch.load(SAVE_NAME), strict=True)
    net_state_dict = net.state_dict()
    print("load normal model ok")

    # input example
    # img = Image.open("/data/wzc/bar_fig_1/imgs/a20240520__0.png")
    # img = torch.from_numpy(np.array(img))
    dir_img = Path(args.example_img_path)
    dataset = ValDataset(dir_img, augment=True)
    dataset.unet_data.augment = False
    example_input = dataset[0][0].unsqueeze(0).to(device=device)
    print(f"example_input.shape:{example_input.shape}")

    # onnx export
    OUTPUT_PATH = os.path.splitext(SAVE_NAME)[0] + ".onnx"
    print(f"ONNX_PATH:{OUTPUT_PATH}")
    onnx_export(
        model=net,
        output_file=OUTPUT_PATH,
        opset=11,
        dynamic_size=False,
        example_input=example_input,
        keep_initializers=False,
        check_forward=True,
        training=False,
        verbose=False,
        input_size=(3, args.row, args.col),
        batch_size=1,
        export_params=True,  # store the trained parameter weights inside the model file
        device=device,
    )

    import onnx
    from onnxsim import simplify
    from onnxsim import model_info

    # load your predefined ONNX model
    model = onnx.load(OUTPUT_PATH)
    # convert model
    print("Simplifying...")
    model_simp, check_ok = simplify(model)

    onnx.save(model_simp, OUTPUT_PATH)
    if check_ok:
        print("Finish! Here is the difference:")
        model_info.print_simplifying_info(model, model_simp)
    else:
        print(
            'Check failed. Please be careful to use the simplified model, or try specifying "--skip-fuse-bn" or "--skip-optimization" (run "onnxsim -h" for details).'
        )
        print("Here is the difference after simplification:")
        model_info.print_simplifying_info(model, model_simp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert unet model to onnx")
    parser.add_argument(
        "--path",
        default="/home/wzc/project/uNet/tr/exp6/checkpoint-14.pth.tar",
        type=str,
        help="Absolute path to load PyTorch model and export onnx model",
    )
    parser.add_argument(
        "--example_img_path",
        default="/home/wzc/project/uNet/sample_train_data",
        type=str,
        help="example input path ",
    )
    parser.add_argument("--row", default=540, type=int, help="Input image rows")
    parser.add_argument("--col", default=960, type=int, help="Input image columns")
    args = parser.parse_args()
    main(args)
