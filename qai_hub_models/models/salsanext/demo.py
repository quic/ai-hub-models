# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from qai_hub_models.models.salsanext.app import SalsaNextApp
from qai_hub_models.models.salsanext.model import INPUT_LIDAR_ADDRESS, SalsaNext
from qai_hub_models.utils.args import get_model_cli_parser, get_on_device_demo_parser


def SalsaNext_demo(
    model_type: type[SalsaNext],
    deafault_input: str,
    is_test: bool = False,
):
    # Demo parameters
    parser = get_model_cli_parser(model_type)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--input_lidar",
        type=str,
        default=deafault_input,
        help="lidar bin file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    model = model_type.from_pretrained()
    app = SalsaNextApp(model)
    pred_np = app.detect(args.input_lidar)
    if not is_test:
        print("output: ", pred_np)


def main(is_test: bool = False) -> None:
    SalsaNext_demo(SalsaNext, str(INPUT_LIDAR_ADDRESS), is_test)


if __name__ == "__main__":
    main()
