from argparse import ArgumentParser
from typing import Tuple, Dict, List


def add_split_dataset_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--data-dir",
        help="--no-documentation-exists--",
        dest="data_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        help="--no-documentation-exists--",
        dest="output_dir",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--train-ratio",
        help="--no-documentation-exists--",
        dest="train_ratio",
        type=float,
        default=0.7,
        required=False,
    )
    parser.add_argument(
        "--val-ratio",
        help="--no-documentation-exists--",
        dest="val_ratio",
        type=float,
        default=0.15,
        required=False,
    )
    parser.add_argument(
        "--test-ratio",
        help="--no-documentation-exists--",
        dest="test_ratio",
        type=float,
        default=0.15,
        required=False,
    )
    parser.add_argument(
        "--random-seed",
        help="--no-documentation-exists--",
        dest="random_seed",
        type=int,
        default=42,
        required=False,
    )
    return parser


from pyargwriter.api.hydra_plugin import add_hydra_parser


def add_train_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--log-level",
        help="--no-documentation-exists--",
        dest="log_level",
        type=str,
        default="INFO",
        required=False,
    )
    return parser


def add_generate_dataset_args(parser: ArgumentParser) -> ArgumentParser:
    return parser


def setup_entrypoint_parser(
    parser: ArgumentParser,
) -> Tuple[ArgumentParser, Dict[str, ArgumentParser]]:
    subparser = {}
    command_subparser = parser.add_subparsers(dest="command", title="command")
    generate_dataset = command_subparser.add_parser(
        "generate-dataset", help="--no-documentation-exists--"
    )
    generate_dataset = add_generate_dataset_args(generate_dataset)
    subparser["generate_dataset"] = generate_dataset
    train = command_subparser.add_parser("train", help="--no-documentation-exists--")
    train = add_train_args(train)
    train = add_hydra_parser(train)
    subparser["train"] = train
    split_dataset = command_subparser.add_parser(
        "split-dataset", help="--no-documentation-exists--"
    )
    split_dataset = add_split_dataset_args(split_dataset)
    subparser["split_dataset"] = split_dataset
    return parser, subparser


def setup_parser(parser: ArgumentParser) -> ArgumentParser:
    parser, _ = setup_entrypoint_parser(parser)
    return parser
