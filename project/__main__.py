from argparse import ArgumentParser
from pathlib import Path
from pyargwriter import api
from project.utils.parser import setup_entrypoint_parser
from project.entrypoint import Entrypoint
from project.utils.parser import setup_parser


def execute(args: dict) -> bool:
    module = Entrypoint()
    _, command_parser = setup_entrypoint_parser(ArgumentParser())
    match args["command"]:
        case "generate-dataset":
            module.generate_dataset()

        case "train":
            api.hydra_plugin.hydra_wrapper(
                module.train,
                args,
                command_parser["train"],
                config_var_name="config",
                config_path=str(Path.cwd().joinpath("config")),
                config_name="train_conf",
                version_base=None,
            )

        case "split-dataset":
            module.split_dataset(
                data_dir=args["data_dir"],
                output_dir=args["output_dir"],
                train_ratio=args["train_ratio"],
                val_ratio=args["val_ratio"],
                test_ratio=args["test_ratio"],
                random_seed=args["random_seed"],
                log_level=args["log_level"],
            )

        case "compare-models":
            module.compare_models(
                input_size=args["input_size"],
                output_size=args["output_size"],
                hidden_size=args["hidden_size"],
                n_g_layers=args["n_g_layers"],
                n_r_layers=args["n_r_layers"],
                n_o_layers=args["n_o_layers"],
            )

        case _:
            return False

    return True


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(description="--no-documentation-exists--")

    parser = setup_parser(parser)

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    args_dict = vars(args)
    if not execute(args_dict):
        parser.print_usage()


if __name__ == "__main__":
    main()
