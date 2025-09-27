from pyargwriter.decorator import add_hydra
from omegaconf import DictConfig


class Entrypoint:
    def generate_dataset(self):
        pass

    @add_hydra(
        config_var_name="config",
        config_path="config",
        config_name="train_conf",
        version_base=None,
    )
    def train(self, config: DictConfig, log_level: str = "INFO"):
        from project.scripts.train import train

        train(config, log_level)

    def split_dataset(
        self,
        data_dir: str,
        output_dir: str = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        log_level: str = "INFO",
    ):
        from project.scripts.data_handler import split_and_save_dataset

        split_and_save_dataset(
            data_dir,
            train_ratio,
            val_ratio,
            test_ratio,
            output_dir,
            random_seed,
            log_level,
        )

    def compare_models(
        self,
        input_size=10,
        output_size=10,
        hidden_size=64,
        n_g_layers=2,
        n_r_layers=2,
        n_o_layers=2,
    ):
        from project.scripts.helper import compare_versions

        compare_versions(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            n_g_layers=n_g_layers,
            n_r_layers=n_r_layers,
            n_o_layers=n_o_layers,
        )
