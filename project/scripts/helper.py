from project.gatel0rd.lightning_module import compare_model_versions
import json


def compare_versions(
    input_size: int,
    output_size: int = -1,
    hidden_size: int = 128,
    n_g_layers: int = 1,
    n_r_layers: int = 1,
    n_o_layers: int = 1,
):
    
    # print hyperparameters
    print("Comparing model versions with the following hyperparameters:")
    print(f"Input size: {input_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"Output size: {output_size}")
    print(f"Number of G layers: {n_g_layers}")
    print(f"Number of R layers: {n_r_layers}")
    print(f"Number of O layers: {n_o_layers}")
    
    comparison = compare_model_versions(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        n_g_layers=n_g_layers,
        n_r_layers=n_r_layers,
        n_o_layers=n_o_layers,
        init_net_kwargs={},
        pre_net_kwargs={},
        out_net_kwargs={},
    )

    # make pretty print
    comparison_str = json.dumps(comparison, indent=4)
    print(comparison_str)
