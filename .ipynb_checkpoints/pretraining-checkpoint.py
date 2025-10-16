from argparse import ArgumentParser
import create_global_mesh as Mesh
import create_global_grid_features as Features
import create_global_forcing as Forcing
import create_parameter_weights as Weights
import time

def main():
    """
    Pre-compute all other scripts
    """
    parser = ArgumentParser(description="Training arguments")
    parser.add_argument(
        "--dataset",
        type=str,
        default=Mesh.DEFAULT_DATASET,
        help="Dataset to create grid features for "
        "(default: global_example_era5)",
    )
    parser.add_argument(
        "--plot",
        type=int,
        default=Mesh.DEFAULT_PLOT,
        help="If fields should be plotted " "(default: 0 (false))",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default=Mesh.DEFAULT_GRAPH,
        help="Name to save graph as (default: global_multiscale)",
    )
    parser.add_argument(
        "--splits",
        default=Mesh.DEFAULT_SPLITS,
        type=int,
        help="Number of splits to triangular mesh (default: 3)",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=Mesh.DEFAULT_LEVELS,
        help="Number of levels to keep, from finest upwards "
        "(default: None (keep all))",
    )
    parser.add_argument(
        "--hierarchical",
        type=int,
        default=Mesh.DEFAULT_HIERARCHICAL,
        help="Generate hierarchical mesh graph (default: 0, no)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=Weights.DEFAULT_BATCH_SIZE,
        help="Batch size when iterating over the dataset",
    )
    parser.add_argument(
        "--step_length",
        type=int,
        default=Weights.DEFAULT_LAM_STEP_LENGTH,
        help="Step length in hours to consider single time step (for LAM only)"
        " (default: 3)",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=Weights.DEFAULT_N_WORKERS,
        help="Number of workers in data loader (default: 4)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=Mesh.DEFAULT_DATASET_PATH,
        help="The path to the folder containing the dataset (default \'data\')",
    )
    parser.add_argument(
        "--periods",
        type=str,
        default=None,
        help=(
            "Periods for train/val/test as: "
            "train:<start>,<end>;val:<start>,<end>;test:<start>,<end>"
        )
    )


    args = parser.parse_args()
    
    
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    # print("")
    # print("-- Creating mesh --")
    # time.sleep(.2)
    # Mesh.create_global_mesh(args.dataset, args.graph, args.plot, args.splits, args.levels, args.hierarchical, args.dataset_path)
    # print("")
    # print("-- Creating features --")
    # time.sleep(.2)
    # Features.create_global_grid_features(args.dataset, args.plot, args.dataset_path)
    # print("")
    print("-- Creating forcings --")
    time.sleep(.2)
    Forcing.create_global_forcing(args.dataset, args.plot, args.dataset_path)
    print("")
    # print("-- Creating weights --")
    # time.sleep(.2)
    # Weights.create_parameter_weights(args.dataset, args.batch_size, args.step_length, args.n_workers, args.dataset_path,periods=args.periods)


if __name__ == "__main__":
    main()