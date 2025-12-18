python train_model.py\
    --name 1980_2022_6h_128_64_graphcast_test_3\
    --dataset global_era5_1980_2022_6h-128x64_equiangular_with_poles_conservative\
    --dataset_path $(ws_find neural_lam)/data/data\
    --model graphcast\
    --n_workers 4\
    --n_example_pred 0\
    --eval_leads 40\
    --hidden_dim 128\
    --processor_layers 4\
    --batch_size 1\
    --graph global_multilevel_128x64\
    --load trained_models/era5_1980_2022_6h-128x64_z128/checkpoints/1980_2022_6h_128_64_03/last.ckpt\
    --periods "train:1980-01-01,2018-12-31;val:2019-01-01,2020-12-31;test:2021-01-01,2022-12-31"\
    --eval test

