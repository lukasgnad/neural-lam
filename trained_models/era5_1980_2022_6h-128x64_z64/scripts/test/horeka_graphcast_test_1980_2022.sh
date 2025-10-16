python train_model.py\
    --name 1980_2022_6h_128_64_z64_graphcast_test_3\
    --dataset global_era5_1980_2022_6h-128x64_equiangular_with_poles_conservative\
    --dataset_path $(ws_find neural_lam)/data/data\
    --model graphcast\
    --n_workers 4\
    --n_example_pred 0\
    --eval_leads 40\
    --hidden_dim 64\
    --processor_layers 4\
    --batch_size 1\
    --graph global_multilevel_128x64\
    --load trained_models/era5_1980_2022_6h-128x64_z64/checkpoints/1980_2022_6h_128_64_z64_03/last.ckpt\
    --periods "train:1980-01-01,2018-12-31;val:2019-01-01,2020-12-31;test:2021-01-01,2022-12-31"\
    --eval test

python train_model.py\
    --name persistence_era5_1990_2022_6h_128_64_z64_test\
    --dataset global_era5_1980_2022_6h-128x64_equiangular_with_poles_conservative\
    --dataset_path $(ws_find neural_lam)/data/data\
    --model persistence\
    --n_workers 40\
    --n_example_pred 0\
    --eval_leads 40\
    --hidden_dim 64\
    --processor_layers 4\
    --batch_size 1\
    --graph global_multilevel_128x64\
    --periods "train:1990-01-01,2015-12-31;val:2016-01-01,2017-12-31;test:2021-01-01,2022-12-31"\
    --eval test