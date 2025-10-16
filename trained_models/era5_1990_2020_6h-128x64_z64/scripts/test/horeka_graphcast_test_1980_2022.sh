python train_model.py\
    --name 1990_2020_6h_128_64_z64_era5_graphcast_test_1_long\
    --dataset global_era5_1990_2020_6h-128x64_equiangular_with_poles_conservative\
    --dataset_path $(ws_find neural_lam)/data/data\
    --model graphcast\
    --n_workers 40\
    --n_example_pred 0\
    --eval_leads 40\
    --hidden_dim 64\
    --processor_layers 4\
    --batch_size 1\
    --graph global_multilevel_era5\
    --load trained_models/era5_1990_2020_6h-128x64_z64/checkpoints/1990_2020_6h_128_64_z64_era5_01/last.ckpt\
    --periods "train:1990-01-01,2015-12-31;val:2016-01-01,2017-12-31;test:2016-01-01,2019-12-31"\
    --wandb_output trained_models/era5_1990_2020_6h-128x64_z64/test_runs/wandb_1990_2020_6h_128_64_z64_era5_graphcast_test_01_long\
    --eval test

python train_model.py\
    --name 1990_2020_6h_128_64_z64_era5_graphcast_test_2_long\
    --dataset global_era5_1990_2020_6h-128x64_equiangular_with_poles_conservative\
    --dataset_path $(ws_find neural_lam)/data/data\
    --model graphcast\
    --n_workers 40\
    --n_example_pred 0\
    --eval_leads 40\
    --hidden_dim 64\
    --processor_layers 4\
    --batch_size 1\
    --graph global_multilevel_era5\
    --load trained_models/era5_1990_2020_6h-128x64_z64/checkpoints/1990_2020_6h_128_64_z64_era5_02/last.ckpt\
    --periods "train:1990-01-01,2015-12-31;val:2016-01-01,2017-12-31;test:2016-01-01,2019-12-31"\
    --wandb_output trained_models/era5_1990_2020_6h-128x64_z64/test_runs/wandb_1990_2020_6h_128_64_z64_era5_graphcast_test_02_long\
    --eval test

python train_model.py\
    --name 1990_2020_6h_128_64_z64_era5_graphcast_test_3_long\
    --dataset global_era5_1990_2020_6h-128x64_equiangular_with_poles_conservative\
    --dataset_path $(ws_find neural_lam)/data/data\
    --model graphcast\
    --n_workers 40\
    --n_example_pred 0\
    --eval_leads 40\
    --hidden_dim 64\
    --processor_layers 4\
    --batch_size 1\
    --graph global_multilevel_era5\
    --load trained_models/era5_1990_2020_6h-128x64_z64/checkpoints/1990_2020_6h_128_64_z64_era5_03/last.ckpt\
    --wandb_output trained_models/era5_1990_2020_6h-128x64_z64/test_runs/wandb_1990_2020_6h_128_64_z64_era5_graphcast_test_03_long\
    --periods "train:1990-01-01,2015-12-31;val:2016-01-01,2017-12-31;test:2016-01-01,2019-12-31"\
    --eval test

python train_model.py\
    --name 1990_2020_6h_128_64_z64_era5_persistence_test_long\
    --dataset global_era5_1990_2020_6h-128x64_equiangular_with_poles_conservative\
    --dataset_path $(ws_find neural_lam)/data/data\
    --model persistence\
    --n_workers 40\
    --n_example_pred 0\
    --eval_leads 40\
    --hidden_dim 64\
    --processor_layers 4\
    --batch_size 1\
    --graph global_multilevel_era5\
    --periods "train:1990-01-01,2015-12-31;val:2016-01-01,2017-12-31;test:2016-01-01,2019-12-31"\
    --wandb_output trained_models/era5_1990_2020_6h-128x64_z64/test_runs/wandb_1990_2020_6h_128_64_z64_era5_graphcast_test_persistence_long\
    --eval test




# Try the same model, but on nextgems data

python train_model.py\
    --name 1990_2020_6h_128_64_z64_era5_graphcast_ON_NEXTGEMS_test_1_long\
    --dataset global_nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative\
    --dataset_path $(ws_find neural_lam)/data/data\
    --model graphcast\
    --n_workers 8\
    --n_example_pred 0\
    --eval_leads 40\
    --hidden_dim 64\
    --processor_layers 4\
    --batch_size 8\
    --graph global_multilevel_era5\
    --load trained_models/era5_1990_2020_6h-128x64_z64/checkpoints/1990_2020_6h_128_64_z64_era5_01/last.ckpt\
    --periods "train:1990-01-01,2015-12-31;val:2016-01-01,2017-12-31;test:2018-01-01,2019-12-31"\
    --wandb_output trained_models/era5_1990_2020_6h-128x64_z64/test_runs/wandb_1990_2020_6h_128_64_z64_era5_graphcast_ON_NEXTGEMS_test_01_long\
    --eval test

python train_model.py\
    --name 1990_2020_6h_128_64_z64_era5_graphcast_ON_NEXTGEMS_test_2_long\
    --dataset global_nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative\
    --dataset_path $(ws_find neural_lam)/data/data\
    --model graphcast\
    --n_workers 8\
    --n_example_pred 0\
    --eval_leads 40\
    --hidden_dim 64\
    --processor_layers 4\
    --batch_size 8\
    --graph global_multilevel_era5\
    --load trained_models/era5_1990_2020_6h-128x64_z64/checkpoints/1990_2020_6h_128_64_z64_era5_02/last.ckpt\
    --periods "train:1990-01-01,2015-12-31;val:2016-01-01,2017-12-31;test:2018-01-01,2019-12-31"\
    --wandb_output trained_models/era5_1990_2020_6h-128x64_z64/test_runs/wandb_1990_2020_6h_128_64_z64_era5_graphcast_ON_NEXTGEMS_test_02_long\
    --eval test


python train_model.py\
    --name 1990_2020_6h_128_64_z64_era5_graphcast_ON_NEXTGEMS_test_3_long\
    --dataset global_nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative\
    --dataset_path $(ws_find neural_lam)/data/data\
    --model graphcast\
    --n_workers 8\
    --n_example_pred 0\
    --eval_leads 40\
    --hidden_dim 64\
    --processor_layers 4\
    --batch_size 8\
    --graph global_multilevel_era5\
    --load trained_models/era5_1990_2020_6h-128x64_z64/checkpoints/1990_2020_6h_128_64_z64_era5_03/last.ckpt\
    --wandb_output trained_models/era5_1990_2020_6h-128x64_z64/test_runs/wandb_1990_2020_6h_128_64_z64_era5_graphcast_ON_NEXTGEMS_test_03_long\
    --periods "train:1990-01-01,2015-12-31;val:2016-01-01,2017-12-31;test:2018-01-01,2019-12-31"\
    --eval test

python train_model.py\
    --name 1990_2020_6h_128_64_z64_era5_graphcast_ON_NEXTGEMS_test_2049_long\
    --dataset global_nextgems_2046_2049_equiangular_with_poles_conservative\
    --dataset_path $(ws_find neural_lam)/data/data\
    --model graphcast\
    --n_workers 8\
    --n_example_pred 0\
    --eval_leads 40\
    --hidden_dim 64\
    --processor_layers 4\
    --batch_size 8\
    --graph global_multilevel_era5\
    --load trained_models/era5_1990_2020_6h-128x64_z64/checkpoints/1990_2020_6h_128_64_z64_era5_03/last.ckpt\
    --periods "train:2046-01-01,2049-12-31;val:2046-01-01,2049-12-31;test:2046-01-01,2049-12-31"\
    --wandb_output trained_models/era5_1990_2020_6h-128x64_z64/test_runs/wandb_1990_2020_6h_128_64_z64_era5_graphcast_ON_NEXTGEMS_test_03_2049_long\
    --eval test