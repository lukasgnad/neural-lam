python train_model.py\
    --name nextgems_1990_2020_6h_128_64_z64_test_01\
    --dataset global_nextGEMS_1990_2020_6h-128x64_transposed_nans_filled_f32\
    --dataset_path $(ws_find neural_lam)/data/data\
    --model graphcast\
    --n_workers 4\
    --n_example_pred 0\
    --eval_leads 40\
    --hidden_dim 64\
    --processor_layers 4\
    --batch_size 1\
    --graph global_multilevel_nextgems\
    --load trained_models/nextgems_1990_2020_6h-128x64_z64/checkpoints/nextgems_1990_2020_6h_128_64_z64_f32_01/last.ckpt\
    --periods "train:1990-01-01,2015-12-31;val:2016-01-01,2017-12-31;test:2018-01-01,2019-12-31"\
    --eval test

python train_model.py\
    --name nextgems_1990_2020_6h_128_64_z64_test_02\
    --dataset global_nextGEMS_1990_2020_6h-128x64_transposed_nans_filled_f32\
    --dataset_path $(ws_find neural_lam)/data/data\
    --model graphcast\
    --n_workers 4\
    --n_example_pred 0\
    --eval_leads 40\
    --hidden_dim 64\
    --processor_layers 4\
    --batch_size 1\
    --graph global_multilevel_nextgems\
    --load trained_models/nextgems_1990_2020_6h-128x64_z64/checkpoints/nextgems_1990_2020_6h_128_64_z64_f32_02/last.ckpt\
    --periods "train:1990-01-01,2015-12-31;val:2016-01-01,2017-12-31;test:2018-01-01,2019-12-31"\
    --eval test

python train_model.py\
    --name nextgems_1990_2020_6h_128_64_z64_test_03\
    --dataset global_nextGEMS_1990_2020_6h-128x64_transposed_nans_filled_f32\
    --dataset_path $(ws_find neural_lam)/data/data\
    --model graphcast\
    --n_workers 4\
    --n_example_pred 0\
    --eval_leads 40\
    --hidden_dim 64\
    --processor_layers 4\
    --batch_size 1\
    --graph global_multilevel_nextgems\
    --load trained_models/nextgems_1990_2020_6h-128x64_z64/checkpoints/nextgems_1990_2020_6h_128_64_z64_f32_03/last.ckpt\
    --periods "train:1990-01-01,2015-12-31;val:2016-01-01,2017-12-31;test:2018-01-01,2019-12-31"\
    --eval test

python train_model.py\
    --name persistence_nextgems_1990_2020_6h_128_64_z64_test\
    --dataset global_nextGEMS_1990_2020_6h-128x64_transposed_nans_filled_f32\
    --dataset_path $(ws_find neural_lam)/data/data\
    --model persistence\
    --n_workers 50\
    --n_example_pred 0\
    --eval_leads 40\
    --hidden_dim 64\
    --processor_layers 4\
    --batch_size 1\
    --graph global_multilevel_nextgems\
    --periods "train:1990-01-01,2015-12-31;val:2016-01-01,2017-12-31;test:2018-01-01,2019-12-31"\
    --eval test

    