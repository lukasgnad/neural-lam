name=nextgems_1990_2020_6h-128x64_try2
name_full=${name}_z64

python train_model.py\
    --name ${name_full}_test_01\
    --dataset global_${name}\
    --dataset_path $(ws_find neural_lam)/data/data\
    --model graphcast\
    --n_workers 4\
    --n_example_pred 0\
    --eval_leads 40\
    --hidden_dim 64\
    --processor_layers 4\
    --batch_size 1\
    --graph global_multilevel_nextgems\
    --load trained_models/${name_full}/checkpoints/checkpoint_01/last.ckpt\
    --periods "train:1990-01-01,2015-12-31;val:2016-01-01,2017-12-31;test:2018-01-01,2019-12-31"\
    --wandb_output trained_models/${name_full}/test_runs/wandb_test_01\
    --eval test

python train_model.py\
    --name ${name_full}_test_02\
    --dataset global_${name}\
    --dataset_path $(ws_find neural_lam)/data/data\
    --model graphcast\
    --n_workers 16\
    --n_example_pred 0\
    --eval_leads 40\
    --hidden_dim 64\
    --processor_layers 4\
    --batch_size 1\
    --graph global_multilevel_nextgems\
    --load trained_models/${name_full}/checkpoints/checkpoint_02/last.ckpt\
    --periods "train:1990-01-01,2015-12-31;val:2016-01-01,2017-12-31;test:2018-01-01,2019-12-31"\
    --wandb_output trained_models/${name_full}/test_runs/wandb_test_02\
    --eval test

python train_model.py\
    --name ${name_full}_test_03\
    --dataset global_${name}\
    --dataset_path $(ws_find neural_lam)/data/data\
    --model graphcast\
    --n_workers 4\
    --n_example_pred 0\
    --eval_leads 40\
    --hidden_dim 64\
    --processor_layers 4\
    --batch_size 1\
    --graph global_multilevel_nextgems\
    --load trained_models/${name_full}/checkpoints/checkpoint_03/last.ckpt\
    --periods "train:1990-01-01,2015-12-31;val:2016-01-01,2017-12-31;test:2018-01-01,2019-12-31"\
    --wandb_output trained_models/${name_full}/test_runs/wandb_test_03\
    --eval test

python train_model.py\
    --name ${name_full}_test_persistence\
    --dataset global_${name}\
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
    --wandb_output trained_models/${name_full}/test_runs/wandb_test_persistence\
    --eval test

    




python train_model.py\
    --name ${name_full}_test_03_2049\
    --dataset global_nextgems_2049\
    --dataset_path $(ws_find neural_lam)/data/data\
    --model graphcast\
    --n_workers 4\
    --n_example_pred 0\
    --eval_leads 40\
    --hidden_dim 64\
    --processor_layers 4\
    --batch_size 1\
    --graph global_multilevel_nextgems\
    --load trained_models/${name_full}/checkpoints/checkpoint_03/last.ckpt\
    --periods "train:2049-01-01,2049-12-31;val:2049-01-01,2049-12-31;test:2049-01-01,2049-12-31"\
    --wandb_output trained_models/${name_full}/test_runs/wandb_test_03_2049\
    --save_forecasts 1\
    --save_levels 500,700,850\
    --eval test

python train_model.py\
    --name ${name_full}_test_03_2049_no_forecasts\
    --dataset global_nextgems_2049\
    --dataset_path $(ws_find neural_lam)/data/data\
    --model graphcast\
    --n_workers 4\
    --n_example_pred 0\
    --eval_leads 40\
    --hidden_dim 64\
    --processor_layers 4\
    --batch_size 1\
    --graph global_multilevel_nextgems\
    --load trained_models/${name_full}/checkpoints/checkpoint_03/last.ckpt\
    --periods "train:2049-01-01,2049-12-31;val:2049-01-01,2049-12-31;test:2049-01-01,2049-12-31"\
    --wandb_output trained_models/${name_full}/test_runs/wandb_test_03_2049_no_forecasts\
    --eval test

