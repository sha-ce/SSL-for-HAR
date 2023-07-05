## Requirements
If you would like to develop the model for your own use, you need to follow the instructions below:
### Installation
```bash
conda create -n ssl_env python=3.7
conda activate ssl_env
pip install -r req.txt
```

### Directory structure
#### data for pre-training
```shell
- data:
  |_ downstream
    |_oppo
      |_ X.npy
      |_ Y.npy
      |_ pid.npy
    |_pamap2
    ...

  |_ ssl # ignore the ssl folder if you don't wish to pre-train using your own dataset
    |_ ssl_capture_24
      |_data
        |_ train
          |_ *.npy
          |_ file_list.csv # containing the paths to all the files
        |_ test
          |_ *.npy
      |_ logs
        |_models
```

#### data for downstream 
placing data in the following structure
```shell
- data:
  |_ downstream
    |_ oppo_dataset
        |_ oppo_30hz_w10_o5_x_train.npy
        |_ oppo_30hz_w10_o5_y_train.npy
        |_ ...
```

### pre-trained model params
I think put `model.mdl` in moodle or. you download the file and place it `experiment_log/pre-train/task4_<time>/best.mdl`.


## Training
### pre-train learning
First you will want to download the processed capture24 dataset on your local machine. Self-supervised training on capture-24 for all of the three tasks can be run using:
```bash
python pre-train.py
```

## downstream
```bash
python3 downstream.py
```
### Results
after the above code is executed, the result is output to `experiment_log/downstream/`.
