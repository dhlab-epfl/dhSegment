# cBAD experiment

## Dataset
Dataset from [ICDAR 2017 Competition on Baseline Detection in Archival Documents (cBAD)](https://zenodo.org/record/835441) ([paper](https://arxiv.org/abs/1705.03311))

## Demo

1. Dowload pretrained weights. 
   ``` shell
    $ cd pretrained_models/
    $ python download_resnet_pretrained_model.py
    $ cd ..
   ```

2. Run the script `make_cbad.py` that will download the dataset and create the masks. This may take some time (10-20 min).

   ``` shell
   $ cd exps/cbad
   $ python make_cbad.py --downloading_dir ../../data/cbad-dataset --masks_dir ../../data/cbad-masks
   $ cd ../..
   ```

3. To train a model run from the root directory `python train.py with demo/demo_cbad_config.json`. 
If you changed the default directory of `--masks_dir` make sure to update the file `demo_cbad_config.json`.

4. To use the trained model on new data, run the `demo_processing.py` script :
   ``` shell
   $ cd exps/cbad
   $ python demo_processing.py ../../data/cbad-masks/simple/test/images/*.jpg
                             --model_dir ../../demo/cbad_simple_model/export/<timestamp-model> 
                             --output_dir ../../demo/baseline_extraction_output
                             --draw_extractions 1
   $ cd ../..
   ```
 5. Have a look at the result in the folder `demo/baseline_extraction_output`.