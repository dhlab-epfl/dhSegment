# Page experiment
Based on paper ["PageNet: Page Boundary Extraction in Historical Handwritten Documents."](https://dl.acm.org/citation.cfm?id=3151522)


## Dataset 
The page annotations come from this [repository](https://github.com/ctensmeyer/pagenet/tree/master/annotations). We use READ-cBAD data with _annotator 1_ and _set1_.

## Demo

1. Dowload pretrained weights. 
   ``` shell
    $ cd pretrained_models/
    $ python download_resnet_pretrained_model.py
    $ cd ..
   ```

2. Run the script `make_page.py` that will download the dataset and create the masks. This may take some time (10-20 min).

   ``` shell
   $ cd exps/page
   $ python make_page.py --downloading_dir ../../data/cbad-dataset --masks_dir ../../data/page-masks
   $ cd ../..
   ```

 
3. To train a model run from the root directory `python train.py with demo/demo_page_config.json`. 
    If you changed the default directory of `--masks_dir` make sure to update the file `demo_config.json`.
    
4. To use the trained model on new data, run the `demo_processing.py` script :
   ``` shell
   $ cd exps/page
   $ python demo_processing.py ../../data/page-masks/test/images/*.jpg
                             --model_dir ../../demo/page_model/export/<timestamp-model> 
                             --output_dir ../../demo/page_extraction_output
                             --draw_extractions 1
   $ cd ../..
   ```

5. Have a look at the result in the folder `demo/page_extraction_output`.