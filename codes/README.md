# Codes

## Environments
`tf.__version__` = `2.2.0`

`tf.keras.__version__` = `2.3.0-tf`

`sklearn.__version__` = `0.23.1`

`cv2.__version__` = `4.2.0`

`skimage.__version__` = `0.17.2`

## Instructions
### Our improved BAGAN-GP model
- Run `improved_bagan_gp.py` independently if you just try it on `MNIST Fashion` or `CIFAR-10` Datasets.
- Otherwise, run `fetch_data.py` for getting our `Cells` dataset. 

  Then, it will generates 4 data files `x_train.npy`, `x_val.npy`, `y_train.npy`, `y_val.npy`.
  
  Then, run `improved_bagan_gp.py` to get cells results. Note: you need to cancel the comment of `Use our datasets`.
  
- If you intend to run this model on your own datasets, you need to save `your_images, your_labels` arrays into `.npy` files.
  
  Our codes will automatically turn your image data into channel last format.
  
- After running `improved_bagan_gp.py`, it will generate a directory `bagan_gp_results/` to store the result samples `generated_plot_%d.png`
  and combine these results into a `training_demo.gif` file.
- If you want to save your generator, you can type `bagan_gp.generator.save(...)` by yourself.

### Note
Run `fetch_data.py` before doing the following instructions, if you don't have your own `x_train.npy`, `x_val.npy`, `y_train.npy`, `y_val.npy`.
### FID scores
- We have an example generator named `bagan_gp_cells_v3_2_epoch100.h5` to run `fid_score.py`.
- After running `fid_score.py`, it will print the `FID(class=n)=...`
- FID score is not fixed at each running time, but it only changes slightly.
- Replace the `gen_path` to get FID for your generator.

### TSNE visualizations
- Run `tsne_plot.py` to see the distribution of labeled latent vectors. 
  We have two example encoders (`bagan_encoder.h5` for BAGAN, `bagan_gp_encoder1.h5` for BAGAN-GP) 
  and one example embedding model (`bagan_gp_embedding1.h5`) for running this code.

- Run `tsne_feature_resnet.py` to see the feature level distribution of the generated samples and real samples.
  We have an example generator `bagan_gp_cells_v3_2_epoch100.h5` for running this code.
