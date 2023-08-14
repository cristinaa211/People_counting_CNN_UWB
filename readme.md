Create Pyhton3 environment:
python3 -m venv /path/to/new/virtual/environment

Activate virtual environment:
source /path/to/new/virtual/environment/bin/activate

Install all dependencies:
pip install requirements.txt


**WORKFLOW:**
TECH STACK: Python3, PyTorch, Spark, PostgreSQL, GIT, Docker, Grafana why not 

**
1. Read all file samples names, store in list
2. Generate JSON files with metainformation about each
 radar sample : label, context, number of persons in the radar range, the radar sample's values,
 the shape of initial radar sample and its type
3. Store the JSON files in a PostgreSQL database - UWB_Radar_Samples.
4. Visualize radar samples from each scenario
5. Make a shallow analysis 
6. Clean data: filtering, clutter removal, extract the dc component 
7. Extract features: PCA 
8. Provide as input for CNN, CRNN or Transformers 
9. Analyse metrics 
**
10. Save model 
11. Create final pipeline
12. Deploy in Docker Container



# ALL THESE ARE IN TRAINER 

# Vanishing Gradients : when the gradient turns to zero
# Exploding Gradients : when the gradient turns to infinit caused by the propagation, where you multiply things that are above zero
# track_grad_norm flag
# reload_dataloaders_every_epoch = True when you have a model in the production or data is changing
# weights_summary = 'full' or = 'top' displays the parameters for each layer 
# progress_bar_refresh_rate 
# profiler = True gives a high level descriptions of methods called and how long it took 
# min_epochs, max_epochs flags = int
# min_steps, max_steps flags = int
# check_val_every_n_epochs = int
# val_check_interval = 0.25 
# num_sanity_val_steps = 2 batches of validation 
# limit_train_batches, limit_val_batches, limit_test_batches
# ON A SINGLE MACHINE 
# GPU: put each tensor by default on the device (cuda)
# gpus = 4, auto_select_gpus = True, log_gpu_memory = "all" , "min_max", benchmark=True tp speed the training if data does not change
# deterministic = True to garantee the reproducible results to reduce the randomness in the training 
# TO RUN ON MULTIPLE MACHINES
# distributed_backend = 'ddp_spwan', gpus = 8, num_nodes = 8
# Debugging flags 
# fast_dev_run = True 
# overfit_batches = 1 pick  a single batch and overfit the batch, if it does not overfit, you have a bug
# accumulate_grad_batches = 4 : the number of forward steps when you have large data 
# mixed precision,  to reduce the memory and speedup the GPUs
# precision = 16
# 