# Remote Sensing Change Detection using Denoising Diffusion Probabilistic Models

[Paper]() |  [Project]()

This is the offical implementation of **Remote Sensing Change Detection using Denoising Diffusion Probabilistic Models** by **Pytorch**.

## Usage
### Environment
```python
pip3 install -r requirement.txt
```

### Collect off-the-shelf remote sensing data to train diffusion model

Dump all the remote sensing data sampled from Google Earth Engine and any other publically available remote sensing images to dataset folder or create a simlink. 

### Training/Resume unconditional diffusion model on remote sensing data

We use ``ddpm_train.json`` to setup the configurations. Update the dataset ``name`` and ``dataroot`` in the json file. The run the following command to start training the diffusion model. The results and log files will be save to ``experiments`` folder. Also, we upload all the metrics to [wandb](https://wandb.ai/home).

```python
python ddpm_train.py --config config/ddpm_train.json -enable_wandb -log_eval
```


### Training/Resume Training
In case, if you want to resume the training from previosely saved point, provide the path to saved model in ``path/resume_state``, else keep it as null.

# Training change detection network
## Download change detection dataset
Download the change detection datasets from the following links.
Links to download datasets:
[LEVIR-CD]()

[WHU-CD]()

[DSIFN-CD]()

[CDD]

Next, update the paths to those folders here [`datasets`][`train`][`dataroot`], [`datasets`][`val`][`dataroot`], [`datasets`][`test`][`dataroot`] in `levir.json`, `whu.json`, `dsifn.json`, and `cdd.json`.

Also udate the path to pre-trained diffusion model weights here [`path`][`resume_state`].






