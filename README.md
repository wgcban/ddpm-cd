# Remote Sensing Change Detection using Denoising Diffusion Probabilistic Models

[Paper]() |  [Project]()

This is the offical implementation of **Remote Sensing Change Detection using Denoising Diffusion Probabilistic Models** by **Pytorch**.

## Usage
### Environment
```python
pip3 install -r requirement.txt
```

# Training diffusion model with remote sensing data
### Collect off-the-shelf remote sensing data to train diffusion model

Dump all the remote sensing data sampled from Google Earth Engine and any other publically available remote sensing images to dataset folder or create a simlink. 

### Training/Resume unconditional diffusion model on remote sensing data

We use ``ddpm_train.json`` to setup the configurations. Update the dataset ``name`` and ``dataroot`` in the json file. The run the following command to start training the diffusion model. The results and log files will be save to ``experiments`` folder. Also, we upload all the metrics to [wandb](https://wandb.ai/home).

```python
python ddpm_train.py --config config/ddpm_train.json -enable_wandb -log_eval
```

In case, if you want to resume the training from previosely saved point, provide the path to saved model in ``path/resume_state``, else keep it as null.

# Training change detection network
## Download change detection dataset
Download the change detection datasets from the following links.
Links to download datasets:

-[`LEVIR-CD`](https://www.dropbox.com/s/18fb5jo0npu5evm/LEVIR-CD256.zip?dl=0)
-[`WHU-CD`](https://www.dropbox.com/s/r76a00jcxp5d3hl/WHU-CD-256.zip?dl=0)
-[`DSIFN-CD`](https://www.dropbox.com/s/1lr4m70x8jdkdr0/DSIFN-CD-256.zip?dl=0)
-[`CDD`](https://www.dropbox.com/s/ls9fq5u61k8wxwk/CDD.zip?dl=0)

Next, update the paths to those folders here [`datasets`][`train`][`dataroot`], [`datasets`][`val`][`dataroot`], [`datasets`][`test`][`dataroot`] in `levir.json`, `whu.json`, `dsifn.json`, and `cdd.json`.

Also udate the path to pre-trained diffusion model weights here [`path`][`resume_state`].






