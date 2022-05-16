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

# Change Detection
## Training
### Download the datasets
Download the change detection datasets from the following links. Place them inside your `datasets` folder.

- [`LEVIR-CD`](https://www.dropbox.com/s/18fb5jo0npu5evm/LEVIR-CD256.zip?dl=0)
- [`WHU-CD`](https://www.dropbox.com/s/r76a00jcxp5d3hl/WHU-CD-256.zip?dl=0)
- [`DSIFN-CD`](https://www.dropbox.com/s/1lr4m70x8jdkdr0/DSIFN-CD-256.zip?dl=0)
- [`CDD`](https://www.dropbox.com/s/ls9fq5u61k8wxwk/CDD.zip?dl=0)


Then, update the paths to those folders here [`datasets`][`train`][`dataroot`], [`datasets`][`val`][`dataroot`], [`datasets`][`test`][`dataroot`] in `levir.json`, `whu.json`, `dsifn.json`, and `cdd.json`.

### Provide the path to pre-trained diffusion model
Udate the path to pre-trained diffusion model weights (`*_gen.pth` and `*_opt.pth`) here [`path`][`resume_state`] in `levir.json`, `whu.json`, `dsifn.json`, and `cdd.json`..

### Training the change detection network
Run the following code to start the training.
- Training on LEVIR-CD:
    ```python
    python ddpm_cd.py --config config/levir.json -enable_wandb -log_eval
    ```
- Training on WHU-CD:
    ```python
    python ddpm_cd.py --config config/whu.json -enable_wandb -log_eval
    ```
- Training on DSIFN-CD:
    ```python
    python ddpm_cd.py --config config/dsifn.json -enable_wandb -log_eval
    ```
- Training on CDD:
    ```python
    python ddpm_cd.py --config config/cdd.json -enable_wandb -log_eval
    ```

The results will be saved in `experiments` and also upload to `wandb`.

## Testing
To obtain the predictions and performance metrics (iou, f1, and OA), first provide the path to pre-trained diffusion model here [`path`][`resume_state`] and path to trained change detection model (the best model) here [`path_cd`][`resume_state`] in `levir_test.json`, `whu_test.json`, `dsifn_test.json`, and `cdd_test.json`.

Run the following code to start the training.
- Test on LEVIR-CD:
    ```python
    python ddpm_cd.py --config config/levir_test.json --phase test -enable_wandb -log_eval
    ```
- Test on WHU-CD:
    ```python
    python ddpm_cd.py --config config/whu_test.json --phase test -enable_wandb -log_eval
    ```
- Test on DSIFN-CD:
    ```python
    python ddpm_cd.py --config config/dsifn_test.json --phase test -enable_wandb -log_eval
    ```
- Test on CDD:
    ```python
    python ddpm_cd.py --config config/cdd_test.json --phase test -enable_wandb -log_eval
    ```

Predictions will be saved in `experiments` and performance metrics will be uploaded to wandb.

## Donwload pre-trained models
- Pre-trained diffusion model: [`Click Here`]()
- Pre-trained change detection networks:
    - "t": [50]
        - LEVIR-CD [`cd-levir-50`](https://www.dropbox.com/sh/y43gh94t1ih18kl/AAB4T2nYGC0vIChYRSg9CVtHa?dl=0)
        - WHU-CD [`cd-whu-50`](https://www.dropbox.com/sh/um1rh7fgcvss4v2/AABvsOSaipDWYA5oeHd6IQ97a?dl=0)
        - DSIFN-CD [`cd-dsifn-50`](https://www.dropbox.com/sh/2z5zlqz9m8bkmus/AACxuhLYC0VsNS5UBKO1JtHua?dl=0)
        - CDD-CD [`cd-cdd-50`](https://www.dropbox.com/sh/2nf77dz30m5i1zg/AABn3o7imwqv1ObSr1v5MHN1a?dl=0)
    - "t": [100]
        - LEVIR-CD [`cd-levir-100`](https://www.dropbox.com/sh/n7ktxbrbxdrtoxy/AACEMhg5cvOlBkN3FSF_MmGca?dl=0)
        - WHU-CD [`cd-whu-100`](https://www.dropbox.com/sh/ux3r8a5xo0g22rf/AACFn5pkGuN3N-lajPUy93_3a?dl=0)
        - DSIFN-CD [`cd-dsifn-100`](https://www.dropbox.com/sh/f7ike5ufdtfpd5h/AACWoOiE78t7n47HnDUGTJsZa?dl=0)
        - CDD-CD [`cd-cdd-100`](https://www.dropbox.com/sh/sdzjch0r2qscrle/AABlIBJlXfwavo-97dPTTKSna?dl=0)
    - "t": [50, 100]
        - LEVIR-CD [`cd-levir-50-100`](https://www.dropbox.com/sh/ie9rapb1j2zgvb7/AAALkpLS-tvngTb4HXqAcbbTa?dl=0)
        - WHU-CD [`cd-whu-50-100`](https://www.dropbox.com/sh/9idrobnmhufo1e7/AABRf38iq-wE7plKZZmwFywva?dl=0)
        - DSIFN-CD [`cd-dsifn-50-100`](https://www.dropbox.com/sh/001czxn335bul5g/AACRaR-nqQNNHEge6iSH_z-6a?dl=0)
        - CDD-CD [`cd-cdd-50-100`](https://www.dropbox.com/sh/62wsy9cl8xizx2h/AAB5Dmu-PuOVAfIBugGqlsd8a?dl=0)
    - "t": [50, 100, 400] (*Best Model*)
        - LEVIR-CD [`cd-levir-50-100-400`](https://www.dropbox.com/sh/sx0aopz230lbuwc/AADKpwP30OHvtYub9FYTyk53a?dl=0)
        - WHU-CD [`cd-whu-50-100-400`](https://www.dropbox.com/sh/l8iuzb2tudb3yrk/AAA7aZwb5eM12SamCXPh7R-Ra?dl=0)
        - DSIFN-CD [`cd-dsifn-50-100-400`](https://www.dropbox.com/sh/ekj7kwsohhnjico/AADuz0vBtxCCrYgdgOCG3LX5a?dl=0)
        - CDD-CD [`cd-cdd-50-100-400`](https://www.dropbox.com/sh/a8dj1i8pnexd5yu/AADnmBGT4VdGY8aZMo7enfS7a?dl=0)







