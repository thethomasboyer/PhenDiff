**Diffusion models based image-to-image translation**

---

# Introduction
This repository contains the code needed both
1. to train (or fine-tune) diffusion models 
2. to perform image-to-image class translation with pretrained models


> [!WARNING]
> This repo is still a WIP; expect breaking changes and broken things!

# ⬇️ Install

## ⚙️ Dependencies

To install the required dependencies, run:
```sh
{conda, mamba} install -f environment.yaml
```

## 📊 Experiment tracker
The only supported experiment tracker for now is [`wandb`](https://wandb.ai/site). You will need a configured Weights & Biases environement to log information for both kind of experiments.

# 📉 Train models
Training or fine-tuning diffusion models is in principle performed by running the following command:
``` sh
accelerate launch {accelerate args} train.py {script args}
```
where:
- `accelerate` must be configured to your training setup, either with `accelerate config` beforehand or by passing the appropriate flags to `accelerate launch` in place of `{accelerate args}` (ee the [accelerate documentation](https://huggingface.co/docs/accelerate) for more details)
- some args are **required** by the training script _in lieu_ of `{script args}` (see the `src/args_parser.py` file for the full list of possible and required training script arguments –you can also call `python train.py --help` in the terminal but it takes quite some time to print)

## 🐥 Example training commands

### Local examples
Some examples of commands launching a training experience can be found in the `examples/examples_training_scripts` folder.  
They consist in `bash/zsh` scripts handling the configuration of both `accelerate` and the training script `train.py`. They can be called directly from the command line.

Two examples:
- The following script:
```sh
./examples/examples_training_scripts/launch_script_DDIM.sh
```
will train a DDIM model from scratch on the data located at `path/to/train/data`.

- This one:
```sh
./examples/examples_training_scripts/launch_script_SD.sh
```
will fine-tune the UNet of `stabilityai/stable-diffusion-2-1` (plus a custom class embedding) on the data located at `path/to/train/data`.

> _Configure these examples launchers to your neeeds!_

> [!NOTE]
> Future version will probably use [Hydra](https://hydra.cc/) to handle the training configuration.

### SLURM examples
The `SLURM_launch_script_<xxx>.sh` files demonstrate how to adapt these bash scripts to a SLURM cluster. 

They are meant to launch a series of runs at different sizes of training data on the `A100` partition of the [Jean Zay CNRS cluster](http://www.idris.fr/eng/jean-zay/index.html). They load a custom `python` environement located at `${SCRATCH}/micromamba/envs/diffusion-experiments` with `micromamba`; adapt to your needs!

# 🎨 Image-to-image class transfer
Image-to-image class transfer experiments are performed with the `img2img_comparison_launcher.py` script, which additionally handles the configuration of `accelerate` and possibly submits the jobs to the `SLURM` manager. It can be called as:
```sh
python img2img_comparison_launcher.py {hydra overrides} &
```
## Configuration

The image-to-image class transfer experiments are configured with [Hydra](https://hydra.cc/). Example configuration files can be found in the `examples/example_img2img_comparison_conf` folder. 

The `img2img_comparison_launcher.py` script expects a configuration folder named `my_img2img_comparison_conf` to be located in the directory where it is called, and a filed named `general_config.yaml` inside this configuration folder.  
These defaults can be overriden with the ` --config-path` and `--config-name` Hydra arguments.

> [!NOTE]
> To prevent the experiment config being modified between the job submission and the job launch (which can typically take quite some time when submitting to SLURM), the entire config is copied to the experiment folder and the submitted job will pull its config from there.

## Hyperparameters sweep

TODO

# Outputs
All experiments (either training or class transfer) output artifacts following the project/run organization of wandb:
```
- exp_parent_folder
|   - project
|   |   - run_name
```
where `exp_parent_folder` is any base path on the system, and `project` and `run_name` are both the names of the folder created by the scripts on the system and the project and run names on wandb (1-on-1 correspondence).  
When Hydra is used, an additional timestamped sub-folder hierarchy is also created under the `run_name` folder:
```
- run_name
|   - day
|   |   - hour
```
This is especially important when doing sweeps, so that runs with the same name but with different hyperparameters do not overwrite each other.
