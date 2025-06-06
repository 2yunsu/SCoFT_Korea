import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from utilities.clipconv_loss import CLIPConvFeatureLoss
from utilities.dreamsim_loss import DreamsimFeatureLoss
from utilities.dino_loss import DINOFetureLoss
from utilities.blip2_loss import BLIP2FeatureLoss
from utilities.lpips_loss import LPIPSFeatureLoss

from PIL import Image

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def save_model_card(repo_id: str, images=None, base_model=str, dataset_name=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- lora
inference: true
---
    """
    model_card = f"""
# LoRA text2image fine-tuning - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--blip_caption_column",
        type=str,
        default="blip_text",
        help="The column of the dataset containing a BLIP caption or a list of BLIP captions.",
    )
    parser.add_argument(
        "--negative_example_column",
        type=str,
        default="negative_imgpath",
        help="The column of the dataset containing the path of cultural-negative examples.",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--perceptualloss",
        action="store_true",
        help="whether to use lpips for contrastive learning",
    )
    parser.add_argument(
        "--onlypositive",
        action="store_true",
        help="whether to use only positive loss for contrastive training",
    )
    parser.add_argument(
        "--clipconvloss",
        action="store_true",
        help="whether to use clip conv loss for contrastive learning",
    )
    parser.add_argument(
        "--dinoloss",
        action="store_true",
        help="whether to dino loss for contrastive training",
    )
    parser.add_argument(
        "--dreamsimloss",
        action="store_true",
        help="whether to dreamsim loss for contrastive training",
    )
    parser.add_argument(
        "--blip2loss",
        action="store_true",
        help="whether to blip2 loss for contrastive training",
    )
    parser.add_argument(
        "--recordfirstgradient",
        action="store_true",
        help="record the first denoising gradient",
    )
    parser.add_argument(
        "--recordlastgradient",
        action="store_true",
        help="record the last denoising gradient",
    )
    parser.add_argument(
        "--recordrandomgradient",
        action="store_true",
        help="record a random denoising gradient",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Name to use for the WandB run (overrides default).",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")
    
    # Self-contrastive perceptual similarity backbone:
    loss_args = [args.blip2loss, args.clipconvloss, args.dreamsimloss, args.dinoloss]
    # Raise an error if more than one or none is set
    if sum(loss_args) > 1:
        raise ValueError("Only one of blip2loss, clipconvloss, dreamsimloss, or dinoloss can be set.")
    elif sum(loss_args) == 0:
        raise ValueError("One of blip2loss, clipconvloss, dreamsimloss, or dinoloss must be set.")
    
    recors_args = [args.recordfirstgradient, args.recordlastgradient, args.recordrandomgradient]
    # Raise an error if more than one or none is set
    if sum(recors_args) > 1:
        raise ValueError("Only one of recordfirstgradient, recordlastgradient, or recordrandomgradient can be set.")
    elif sum(recors_args) == 0:
        raise ValueError("One of recordfirstgradient, recordlastgradient, or recordrandomgradient must be set.")

    return args

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)

    text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # now we will add new LoRA weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Set correct lora layers
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=args.rank,
        )

    unet.set_attn_processor(lora_attn_procs)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
        
    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    lora_layers = AttnProcsLayers(unet.attn_processors)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        lora_layers.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir,"train","**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    # column_names = ['image', 'text', 'blip_text', 'negative_imgpath']
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = None

    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )
        blip_caption_column = args.blip_caption_column
        if blip_caption_column not in column_names:
            raise ValueError(
                f"--blip_caption_column' value '{args.blip_caption_column}' needs to be one of: {', '.join(column_names)}"
            )
        negative_example_column = args.negative_example_column
        if negative_example_column not in column_names:
            raise ValueError(
                f"--negative_example_column' value '{args.negative_example_column}' needs to be one of: {', '.join(column_names)}"
            )
    
    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    
    def tokenize_blip_captions(examples, is_train=True):
        captions = []
        for caption in examples[blip_caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{blip_caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    
    # this function get the null tokens for CFG
    def tokenize_null_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            captions.append("")
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    
    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        examples["input_blip_ids"] = tokenize_blip_captions(examples)
        examples["input_null_ids"] = tokenize_null_captions(examples)

        # Process negative images
        negative_images = []
        for neg_paths in examples[negative_example_column]:
            one_neg_images = []
            if len(neg_paths) > 0:
                for neg_path in neg_paths:
                    neg_full_path = os.path.join(args.train_data_dir, neg_path)
                    negative_img = Image.open(neg_full_path).convert("RGB")
                    negative_pixel_value = train_transforms(negative_img)
                    one_neg_images.append(negative_pixel_value)
            else:
                # use a fake flag to indicate there is no negative image
                negative_pixel_value_fakeflag = torch.zeros([3,512,512])
                one_neg_images.append(negative_pixel_value_fakeflag)
            
            one_neg_images = torch.stack(one_neg_images)
            negative_images.append(one_neg_images)
        examples["neg_pixel_values"] = negative_images

        return examples
    
    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)
    
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        input_blip_ids = torch.stack([example["input_blip_ids"] for example in examples])
        input_null_ids = torch.stack([example["input_null_ids"] for example in examples])
        neg_pixel_values = torch.stack([example["neg_pixel_values"] for example in examples])
        neg_pixel_values = neg_pixel_values.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values, "input_ids": input_ids, "input_blip_ids": input_blip_ids, "input_null_ids": input_null_ids, "neg_pixel_values":neg_pixel_values}
    
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_layers, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        # accelerator.init_trackers("text2image-contrastive-fine-tune", config=vars(args))
        accelerator.init_trackers("scoft-fine-tune", config=vars(args))
    

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    FeatureLoss = None
    if args.clipconvloss:
        FeatureLoss = CLIPConvFeatureLoss()
    elif args.dreamsimloss:
        FeatureLoss = DreamsimFeatureLoss()
    elif args.dinoloss:
        FeatureLoss = DINOFetureLoss()
    elif args.blip2loss:
        FeatureLoss = BLIP2FeatureLoss(device=accelerator.device)

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        contra_loss_log = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(1, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device) #0으로 생성되면 self-contrastive perceptual loss에서 0으로만 된 리스트가 생겨서 오류 발생. 따라서 0을 1로 수정.
                # print(timesteps)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                encoder_blip_hidden_states = text_encoder(batch["input_blip_ids"])[0]
                encoder_null_hidden_states = text_encoder(batch["input_null_ids"])[0]

                with torch.no_grad():
                   model_blip_pred = unet(noisy_latents[:encoder_blip_hidden_states.size(0)], timesteps[:encoder_blip_hidden_states.size(0)], encoder_blip_hidden_states).sample
                   model_null_pred = unet(noisy_latents[:encoder_null_hidden_states.size(0)], timesteps[:encoder_null_hidden_states.size(0)], encoder_null_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
                
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # LDM loss
                loss_ldm = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Memorization loss
                loss_mem = F.mse_loss(model_pred.float(), model_blip_pred.float(), reduction="mean")


                # Self-contrastive perceptual Loss
                loss_contrastive = torch.zeros(1).to(accelerator.device)
                loss_lpips = torch.zeros(1).to(accelerator.device)

                # Compute the self-contrastive perceptual loss in every 10 steps
                if (args.clipconvloss or args.dreamsimloss or args.dinoloss or args.blip2loss) and step%10 == 0: 
                    
                    # First, generate a denoised images as a generating sample
                    # generate images through more samplings
                    num_samplings = 20
                    custom_timesteps = []
                    if timesteps < (num_samplings-1) and timesteps!= 0:
                        custom_timesteps.append((timesteps.cpu())[0])
                        custom_timesteps.append(0)
                    else:
                        custom_steps = timesteps.cpu()//(num_samplings-1)
                        for i_sampling in range(num_samplings):
                            if i_sampling != (num_samplings-1):
                                custom_timesteps.append((timesteps.cpu() - i_sampling*custom_steps)[0])
                            else:
                                custom_timesteps.append(0)
                    # set timesteps
                    noise_scheduler.set_timesteps(timesteps=custom_timesteps, device=accelerator.device)

                    # generate images
                    CGF_Scale = 4.5
                    rand_grad_idx = torch.randint(0, len(noise_scheduler.timesteps), (bsz,), device=latents.device)        # only useful when args.recordrandomgradient
                    for i, t in enumerate(noise_scheduler.timesteps):
                        if args.recordfirstgradient:
                            if i != 0:      # 0 has been predicted
                                noisy_latents = noise_scheduler.scale_model_input(noisy_latents, t)
                                with torch.no_grad():
                                    model_pred = unet(noisy_latents, t, encoder_hidden_states).sample
                                    model_null_pred = unet(noisy_latents[:encoder_null_hidden_states.size(0)], t, encoder_null_hidden_states).sample
                            noise_CFG = model_null_pred + CGF_Scale * (model_pred - model_null_pred)
                            noisy_latents = noise_scheduler.step(noise_CFG, t, noisy_latents)[0]
                        elif args.recordlastgradient:
                            # do the first calculate again but don't record its gradient, and record the last gradient
                            if i != len(noise_scheduler.timesteps)-1:
                                with torch.no_grad():
                                    model_pred = unet(noisy_latents, t, encoder_hidden_states).sample
                                    model_null_pred = unet(noisy_latents[:encoder_null_hidden_states.size(0)], t, encoder_null_hidden_states).sample
                                noise_CFG = model_null_pred + CGF_Scale * (model_pred - model_null_pred)
                                noisy_latents = noise_scheduler.step(noise_CFG, t, noisy_latents)[0]
                                noisy_latents = noise_scheduler.scale_model_input(noisy_latents, t)
                            else:
                                model_pred = unet(noisy_latents, t, encoder_hidden_states).sample
                                with torch.no_grad():
                                    model_null_pred = unet(noisy_latents[:encoder_null_hidden_states.size(0)], t, encoder_null_hidden_states).sample
                                noise_CFG = model_null_pred + CGF_Scale * (model_pred - model_null_pred)
                                noisy_latents = noise_scheduler.step(noise_CFG, t, noisy_latents)[0]
                        elif args.recordrandomgradient:
                            if i != rand_grad_idx:
                                with torch.no_grad():
                                    model_pred = unet(noisy_latents, t, encoder_hidden_states).sample
                                    model_null_pred = unet(noisy_latents[:encoder_null_hidden_states.size(0)], t, encoder_null_hidden_states).sample
                                noise_CFG = model_null_pred + CGF_Scale * (model_pred - model_null_pred)
                                noisy_latents = noise_scheduler.step(noise_CFG, t, noisy_latents)[0]
                                noisy_latents = noise_scheduler.scale_model_input(noisy_latents, t)
                            else:
                                model_pred = unet(noisy_latents, t, encoder_hidden_states).sample
                                with torch.no_grad():
                                    model_null_pred = unet(noisy_latents[:encoder_null_hidden_states.size(0)], t, encoder_null_hidden_states).sample
                                noise_CFG = model_null_pred + CGF_Scale * (model_pred - model_null_pred)
                                noisy_latents = noise_scheduler.step(noise_CFG, t, noisy_latents)[0]
                    
                    latent_t0 = noisy_latents.to(accelerator.device, dtype=weight_dtype)
                    x0_estimate = vae.decode(latent_t0 / vae.config.scaling_factor, return_dict=False)[0]

                    # Second, get the positive and negative image lists
                    x0_positive = batch["pixel_values"].to(dtype=weight_dtype)
                    x0_negatives = batch["neg_pixel_values"].to(dtype=weight_dtype)
                    x0_negative_list = torch.unbind(x0_negatives[0])

                    # Third, get feature similarity between positive exampels
                    if args.perceptualloss:
                        LPIPSLoss = LPIPSFeatureLoss()
                        loss_lpips = LPIPSLoss(x0_estimate, x0_positive)
                    
                    positive_loss = FeatureLoss(x0_estimate, x0_positive)

                    # Forth, get the feature similarity between negative examples
                    negative_loss = torch.zeros(1).to(accelerator.device)
                    count_negative = 0
                    if not args.onlypositive:
                        for x0_negative in x0_negative_list:
                            with torch.no_grad():
                                x0_negative_image = x0_negative[None]
                            
                            # check if x0_negative_image is a torch.zeros, this means positive images don't have matching negative images
                            if not (torch.max(x0_negative_image)==0 and torch.min(x0_negative_image)==0):
                                negative_loss += FeatureLoss(x0_estimate, x0_negative_image)
                                count_negative += 1
                        
                        if count_negative > 0:
                            negative_loss = negative_loss / count_negative
                        else:
                            negative_loss = torch.zeros(1).to(accelerator.device)
                    
                    loss_contrastive = positive_loss - negative_loss
                
                # Compute the loss
                loss = 0.7*loss_ldm + 0.3*loss_mem + 0.05 * loss_contrastive + 0.01 * loss_lpips

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                avg_contras_loss = accelerator.gather(loss_contrastive.repeat(args.train_batch_size)).mean()
                contra_loss_log += avg_contras_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Log the losses
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                accelerator.log({"contras_loss": contra_loss_log}, step=global_step)
                contra_loss_log = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                with torch.no_grad():
                    logger.info(
                        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                        f" {args.validation_prompt}."
                    )
                    # create pipeline
                    pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=accelerator.unwrap_model(unet),
                        revision=args.revision,
                        torch_dtype=weight_dtype,
                    )
                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)

                    # run inference
                    generator = torch.Generator(device=accelerator.device)
                    if args.seed is not None:
                        generator = generator.manual_seed(args.seed)
                    images = []
                    for _ in range(args.num_validation_images):
                        images.append(
                            pipeline(args.validation_prompt, num_inference_steps=30, generator=generator).images[0]
                        )

                    for tracker in accelerator.trackers:
                        if tracker.name == "tensorboard":
                            np_images = np.stack([np.asarray(img) for img in images])
                            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                        if tracker.name == "wandb":
                            tracker.log(
                                {
                                    "validation": [
                                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                        for i, image in enumerate(images)
                                    ]
                                }
                            )

                    del pipeline
                    torch.cuda.empty_cache()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unet.save_attn_procs(args.output_dir)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                dataset_name=args.dataset_name,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    # Final inference
    # Load previous pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=weight_dtype
    )
    pipeline = pipeline.to(accelerator.device)

    # load attention processors
    pipeline.unet.load_attn_procs(args.output_dir)

    # run inference
    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
    images = []
    for _ in range(args.num_validation_images):
        images.append(pipeline(args.validation_prompt, num_inference_steps=30, generator=generator).images[0])

    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            if len(images) != 0:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            "test": [
                                wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                for i, image in enumerate(images)
                            ]
                        }
                    )

    accelerator.end_training()


if __name__ == "__main__":
    main()




