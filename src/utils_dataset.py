# Copyright 2023 The HuggingFace Team and Thomas Boyer. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from argparse import Namespace
from pathlib import Path

import torch
from accelerate.logging import MultiProcessAdapter
from datasets import load_dataset
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def setup_dataset(
    args: Namespace, logger: MultiProcessAdapter
) -> tuple[ImageFolder | Subset, int]:
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        raise NotImplementedError("Not tested yet")
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
    elif args.use_pytorch_loader:
        dataset: ImageFolder | Subset = ImageFolder(
            root=Path(args.train_data_dir, args.split).as_posix(),
            transform=lambda x: augmentations(x.convert("RGB")),
            target_transform=lambda y: torch.tensor(y).long(),
        )
    else:
        raise NotImplementedError("Not tested yet")
        dataset = load_dataset(
            "imagefolder",
            data_dir=args.train_data_dir,
            cache_dir=args.cache_dir,
            split="train",
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    if args.perc_samples is not None:
        logger.warning(
            f"Subsampling the dataset to {args.perc_samples*100}% of samples per class"
        )
        dataset = select_subset_of_dataset(dataset, args.perc_samples)

    # Preprocessing the datasets and DataLoaders creation.
    augmentations = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # map to [-1, 1] for SiLU
        ]
    )

    def transform_images(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        class_labels = examples["label"]
        return {"images": images, "class_labels": class_labels}

    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Number of classes: {len(dataset.classes)}")

    if not args.use_pytorch_loader:
        raise NotImplementedError("Not tested yet")
        dataset.set_transform(transform_images)

    return dataset, len(dataset.classes)


def select_subset_of_dataset(dataset: ImageFolder, perc_samples: float) -> Subset:
    """Subsamples the given dataset to have <perc_samples>% of each class."""

    # 1. First test if the dataset is balanced; for now we assume it is
    class_counts = dict.fromkeys(
        [dataset.class_to_idx[cl] for cl in dataset.classes], 0
    )
    for _, label in dataset.samples:
        class_counts[label] += 1

    nb_classes = len(class_counts)

    assert (
        list(class_counts.values()) == [class_counts[0]] * nb_classes
    ), "The dataset is not balanced between classes"

    # 2. Then manually sample <perc_samples>% of each class
    orig_nb_samples_per_balanced_classes = class_counts[0]

    nb_selected_samples_per_class = int(
        orig_nb_samples_per_balanced_classes * perc_samples
    )

    sample_indices = []

    nb_selected_samples = dict.fromkeys(
        [dataset.class_to_idx[cl] for cl in dataset.classes], 0
    )

    # random.sample(x, len(x)) shuffles x out-of-place
    iterable = enumerate(random.sample(dataset.samples, len(dataset)))

    while (
        list(nb_selected_samples.values())
        != [nb_selected_samples_per_class] * nb_classes
    ):
        idx, (_, class_label) = next(iterable)
        if nb_selected_samples[class_label] < nb_selected_samples_per_class:
            sample_indices.append(idx)
            nb_selected_samples[class_label] += 1

    assert (
        len(sample_indices) == nb_selected_samples_per_class * nb_classes
    ), "Something went wrong in the subsampling..."

    # 3. Return the subset
    subset = Subset(dataset, sample_indices)
    # hacky but ok to do this because each class is present in the subset
    subset.classes = dataset.classes

    return subset
