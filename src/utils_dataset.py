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
from typing import Optional

import torch
from accelerate.logging import MultiProcessAdapter
from datasets import load_dataset
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class NoLabelsDataset(ImageFolder):
    """A custom dataset that only returns the images, without their labels."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            sample
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def setup_dataset(
    args: Namespace, logger: MultiProcessAdapter
) -> tuple[ImageFolder | Subset, NoLabelsDataset | Subset, int]:
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
            transform=lambda x: transformations(x.convert("RGB")),
            target_transform=lambda y: torch.tensor(y).long(),
        )
        raw_dataset: NoLabelsDataset | Subset = NoLabelsDataset(
            root=Path(args.train_data_dir, args.split).as_posix(),
            transform=lambda x: raw_transformations(x.convert("RGB")),
        )
        assert len(dataset) == len(
            raw_dataset
        ), "dataset and raw_dataset should have the same length"
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
            f"Subsampling the training dataset to {args.perc_samples}% of samples per class"
        )
        if not args.compute_metrics_full_dataset:
            logger.warning(
                f"Metrics computation will be done against the subsampled dataset"
            )
            dataset, raw_dataset = _select_subset_of_dataset(args, dataset, raw_dataset)
        else:
            dataset = _select_subset_of_dataset(args, dataset)  # type: ignore

    # Preprocessing the datasets and DataLoaders creation
    # transforms for the training dataset
    list_transforms = [
        transforms.Resize(
            args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # map to [-1, 1] for SiLU
    ]
    if args.data_aug_on_the_fly:
        list_transforms += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    transformations = transforms.Compose(list_transforms)
    # transforms for the raw dataset
    raw_transformations = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.PILToTensor(),
        ]
    )

    def transform_images(examples):
        images = [transformations(image.convert("RGB")) for image in examples["image"]]
        class_labels = examples["label"]
        return {"images": images, "class_labels": class_labels}

    if not args.use_pytorch_loader:
        raise NotImplementedError("Not tested yet")
        dataset.set_transform(transform_images)

    return dataset, raw_dataset, len(dataset.classes)


def _select_subset_of_dataset(
    args: Namespace,
    full_dataset: ImageFolder,
    full_raw_dataset: Optional[NoLabelsDataset] = None,
) -> tuple[Subset, Subset] | Subset:
    """Subsamples the given dataset(s) to have <perc_samples>% of each class."""

    # 1. First test if the dataset is balanced; for now we assume it is
    class_counts = dict.fromkeys(
        [full_dataset.class_to_idx[cl] for cl in full_dataset.classes], 0
    )
    for _, label in full_dataset.samples:
        class_counts[label] += 1

    nb_classes = len(class_counts)

    assert (
        list(class_counts.values()) == [class_counts[0]] * nb_classes
    ), "The dataset is not balanced between classes"

    # 2. Then manually sample <perc_samples>% of each class
    orig_nb_samples_per_balanced_classes = class_counts[0]

    nb_selected_samples_per_class = int(
        orig_nb_samples_per_balanced_classes * args.perc_samples / 100
    )

    sample_indices = []

    nb_selected_samples = dict.fromkeys(
        [full_dataset.class_to_idx[cl] for cl in full_dataset.classes], 0
    )

    # set seed
    # `random` is only used here, for the dataset subsampling
    random.seed(args.seed)

    # random.sample(x, len(x)) shuffles x out-of-place
    iterable = random.sample(list(enumerate(full_dataset.samples)), len(full_dataset))

    for idx, (_, class_label) in iterable:
        # stop condition
        if (
            list(nb_selected_samples.values())
            == [nb_selected_samples_per_class] * nb_classes
        ):
            break
        # select sample
        if nb_selected_samples[class_label] < nb_selected_samples_per_class:
            sample_indices.append(idx)
            nb_selected_samples[class_label] += 1

    assert (
        len(sample_indices) == nb_selected_samples_per_class * nb_classes
    ), "Something went wrong in the subsampling..."

    # 3. Return the subset(s)
    subset = Subset(full_dataset, sample_indices)
    if full_raw_dataset is not None:
        raw_subset = Subset(full_raw_dataset, sample_indices)
        assert subset.indices == raw_subset.indices
    else:
        raw_subset = None

    # hacky but ok to do this because each class is present in the subset
    subset.classes = full_dataset.classes
    if full_raw_dataset is not None:
        raw_subset.classes = full_raw_dataset.classes

    subset.targets = [full_dataset.targets[i] for i in subset.indices]
    if full_raw_dataset is not None:
        raw_subset.targets = [full_raw_dataset.targets[i] for i in raw_subset.indices]

    if full_raw_dataset is not None:
        return subset, raw_subset
    else:
        return subset
