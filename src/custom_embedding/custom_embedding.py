import torch
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config


class CustomEmbedding(ModelMixin, ConfigMixin):
    r"""
    Just a nn.Embedding wrapped in Hugging Face's API.
    """

    @register_to_config
    def __init__(self, num_classes: int, class_embedding_dim: int):
        super().__init__()
        self.inner_module = torch.nn.Embedding(num_classes, class_embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner_module(x)
