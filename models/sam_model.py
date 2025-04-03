import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Any


class Sam(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        prompt_encoder: nn.Module,
        mask_decoder: nn.Module,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Args:
            image_encoder (nn.Module): Backbone to encode the image into embeddings.
            prompt_encoder (nn.Module): Encodes input prompts.
            mask_decoder (nn.Module): Decodes masks from embeddings and prompts.
            pixel_mean (list): Mean for input normalization.
            pixel_std (list): Std for input normalization.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), persistent=False)

    def forward(self, batched_input, multimask_output: bool, image_size: int) -> dict:
        return self.forward_train(batched_input, multimask_output, image_size)

    def forward_train(self, batched_input, multimask_output: bool, image_size: int) -> dict:
        hw_size, d_size = batched_input.shape[-2], batched_input.shape[1]
        batched_input = batched_input.contiguous().view(-1, 3, hw_size, hw_size)

        input_images = self.preprocess(batched_input)
        image_embeddings = self.image_encoder(input_images, d_size)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, boxes=None, masks=None
        )

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output
        )

        masks = self.postprocess_masks(
            low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )

        return {
            'masks': masks,
            'iou_predictions': iou_predictions,
            'low_res_logits': low_res_masks
        }

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, int],
        original_size: Tuple[int, int],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks
