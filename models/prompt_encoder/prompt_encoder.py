import torch
import torch.nn as nn

from .position_embedding import PositionEmbeddingRandom
from ..image_encoder.layernorm2d import LayerNorm2d


class PromptEncoder(nn.Module):

    def __init__(
        self,
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(1024, 1024),
        mask_in_chans=16,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size

        # Random Fourier Position Embedding
        self.pe_layer = PositionEmbeddingRandom(
            num_pos_feats=embed_dim // 2,
        )

        # Four learnable prompt embeddings
        self.num_point_embeddings = 4

        self.point_embeddings = nn.ModuleList(
            [
                nn.Embedding(1, embed_dim)
                for _ in range(self.num_point_embeddings)
            ]
        )

        # Padding point embedding
        self.not_a_point_embed = nn.Embedding(
            1,
            embed_dim,
        )

        # Mask embedding network
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )

        # No-mask embedding
        self.no_mask_embed = nn.Embedding(
            1,
            embed_dim,
        )

    # --------------------------------------------------------
    # Dense Positional Encoding
    # --------------------------------------------------------

    def get_dense_pe(self):

        # (1,256,64,64)
        return self.pe_layer(
            self.image_embedding_size,
        )

    # --------------------------------------------------------
    # Batch Size
    # --------------------------------------------------------

    def _get_batch_size(
        self,
        points,
        boxes,
        masks,
    ):

        if points is not None:
            return points[0].shape[0]

        if boxes is not None:
            return boxes.shape[0]

        if masks is not None:
            return masks.shape[0]

        return 1

    # --------------------------------------------------------
    # Point Embedding
    # --------------------------------------------------------

    def _embed_points(
        self,
        points,
        labels,
        pad,
    ):

        if pad:

            # Padding Point (B,1,2)
            padding_point = torch.zeros(
                (points.shape[0], 1, 2),
                device=points.device,
            )

            # Padding Label (B,1)
            padding_label = -torch.ones(
                (labels.shape[0], 1),
                device=labels.device,
            )

            # (B,N,2) -> (B,N+1,2)
            points = torch.cat(
                [points, padding_point],
                dim=1,
            )

            # (B,N) -> (B,N+1)
            labels = torch.cat(
                [labels, padding_label],
                dim=1,
            )

        # Shift to Pixel Center (B,N,2) -> (B,N,2)
        points = points + 0.5

        # Position Encoding (B,N,2) -> (B,N,256)
        point_embedding = self.pe_layer.forward_with_coords(
            points,
            self.input_image_size,
        )

        # Padding Point
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight

        # Background Point
        point_embedding[labels == 0] += self.point_embeddings[0].weight

        # Foreground Point
        point_embedding[labels == 1] += self.point_embeddings[1].weight

        return point_embedding

    # --------------------------------------------------------
    # Box Embedding
    # --------------------------------------------------------

    def _embed_boxes(
        self,
        boxes,
    ):

        # Shift to Pixel Center (B,4) -> (B,4)
        boxes = boxes + 0.5

        # (B,4) -> (B,2,2)
        coords = boxes.reshape(
            -1,
            2,
            2,
        )

        # Position Encoding (B,2,2) -> (B,2,256)
        corner_embedding = self.pe_layer.forward_with_coords(
            coords,
            self.input_image_size,
        )

        # Top Left Corner
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight

        # Bottom Right Corner
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight

        return corner_embedding

    # --------------------------------------------------------
    # Mask Embedding
    # --------------------------------------------------------

    def _embed_masks(
        self,
        masks,
    ):

        # (B,1,256,256) -> (B,256,64,64)
        return self.mask_downscaling(masks)

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        points=None,
        boxes=None,
        masks=None,
    ):

        batch_size = self._get_batch_size(
            points,
            boxes,
            masks,
        )

        device = self.no_mask_embed.weight.device

        # Initialize Sparse Embedding (B,0,256)
        sparse_embeddings = torch.empty(
            (
                batch_size,
                0,
                self.embed_dim,
            ),
            device=device,
        )

        # ----------------------------------------------------
        # Point Prompt
        # ----------------------------------------------------

        if points is not None:

            coords, labels = points

            point_embeddings = self._embed_points(
                coords,
                labels,
                pad=(boxes is None),
            )

            # Concatenate Sparse Embeddings
            # (B,N,256)
            sparse_embeddings = torch.cat(
                [
                    sparse_embeddings,
                    point_embeddings,
                ],
                dim=1,
            )

        # ----------------------------------------------------
        # Box Prompt
        # ----------------------------------------------------

        if boxes is not None:

            box_embeddings = self._embed_boxes(
                boxes,
            )

            # Concatenate Sparse Embeddings
            sparse_embeddings = torch.cat(
                [
                    sparse_embeddings,
                    box_embeddings,
                ],
                dim=1,
            )

        # ----------------------------------------------------
        # Mask Prompt
        # ----------------------------------------------------

        if masks is not None:

            # (B,1,256,256) -> (B,256,64,64)
            dense_embeddings = self._embed_masks(
                masks,
            )

        else:

            # (1,256) -> (B,256,64,64)
            dense_embeddings = self.no_mask_embed.weight.reshape(
                1,
                self.embed_dim,
                1,
                1,
            ).expand(
                batch_size,
                -1,
                self.image_embedding_size[0],
                self.image_embedding_size[1],
            )

        return sparse_embeddings, dense_embeddings