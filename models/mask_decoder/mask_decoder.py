import torch
import torch.nn as nn

from ..two_way_transformer.two_way_transformer import TwoWayTransformer
from ..image_encoder.layernorm2d import LayerNorm2d
from .mlp import MLP


class MaskDecoder(nn.Module):

    def __init__(
        self,
        transformer_dim=256,
        transformer=None,
        num_multimask_outputs=3,
        activation=nn.GELU,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    ):
        super().__init__()

        self.transformer_dim = transformer_dim

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)

        self.num_mask_tokens = num_multimask_outputs + 1

        self.mask_tokens = nn.Embedding(
            self.num_mask_tokens,
            transformer_dim,
        )

        if transformer is None:
            self.transformer = TwoWayTransformer(
                depth=2,
                embedding_dim=transformer_dim,
                num_heads=8,
                mlp_dim=2048,
            )
        else:
            self.transformer = transformer

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim,
                transformer_dim // 4,
                kernel_size=2,
                stride=2,
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4,
                transformer_dim // 8,
                kernel_size=2,
                stride=2,
            ),
            activation(),
        )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(
                    input_dim=transformer_dim,
                    hidden_dim=transformer_dim,
                    output_dim=transformer_dim // 8,
                    num_layers=3,
                )
                for _ in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            input_dim=transformer_dim,
            hidden_dim=iou_head_hidden_dim,
            output_dim=self.num_mask_tokens,
            num_layers=iou_head_depth,
        )


    def predict_masks(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
    ):
        """
        image_embeddings         : (B,256,64,64)

        image_pe                 : (B,256,64,64)

        sparse_prompt_embeddings : (B,N,256)

        dense_prompt_embeddings  : (B,256,64,64)
        """

        # Create output tokens

        # (1,256) + (4,256) -> (5,256)
        output_tokens = torch.cat(
            [
                self.iou_token.weight,
                self.mask_tokens.weight,
            ],
            dim=0,
        )

        # Batch Dimension
        b = image_embeddings.shape[0]

        # Expand output tokens to batch (5,256) -> (B,5,256)
        output_tokens = output_tokens.unsqueeze(0).expand(
            b,
            -1,
            -1,
        )

        # Concatenate output tokens and sparse prompt embeddings (B,5,256) + (B,N,256) -> (B,5+N,256)
        tokens = torch.cat(
            [
                output_tokens,
                sparse_prompt_embeddings,
            ],
            dim=1,
        )

        # Expand image embeddings to batch (B,256,64,64) -> (B,256,64,64)
        image_embeddings = image_embeddings + dense_prompt_embeddings

        # Repeat image position embeddings to batch (1,256,64,64) -> (B,256,64,64)
        image_pe = image_pe.expand(
            b,
            -1,
            -1,
            -1,
        )

        # --------------------------------------------------------
        # 3. Two Way Transformer
        # --------------------------------------------------------

        # Two Way Transformer
        #
        # Input :
        #   tokens           : (B,5+N,256)
        #   image_embeddings : (B,256,64,64)
        #   image_pe         : (B,256,64,64)
        #
        # Output :
        #   queries          : (B,5+N,256)
        #   keys             : (B,256,64,64)
        queries, keys = self.transformer(
            image_embeddings,
            image_pe,
            tokens,
        )

        # Output tokens corresponding to IoU prediction token (B,1,256)
        iou_token_out = queries[:, 0, :]

        # Output tokens corresponding to mask tokens (B,4,256)
        mask_tokens_out = queries[:, 1 : 1 + self.num_mask_tokens, :]

        # Keys (B, H*W, C) -> (B, H, W, C) -> (B, C, H, W)
        b_keys, _, h_keys, w_keys = image_embeddings.shape
        keys = keys.reshape(b_keys, h_keys, w_keys, keys.shape[-1]).permute(0, 3, 1, 2)

        # --------------------------------------------------------
        # 4. Output Upscaling
        # --------------------------------------------------------

        # Output Upscaling (B,256,64,64) -> (B,32,256,256)
        upscaled_embedding = self.output_upscaling(
            keys,
        )

        # --------------------------------------------------------
        # 5. Hypernetworks
        # --------------------------------------------------------

        # Hypernetworks
        #
        # Input :
        #   mask_tokens_out : (B,4,256)
        #
        # Output :
        #   hyper_in        : (B,4,32)

        hyper_in_list = []

        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](
                    mask_tokens_out[:, i, :],
                )
            )

        # (B,4,32)
        hyper_in = torch.stack(
            hyper_in_list,
            dim=1,
        )

        # --------------------------------------------------------
        # 6. Mask Generation
        # --------------------------------------------------------

        # Reshape upscaled embedding (B,32,256,256) -> (B,32,65536)
        b, c, h, w = upscaled_embedding.shape

        upscaled_embedding = upscaled_embedding.view(
            b,
            c,
            h * w,
        )

        # (B',4,32) @ (B',32,65536) -> (B',4,65536)
        masks = hyper_in @ upscaled_embedding

        # (B',4,65536) -> (B',4,256,256)
        masks = masks.view(
            b,
            self.num_mask_tokens,
            h,
            w,
        )

        # --------------------------------------------------------
        # 7. IoU Prediction
        # --------------------------------------------------------

        # (B',256) -> (B',4)
        iou_pred = self.iou_prediction_head(
            iou_token_out,
        )

        return masks, iou_pred

    def forward(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output,
    ):
        """
        image_embeddings         : (B,256,64,64)

        image_pe                 : (B,256,64,64)

        sparse_prompt_embeddings : (B,N,256)

        dense_prompt_embeddings  : (B,256,64,64)

        multimask_output         : bool
        """

        # Predict Mask

        # masks    : (B,4,256,256)
        # iou_pred : (B,4)
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select output mask

        if multimask_output:

            # Return Mask Tokens 1,2,3
            mask_slice = slice(1, None)

        else:

            # Return Best Mask
            mask_slice = slice(0, 1)

        # (B,4,256,256) -> (B,3,256,256) or (B,1,256,256)
        masks = masks[:, mask_slice, :, :]

        # (B,4) -> (B,3) or (B,1)
        iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred
