import torch
import torch.nn as nn

from .image_encoder.image_encoder import ImageEncoder
from .prompt_encoder.prompt_encoder import PromptEncoder
from .mask_decoder.mask_decoder import MaskDecoder
from .two_way_transformer.two_way_transformer import TwoWayTransformer


class SAM(nn.Module):

    def __init__(
        self,
        image_encoder=None,
        prompt_encoder=None,
        mask_decoder=None,
    ):
        super().__init__()

        # Image Encoder

        if image_encoder is None:

            self.image_encoder = ImageEncoder()

        else:

            self.image_encoder = image_encoder

        # Prompt Encoder

        if prompt_encoder is None:

            self.prompt_encoder = PromptEncoder()

        else:

            self.prompt_encoder = prompt_encoder

        # Mask Decoder

        if mask_decoder is None:

            self.mask_decoder = MaskDecoder(

                transformer=TwoWayTransformer(),

            )

        else:

            self.mask_decoder = mask_decoder


    def forward(
        self,
        image,
        points=None,
        boxes=None,
        masks=None,
        multimask_output=False,
    ):
        """
        Args
        ----
        image : (B,3,1024,1024)

        points : ((B,N,2),(B,N))

        boxes : (B,4)

        masks : (B,1,256,256)

        multimask_output : bool

        Returns
        -------
        masks : (B,1,256,256) or (B,3,256,256)

        iou_predictions : (B,1) or (B,3)
        """

        # --------------------------------------------------------
        # 1. Image Encoder
        # --------------------------------------------------------

        # Image Encoder (B,3,1024,1024) -> (B,256,64,64)
        image_embeddings = self.image_encoder(
            image,
        )

        # --------------------------------------------------------
        # 2. Prompt Encoder
        # --------------------------------------------------------

        # Prompt Encoder
        #
        # Sparse Embedding : (B,N,256)
        #
        # Dense Embedding : (B,256,64,64)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(

            points=points,

            boxes=boxes,

            masks=masks,

        )

        # --------------------------------------------------------
        # 3. Dense Position Encoding
        # --------------------------------------------------------

        # Dense Position Encoding (1,256,64,64)
        image_pe = self.prompt_encoder.get_dense_pe()

        # --------------------------------------------------------
        # 4. Mask Decoder
        # --------------------------------------------------------

        # Mask Decoder
        #
        # masks : (B,1,256,256) or (B,3,256,256)
        #
        # iou_predictions : (B,1) or (B,3)
        masks, iou_predictions = self.mask_decoder(

            image_embeddings=image_embeddings,

            image_pe=image_pe,

            sparse_prompt_embeddings=sparse_embeddings,

            dense_prompt_embeddings=dense_embeddings,

            multimask_output=multimask_output,

        )

        return masks, iou_predictions

    

    
