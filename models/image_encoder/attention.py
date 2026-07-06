import torch
import torch.nn as nn
import torch.nn.functional as F


def get_rel_pos(
    q_size,
    k_size,
    rel_pos,
):
    """
    Get relative positional embeddings according to the relative
    positions of query and key.

    Args
    ----
    q_size : int
        Query spatial size

    k_size : int
        Key spatial size

    rel_pos :
        Shape
        (L, head_dim)

        where

        L = 2*window_size-1

    Returns
    -------
    Relative positional embeddings

    Shape

    (q_size, k_size, head_dim)
    """


    # Required table length: q=14,k=14 then max_rel_dist=27
    
  

    max_rel_dist = 2 * max(q_size, k_size) - 1

    if rel_pos.shape[0] != max_rel_dist:

# (L,C) -> (1,C,L)
    

        rel_pos_resized = F.interpolate(

            rel_pos.reshape(
                1,
                rel_pos.shape[1],
                rel_pos.shape[0],
            ),

            size=max_rel_dist,

            mode="linear",

        )

        # (1,C,L) -> (L,C)
   

        rel_pos_resized = rel_pos_resized.reshape(
            rel_pos.shape[1],
            max_rel_dist,
        ).permute(1,0)

    else:

        rel_pos_resized = rel_pos



    q_coords = torch.arange(q_size)

    k_coords = torch.arange(k_size)


    # Scale coordinates nedded when q_size != k_size

    q_coords = q_coords * max(k_size / q_size, 1.0)

    k_coords = k_coords * max(q_size / k_size, 1.0)


    # Relative coordinates: shape (q_size,k_size)


    relative_coords = (

        q_coords[:,None]

        -

        k_coords[None,:]

    )

    
    # Shift: example: q_size=14,k_size=14, then relative_coords = -13...13 -> 0...26

  
    

    relative_coords += (k_size - 1) * max(
        q_size / k_size,
        1.0,
    )

   
    # Lookup

    # Output

    # (q_size,k_size,head_dim)
 

    return rel_pos_resized[
        relative_coords.long()
    ]



def add_decomposed_rel_pos(
    attn,
    q,
    rel_pos_h,
    rel_pos_w,
    q_size,
    k_size,
):

    Rh = get_rel_pos(
        q_size[0],
        k_size[0],
        rel_pos_h,
    )

    Rw = get_rel_pos(
        q_size[1],
        k_size[1],
        rel_pos_w,
    )

    B, heads, _, dim = q.shape

    q = q.reshape(
        B,
        heads,
        q_size[0],
        q_size[1],
        dim,
    )

    rel_h = torch.einsum(
        "bhqwc,qkc->bhqwk",
        q,
        Rh,
    )

    rel_w = torch.einsum(
        "bhqwc,wkc->bhqwk",
        q,
        Rw,
    )

    attn = attn.reshape(
        B,
        heads,
        q_size[0],
        q_size[1],
        k_size[0],
        k_size[1],
    )

    attn = (
        attn
        + rel_h[:, :, :, :, :, None]
        + rel_w[:, :, :, :, None, :]
    )

    attn = attn.reshape(
        B,
        heads,
        q_size[0] * q_size[1],
        k_size[0] * k_size[1],
    )

    return attn


class Attention(nn.Module):

    def __init__(
        self,
        dim=768,
        num_heads=12,
        use_rel_pos=True,
        input_size=(14,14),
    ):
        super().__init__()

        self.dim = dim
        self.input_size = input_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos

        if use_rel_pos:

            self.rel_pos_h = nn.Parameter(
                torch.zeros(
                    2 * input_size[0] - 1,
                    self.head_dim,
                )
            )

            self.rel_pos_w = nn.Parameter(
                torch.zeros(
                    2 * input_size[1] - 1,
                    self.head_dim,
                )
            )

    def forward(self, x):
        """
        Args:
            x : (B, N, C)

            B = Batch Size (actually Batch × Number of Windows)
            N = Number of Tokens in one window (14×14 = 196)
            C = Embedding Dimension (768)

        Returns:
            out : (B, N, C)
        """

        # ---------------------------------------------------------
        # Input
        #
        # x
        # Shape:
        # (B, 196, 768)
     
        B, N, C = x.shape

       
        # Query Projection
        #
        # (B,196,768)
        # ->
        # (B,196,768)
 
        Q = self.q_proj(x)

        
        # Key Projection
        #
        # (B,196,768)
        # ->
        # (B,196,768)
       
        K = self.k_proj(x)

       
        # Value Projection
        #
        # (B,196,768)
        # ->
        # (B,196,768)
      
        V = self.v_proj(x)

       


        # Query
        #
        # (B,196,768)
        # ->
        # (B,196,12,64)
  
        Q = Q.reshape(
            B,
            N,
            self.num_heads,
            self.head_dim,
        )

        # (B,196,12,64)
        # ->
        # (B,12,196,64)
        Q = Q.permute(
            0,
            2,
            1,
            3,
        )

        K = K.reshape(
            B,
            N,
            self.num_heads,
            self.head_dim,
        )

        K = K.permute(
            0,
            2,
            1,
            3,
        )

        # Shape
        # (B,12,196,64)

        V = V.reshape(
            B,
            N,
            self.num_heads,
            self.head_dim,
        )

        V = V.permute(
            0,
            2,
            1,
            3,
        )

        # Shape
        # (B,12,196,64)



        # Q
        # (B,12,196,64)
        #
        # K^T
        # (B,12,64,196)
        #
        # ->
        #
        # (B,12,196,196)
        attn = Q @ K.transpose(-2, -1)


        attn = attn * self.scale



        if self.use_rel_pos:

            attn = add_decomposed_rel_pos(
                attn=attn,
                q=Q,
                rel_pos_h=self.rel_pos_h,
                rel_pos_w=self.rel_pos_w,
                q_size=self.input_size,
                k_size=self.input_size,
            )

            # Shape
            # (B,12,196,196)



        attn = torch.softmax(
            attn,
            dim=-1,
        )

        # Shape
        # (B,12,196,196)

      

        out = attn @ V

        # Shape
        # (B,12,196,64)


        out = out.permute(
            0,
            2,
            1,
            3,
        )

        # Shape
        # (B,196,12,64)

        out = out.reshape(
            B,
            N,
            C,
        )

        # Shape
        # (B,196,768)

       

        out = self.out_proj(out)

        # Shape
        # (B,196,768)

        return out