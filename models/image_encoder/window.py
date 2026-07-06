import torch
import torch.nn.functional as F


# ----------------------------------------------------------
# Window Partition
# ----------------------------------------------------------
def window_partition(x, window_size):
    """
    Partition a feature map into non-overlapping windows.

    Input:
        x : (B, H, W, C)

    Output:
        windows : (B*num_windows, window_size, window_size, C)
        pad_hw  : (Hp, Wp)
    """

   
    
  
    B, H, W, C = x.shape   # x : (B, H, W, C)

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size


    if pad_h > 0 or pad_w > 0:
        x = F.pad(
            x,
            (0, 0,          # Channel
             0, pad_w,      # Width
             0, pad_h)      # Height
        )

    Hp = H + pad_h
    Wp = W + pad_w

    
    # Split Height and Width into windows  (B, Hp, Wp, C) -> (B, Hp//ws, ws, Wp//ws, ws, C)
  
    x = x.view(B, Hp // window_size,window_size, Wp // window_size, window_size, C,)

    # Shape: (B, Hp/ws, ws, Wp/ws, ws, C)


    # Swap window indices  (B, Hp/ws, Wp/ws, ws, ws, C)
   
    x = x.permute(0,1,3,2,4,5,).contiguous()

    # Shape:(B, Hp/ws, Wp/ws, ws, ws, C)

    
    # Flatten all windows into batch dimension (B*num_windows, ws, ws,  C)
    
    windows = x.view(-1, window_size,window_size,C,) # Shape:(B*num_windows, ws, ws, C)

    

    return windows, (Hp, Wp)



# Window Unpartition

def window_unpartition(windows,window_size,pad_hw,original_hw,):

    """
    Merge windows back into the original feature map.

    Input
        windows : (B*num_windows, ws, ws, C)

    Output
        x : (B, H, W, C)
    """

    Hp, Wp = pad_hw
    H, W = original_hw

  
    # Recover Batch Size
    
    # windows.shape[0] = Batch × NumWindows
   
    B = windows.shape[0] // ((Hp // window_size)* (Wp // window_size))


    # Restore window grid
    # (B*num_windows, ws, ws, C) -> (B,Hp/ws,Wp/ws,ws,ws,C)
  
    x = windows.view(
        B,
        Hp // window_size,
        Wp // window_size,
        window_size,
        window_size,
        -1,
    )

    # Shape:(B, Hp/ws, Wp/ws, ws, ws, C)

    # Undo permutation by permuting with the same indices used in window_partition

    x = x.permute(0,1,3,2,4,5,).contiguous()    # Shape:(B, Hp/ws, ws, Wp/ws, ws, C)

    # Merge windows (B, Hp, Wp, C)
   
    x = x.view(
        B,
        Hp,
        Wp,
        -1,
    )
   
    # Remove padding (B, Hp, Wp, C) -> (B, H, W, C)
   
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]

    
    return x    # Final Shape:(B, H, W, C)
