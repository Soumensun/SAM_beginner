import torch

from models.image_encoder.window import window_partition, window_unpartition

x = torch.randn(2, 64, 64, 768)

print("Input Shape:")
print(x.shape)


windows, pad_hw = window_partition(
    x,
    window_size=14,
)

print("\nAfter Window Partition:")
print(windows.shape)

print("\nPadded Size:")
print(pad_hw)


reconstructed = window_unpartition(
    windows,
    window_size=14,
    pad_hw=pad_hw,
    original_hw=(64, 64),
)

print("\nAfter Window Unpartition:")
print(reconstructed.shape)

print("\nRecovered Correctly?")
print(torch.allclose(x, reconstructed))


def main():
    pass


if __name__ == "__main__":
    main()

