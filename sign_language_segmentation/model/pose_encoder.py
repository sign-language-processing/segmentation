from collections import namedtuple
from typing import List

import torch
import torch.nn as nn

# named tuple for the sizes of the hidden layers, with size, channels, and temporal window
ConvDef = namedtuple("ConvDef", ["in_channels", "out_channels", "kernel_size", "stride"])


class PoseEncoderUNetBlock(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 convolutions: List[ConvDef]):
        super().__init__()

        self.encoder_layers = nn.ModuleList()
        for conv in convolutions:
            assert conv.kernel_size % 2 == 1, "Temporal window (kernel size) must be odd"
            assert conv.stride & (conv.stride - 1) == 0, "Stride must be a power of 2"

            self.encoder_layers.append(nn.Sequential(
                nn.Conv1d(
                    in_channels=conv.in_channels,
                    out_channels=conv.out_channels,
                    kernel_size=conv.kernel_size,
                    stride=conv.stride,
                    padding=conv.kernel_size // 2,
                ),
                nn.BatchNorm1d(conv.out_channels),
                nn.SiLU()
            ))

        stride_to_output_pad = {1: 0, 2: 1, 4: 3, 8: 4}

        self.decoder_layers = nn.ModuleList()
        for conv in reversed(convolutions):
            if conv.stride not in stride_to_output_pad:
                raise ValueError(f"Stride {conv.stride} not supported for output padding. Manually add it!")

            self.decoder_layers.append(nn.Sequential(
                nn.ConvTranspose1d(
                    in_channels=conv.out_channels,
                    out_channels=conv.in_channels,
                    kernel_size=conv.kernel_size,
                    stride=conv.stride,
                    padding=conv.kernel_size // 2
                ),
                nn.BatchNorm1d(conv.in_channels),
                nn.SiLU()
            ))

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [batch_size, sequence_length, input_size, input_channels]
        returns shape: [batch_size, sequence_length, input_size, output_channels]
        """
        batch_size, sequence_length, input_size, input_channels = x.shape

        # Rearrange to [batch_size * input_size, input_channels, sequence_length]
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size * input_size, input_channels, sequence_length)

        # Encode values with reducing temporal dimension
        intermediate_values = []
        for layer in self.encoder_layers:
            x = layer(x)
            intermediate_values.append(x)

        # Decode values with increasing temporal dimension, using skip connections
        for layer in self.decoder_layers:
            skip = intermediate_values.pop()
            skip_batch_size, skip_input_size, skip_sequence_length = skip.shape

            diff = skip_sequence_length - x.shape[-1]
            if diff > 0:
                left_pad = diff // 2
                right_pad = diff - left_pad  # ensures leftover goes on the right
                x = nn.functional.pad(x, (left_pad, right_pad), mode="constant", value=0)

            x = layer(x + skip)

        # [batch_size*input_size, output_channels, sequence_length]
        _, output_channels, new_sequence_length = x.shape

        # Reshape back to [batch_size, sequence_length, input_size, output_channels]
        x = x.view(batch_size, input_size, output_channels, new_sequence_length).permute(0, 3, 1, 2)

        # Average pool the channel output [batch_size, sequence_length, input_size]
        x = x.mean(dim=-1)
        return self.fc(x)


if __name__ == "__main__":
    batch_size, sequence_length = 2, 1000
    input_size, input_channels = 17, 3
    temporal_window = 11
    output_channels = 8
    output_size = 32

    model = PoseEncoderUNetBlock(input_size=input_size, output_size=output_size, convolutions=[
        ConvDef(in_channels=3, out_channels=8, kernel_size=5, stride=1),
        ConvDef(in_channels=8, out_channels=16, kernel_size=11, stride=2),
        ConvDef(in_channels=16, out_channels=32, kernel_size=21, stride=2),
        ConvDef(in_channels=32, out_channels=64, kernel_size=21, stride=2),
    ])
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    inp = torch.randn(batch_size, sequence_length, input_size, input_channels)

    out = model(inp)

    # for i in tqdm(range(100)):
    #     out = model(inp)

    print("Input shape:", inp.shape)  # [2, 16, 17, 3]
    print("Output shape:", out.shape)  # currently [2, 4, 17], i want [2, 16, 17]
