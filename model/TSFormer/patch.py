import torch
import torch.nn as nn

class Patch(nn.Module):
    def __init__(self, patch_size, input_channel, output_channel, spectral=True):
        super().__init__()
        self.output_channel = output_channel
        self.P = patch_size
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.spectral = spectral
        if spectral:
            self.emb_layer = nn.Linear(int(patch_size/2+1)*2, output_channel)
        else:
            self.input_embedding = nn.Conv2d(input_channel, output_channel, kernel_size=(self.P, 1), stride=(self.P, 1))

    def forward(self, input):
        """
        Args:
            input: B, N, C, L
        Returns:
            output: B, N, d, L/P
        """
        B, N, C, L = input.shape
        if self.spectral:
            spec_feat_ = torch.fft.rfft(input.unfold(-1, self.P, self.P), dim=-1)
            real = spec_feat_.real
            imag = spec_feat_.imag
            spec_feat = torch.cat([real, imag], dim=-1).squeeze(2)
            output = self.emb_layer(spec_feat).transpose(-1, -2)
        else:
            
            input = input.unsqueeze(-1)             # B, N, C, L, 1
            input = input.reshape(B*N, C, L, 1)                                     # B*N,  C, L, 1
            # FC for (,p) to (,1)
            output = self.input_embedding(input)                                    # B*N,  d, L/P, 1
            output = output.squeeze(-1).view(B, N, self.output_channel, -1)
            assert output.shape[-1] == L / self.P
        return output
