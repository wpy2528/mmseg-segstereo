import torch
import torch.nn as nn

class ApproxDeconv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, block_index):
        super().__init__()
        kd, kh, kw = kernel_size
        sd, sh, sw = stride
        pd, ph, pw = padding

        self.deconv2d = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=(kh, kw),
            stride=(sh, sw),
            padding=(ph, pw),
            bias=bias
        )
        self.block_index = block_index
        if self.block_index == 3:
            conv2d_in_channels = 6
            conv2d_out_channels = 12
        elif self.block_index == 2:
            conv2d_in_channels = 12
            conv2d_out_channels = 24
        elif self.block_index == 1:
            conv2d_in_channels = 24
            conv2d_out_channels = 48
        else:
            raise ValueError(f"Invalid block index: {self.block_index}")
        self.conv2d= nn.Conv2d(
            conv2d_in_channels, conv2d_out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=bias
        )

    def forward(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape

        # ---- Step1: flatten depth ----
        x = x.permute(0, 2, 1, 3, 4)          # [B, D, C, H, W]
        x = x.reshape(B*D, C, H, W)           # [B*D, C, H, W]

        # ---- Step2: 2D deconv on (H, W) ----
        x = self.deconv2d(x)                  # [B*D, Cout, H’, W’]

        # ---- Step3: restore [B, Cout, D, H’, W’] ----
        _, Cout, H2, W2 = x.shape
        x = x.view(B, D, Cout, H2, W2)        # [B, D, Cout, H’, W’]
        x = x.permute(0, 2, 1, 3, 4)          # [B, Cout, D, H’, W’]

        # ---- Step4: 2D deconv along depth (using 1xkd kernel) ----
        x = x.reshape(B*Cout, D, H2, W2)                      # [B*Cout, D, H',W']
        x = self.conv2d(x)                    # [B*Cout, D', H',W']

        x = x.reshape(B, Cout, -1, H2, W2)

        return x

def replace_deconv3d_to_deconv2d_with_deconv1d(in_channels, out_channels, bias, kernel_size, padding, stride, block_index):
    return ApproxDeconv3D(in_channels, out_channels, kernel_size, stride, padding, bias, block_index)


if __name__ == "__main__":
    B, C, D, H, W = 2, 48, 10, 32, 32
    x = torch.randn(B, C, D, H, W)

    # 原始 ConvTranspose3d
    deconv3d = nn.ConvTranspose3d(
        in_channels=48, out_channels=32,
        kernel_size=(4,4,4), stride=(2,2,2),
        padding=(1,1,1), bias=False
    )
    y3d = deconv3d(x)
    print("ConvTranspose3d 输出 shape:", y3d.shape)

    # 替代版
    deconv_approx = replace_deconv3d_to_deconv2d_with_deconv1d(
        in_channels=48,
        out_channels=32,
        bias=False,
        kernel_size=(4,4,4),
        stride=(2,2,2),
        padding=(1,1,1)
    )
    y_approx = deconv_approx(x)
    print("替代版输出 shape:", y_approx.shape)
