import torch
from torch import nn


class CharCNN(nn.Module):

    def __init__(self, hidden_size, kernels=[2, 3, 4]):
        """1DCNN layer with max pooling

        Args:
            hidden_size (int): embedding dimension
            kernels (list, optional): kernel sizes for convolution. Defaults to [2, 3, 4].
        """

        super().__init__()

        self.pool = nn.AdaptiveMaxPool1d((1))

        self.convs = nn.ModuleList()

        for k in kernels:

            cv = nn.Conv1d(hidden_size, hidden_size, kernel_size=k, bias=False)

            self.convs.append(cv)

    def forward(self, x):
        """Forward function

        Args:
            x (torch.Tensor): [batch_size, length, hidden_size]

        Returns:
            torch.Tensor: [batch_size, hidden_size]
        """

        x = x.transpose(1, 2)

        convs = []

        for conv in self.convs:

            convolved = conv(x)

            convolved = self.pool(convolved).squeeze(-1)

            convs.append(convolved)

        convs = torch.stack(convs, dim=0)

        return convs.max(0).values
