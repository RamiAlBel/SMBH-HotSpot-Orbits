import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int] = (256, 256),
        num_blocks: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        layers: list[nn.Module] = []
        
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_blocks):
            block = nn.Sequential(
                nn.Linear(hidden_dims[0], hidden_dims[0]),
                nn.BatchNorm1d(hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims[0], hidden_dims[0]),
                nn.BatchNorm1d(hidden_dims[0]),
            )
            layers.append(block)
        
        self.stem = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[0], 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem[0](x)
        for layer in self.stem[1:]:
            if isinstance(layer, nn.Sequential):
                res = out
                out = layer(out)
                out = torch.relu(out + res)
            else:
                out = layer(out)
        return self.head(out)
