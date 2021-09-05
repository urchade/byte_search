import torch
from torch import nn

class CharCNN(nn.Module):
    
    def __init__(self, d_model, kernels=[2, 3, 4]):
        
        super().__init__()
                
        self.convs = nn.ModuleList()
        
        for k in kernels:
            
            cv = nn.Conv1d(d_model, d_model, kernel_size=k, bias=False)
            
            self.convs.append(cv)
                                                
    def forward(self, x):
        
        x = x.transpose(1, 2)
        
        convs = []
        
        for conv in self.convs:
            
            convolved = conv(x)
            
            convolved, _ = torch.max(convolved, dim=-1)
                                    
            convs.append(convolved)
        
        convs = torch.stack(convs, dim=0)
                           
        return convs.max(0).values