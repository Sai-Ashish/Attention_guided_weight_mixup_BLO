import torch
from torch import Tensor
import math

# My linear operation
class Linear(torch.nn.Linear):
    def forward(self, input: Tensor, alpha_w1, alpha_w2, pretrained=False) -> Tensor:
        if pretrained:
            return input@(((alpha_w1@alpha_w2)*self.weight).T)
        else:
            return input@(((alpha_w1@alpha_w2)*self.weight).T) + self.bias

# My linear function
class blo_linear(torch.nn.Module):
    
    def __init__(self, in_features, out_features, bias = True):
        
        super(blo_linear, self).__init__()
        
        # For convex combination
        
        self.linear_layer = Linear(in_features, out_features, bias = bias).requires_grad_() # define the linear layer
        
        self.pretrained_layer = Linear(in_features, out_features, bias = bias) # define the pretrained layer
        
        ########################################################################
        # parameter list
        r = 1
        self.alpha = torch.nn.ParameterList([torch.nn.Parameter(torch.empty(out_features, r)), torch.nn.Parameter(torch.empty(r, in_features))])
        
        ### Initialize the weights of the alpha layer ###
        torch.nn.init.normal_(self.alpha[0], 1.0, 0.005) # initialization for weights 1
        torch.nn.init.normal_(self.alpha[1], 1.0, 0.005) # initialization for weights 2

        
    def forward(self, x):
        # convex combination alpha*W+(1-alpha)*W0 alpha in [0,1]
        x_out = self.linear_layer(x, self.alpha[0], self.alpha[1], pretrained=False) # output of the linear layer which is trainable

        x_pretrained = self.pretrained_layer(x, 1 - self.alpha[0], 1 - self.alpha[1], pretrained=True) # the output from the second half of passing through the pretrained model
        
        return x_out + x_pretrained