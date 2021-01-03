"""
graph neural networks for link prediction

    link preiction モデルのclassを定義する

Todo:

"""

import torch
import torch.nn.functional as F

from link_prediction.my_util import my_utils

class Bias(torch.nn.Module):
    '''Custom Layer
    
    Layer which add the same scalar to an input tensor

    Attributes:
        bias (torch.nn.Parameter[1]): scalar bias
    '''
    def __init__(self, initial_value=-2.0):
        super(Bias, self).__init__()
        self.bias = torch.nn.Parameter(torch.Tensor([initial_value]))

    def forward(self, x):
        x = x + self.bias.expand(x.size())
        return x

class GAE(torch.nn.Module):
    '''Graph Auto-Encoder

    gets link probabilities by calculating inner products of node features

    Attributes:
        device (:obj:`int`): 'cpu', 'cuda'. 
        encoder:
        self_loop_bias:
        sigmoid_bias
        bias (torch.nn.ModuleList): list of sigmoid bias.
    '''   
    def __init__(self, encoder, self_loop_mask = True, sigmoid_bias = False, sigmoid_bias_initial_value=-2.0):
        '''
        Args:
            encoder
            self_loop_bias
            sigmoid_bias
        '''
        super(GAE, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = encoder
        self.self_loop_mask = self_loop_mask
        self.sigmoid_bias = sigmoid_bias

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(1, 1))
        self.bias = torch.nn.ModuleList()
        self.bias.append(Bias(initial_value=sigmoid_bias_initial_value))

    def encode(self, *args, **kwargs):
        '''
        runs the encoder and computes node-wise latent features.

        Returns:
            z (torch.tensor[num_nodes, output_channels]): node-wise latent features.
        '''
        return self.encoder(*args, **kwargs)

    def decode(self, z, decode_edge_index=None):
        '''
        gets link probabilities from the outputs of the encode model

        Parameters:
            z (torch.tensor[num_nodes, output_channels]): node-wise latent features.

        Returns:
            probs (torch.tensor[num_edges, num_edges]: adjacency matrix of link probabilities.
        '''
        if self.sigmoid_bias is True:
            if self.self_loop_mask is True:
                probs = torch.sigmoid((self.bias[0](torch.mm(z, z.t()))) * (torch.eye(z.size(0))!=1).to(self.device))
            else:
                probs = torch.sigmoid(self.bias[0](torch.mm(z, z.t())))
        else:
            if self.self_loop_mask is True:
                probs = torch.sigmoid((torch.mm(z, z.t())) * (torch.eye(z.size(0))!=1).to(self.device))
            else:
                probs = torch.sigmoid(torch.mm(z, z.t()))
        
        if decode_edge_index is not None:
            return probs[decode_edge_index.cpu().numpy()]
        else:
            return probs.flatten()

    def encode_decode(self, decode_edge_index=None, *args, **kwargs):
        '''
        runs the encoder, computes node-wise latent variables, and gets link probabilities.

        Returns:
            probs (torch.tensor[num_edges, num_edges]: adjacency matrix of link probabilities.
        '''
        z = self.encode(*args, **kwargs)
        return self.decode(z, decode_edge_index)

class VGAE(GAE):
    r'''The Variational Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper.
    Args:
        encoder (Module): The encoder module to compute :math:`\mu` and
            :math:`\log\sigma^2`.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    '''
    def __init__(self, encoder, self_loop_mask = True, sigmoid_bias = False, sigmoid_bias_initial_value=-2.0):
        super(VGAE, self).__init__(encoder, self_loop_mask, sigmoid_bias)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.MAX_LOGSTD = 10

        self.bias = torch.nn.ModuleList()
        self.bias.append(Bias(initial_value=sigmoid_bias_initial_value))

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs):
        '''Runs the encoder and computes node-wise latent variables.
        '''

        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=self.MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def kl_loss(self, mu=None, logstd=None):
        r'''
        Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.
        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logstd (Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        '''

        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=self.MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

class Cat_Linear_Decoder(torch.nn.Module):
    def __init__(self, encoder, in_channels, hidden_channels, out_channels=1, num_layers=2, dropout=0.5, self_loop_mask = True, sigmoid_bias=True, sigmoid_bias_initial_value=-2.0):
        super(Cat_Linear_Decoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = encoder
        self.self_loop_mask = self_loop_mask
        self.sigmoid_bias = sigmoid_bias

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels * 2, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.bias = torch.nn.ModuleList()
        self.bias.append(Bias(initial_value=sigmoid_bias_initial_value))

        self.dropout = dropout

    def encode(self, *args, **kwargs):
        '''Runs the encoder and computes node-wise latent variables.
        '''

        return self.encoder(*args, **kwargs)

    def decode(self, z, decode_edge_index=None):
        if decode_edge_index is not None:
            z_i = z[torch.cat([decode_edge_index[0], decode_edge_index[1]], dim=-1)]
            z_j = z[torch.cat([decode_edge_index[1], decode_edge_index[0]], dim=-1)]
            z_ij = torch.cat([z_i, z_j], dim=0)

            for _ in range(z_ij.size(0)):
                for lin in self.lins[:-1]:
                    x = lin(z_ij)
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.lins[-1](x)

            if self.sigmoid_bias is True:
                return torch.sigmoid(self.bias[0](x)).flatten()
            else:
                return torch.sigmoid(x).flatten()

        else:
            probs = torch.zeros(0, z.size(0))

            for i in range(z.size(0)):
                z_i = z[[i]*z.size(0)]
                z_ij = torch.cat([z_i, z], dim=-1)

                for _ in range(z_ij.size(0)):
                    for lin in self.lins[:-1]:
                        x = lin(z_ij)
                        x = F.relu(x)
                        x = F.dropout(x, p=self.dropout, training=self.training)
                    x = self.lins[-1](x)

                if self.sigmoid_bias is True:
                    probs = torch.cat([probs, torch.sigmoid(self.bias[0](x))], dim=0)
                else:
                    probs = torch.cat([probs, torch.sigmoid(x)], dim=0)

            return probs.flatten()


    def encode_decode(self, decode_edge_index, *args, **kwargs):
        return self.decode(self.encoder(*args, **kwargs), decode_edge_index)