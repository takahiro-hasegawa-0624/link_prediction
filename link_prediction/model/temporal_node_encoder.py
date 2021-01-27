import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GCN2Conv, GATConv, global_mean_pool, JumpingKnowledge

from link_prediction.my_util import my_utils

from torch.nn import LSTM

class GCRN(torch.nn.Module):  
    '''Vanilla Graph Neural Network

    ノードの特徴量に対してVanilla GCNを適用するモデル

    Attributes:
        device (:obj:`int`): 'cpu', 'cuda'. 
        lins (torch.nn.ModuleList): 全結合層のリスト.
        convs (torch.nn.ModuleList): torch_geometric.nn.GCNConvのリスト.
        batchnorms (torch.nn.ModuleList): batch normalizationのリスト.
        activation (obj`int` or None): activation functionを指定。None, "relu", "leaky_relu", or "tanh". (Default: None)
        dropout (float): 各層のDropoutの割合. 
        num_layers (int): 隠れ層の数.
        hidden_channels_str (str): 各層の出力の次元を文字列として記録.
        train_pos_edge_adj_t (torch.SparseTensor[2, num_pos_edges]): trainデータのリンク.
    '''     
    def __init__(self, data_list, decode_modelname, train_pos_edge_adj_t, num_hidden_channels = None, num_layers = None, hidden_channels = None, activation = None, dropout = 0.0, future_prediction=True):
        '''
        Args:
            data (torch_geometric.data.Data): グラフデータ.
            decode_modelname
            train_pos_edge_adj_t (torch.SparseTensor[2, num_pos_edges]): trainデータのリンク.
            num_hidden_channels (int or None): 隠れ層の出力次元数. 全ての層で同じ値が適用される. (Default: None)
            num_layers (int or None): 隠れ層の数. (Default: None)
            hidden_channels (list of int, or None): 各隠れ層の出力の配列. 指定するとnum_hidden_channels とnum_layersは無効化される. (Default: None)
            activation (obj`int` or None): activation functionを指定。None, "relu", "leaky_relu", or "tanh". (Default: None)
            dropout (float): 各層のDropoutの割合. (Default: 0.0)
        '''
        super(GCRN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decode_modelname = decode_modelname

        self.train_pos_edge_adj_t = train_pos_edge_adj_t

        self.lins = torch.nn.ModuleList()

        self.recurrents = torch.nn.ModuleList()
        
        self.convs = torch.nn.ModuleList()

        self.batchnorms = torch.nn.ModuleList()

        if hidden_channels is None:
            self.hidden_channels_str = (f'{num_hidden_channels}_'*num_layers)[:-1]

            self.num_layers = num_layers

            for t in range(len(data_list)):
                self.convs.append(GCNConv(data_list[t].x.size(1), num_hidden_channels))
                for _ in range(num_layers - 1):
                    self.convs.append(GCNConv(num_hidden_channels, num_hidden_channels))

                for _ in range(num_layers - 1):
                    self.batchnorms.append(torch.nn.BatchNorm1d(num_hidden_channels))

                if self.decode_modelname in ['VGAE', 'Shifted-VGAE']:
                    self.convs.append(GCNConv(num_hidden_channels, num_hidden_channels))
                if self.decode_modelname == 'S_VAE':
                    self.convs.append(GCNConv(num_hidden_channels, 1))

            self.recurrents.append(LSTM(input_size = num_hidden_channels, hidden_size = num_hidden_channels))


        else:            
            self.hidden_channels_str = ''
            for num in hidden_channels:
                self.hidden_channels_str += str(num)+'_'
            self.hidden_channels_str = self.hidden_channels_str[:-1]

            self.num_layers = len(hidden_channels)

            for t in range(len(data_list)):
                self.convs.append(GCNConv(data_list[t].x.size(1), hidden_channels[0]))
                for i in range(len(hidden_channels) - 1):
                    self.convs.append(GCNConv(hidden_channels[i], hidden_channels[i+1]))

                for i in range(len(hidden_channels) - 1):
                    self.batchnorms.append(torch.nn.BatchNorm1d(hidden_channels[i]))

                if self.decode_modelname in ['VGAE', 'Shifted-VGAE']:
                    self.convs.append(GCNConv(hidden_channels[-2], hidden_channels[-1]))
                if self.decode_modelname == 'S_VAE':
                    self.convs.append(GCNConv(hidden_channels[-2], 1))

            self.recurrents.append(LSTM(input_size = hidden_channels[-1], hidden_size = hidden_channels[-1]))

        self.activation = activation
        self.dropout = dropout
        self.future_prediction = future_prediction

    def forward(self, x_seq, edge_index_seq):
        '''
        Parameters:
            x (torch.tensor[seq_length, num_nodes, input_channels]): input of the model (node features).
            edge_index (torch.tensor[seq_length, 2, num_edges]): tonsor of the pair of node indexes with edges.

        Returns:
            z (torch.tensor[seq_length, num_nodes, output_channels]): node-wise latent features.
        '''
                
        z_seq = []
        hx_list = [None]*(len(x_seq)+1)
        for t, x in enumerate(x_seq):
            if (self.future_prediction is True) and (t == len(x_seq) - 1):
                break

            z = x
            for i in range(len(self.convs)):
                idx = t*len(x_seq) + i
                z = F.dropout(z, self.dropout, training = self.training)

                z = self.convs[idx](z, edge_index_seq[t])
                if i < len(self.convs) - 1:
                    z = self.batchnorms[i](z)
                    if self.activation == "relu":
                        z = z.relu()
                    elif self.activation == "leaky_relu":
                        z = F.leaky_relu(z, negative_slope=0.01)
                    elif self.activation == "tanh":
                        z = torch.tanh(z)

                z_seq.append(z)

        z_seq_tensor = torch.stack(z_seq,0)
        z_seq_tensor , (h_, c_) = self.recurrents[0](z_seq_tensor)

        for i in range(len(z_seq)):
            z_seq[i] = z_seq_tensor[i]

        if self.future_prediction is True:
            z_seq.append(h_[-1][0]).squeeze()

        return z_seq

class GCRNII(torch.nn.Module):       
    '''GCNII

    ノードの特徴量に対してGCNIIを適用するモデル

    Attributes:
        device (:obj:`int`): 'cpu', 'cuda'. 
        lins (torch.nn.ModuleList): 全結合層のリスト.
        convs (torch.nn.ModuleList): torch_geometric.nn.GCNConvのリスト.
        batchnorms (torch.nn.ModuleList): batch normalizationのリスト.
        activation (obj`int` or None): activation functionを指定。None, "relu", "leaky_relu", or "tanh". (Default: None)
        dropout (float): 各層のDropoutの割合. 
        num_layers (int or None): 隠れ層の数.
        hidden_channels_str (str): 各層の出力の次元を文字列として記録
        train_pos_edge_adj_t (torch.SparseTensor[2, num_pos_edges]): trainデータのリンク.
    '''    

    def __init__(self, data, decode_modelname, train_pos_edge_adj_t, num_hidden_channels, num_layers, alpha=0.1, theta=0.5, shared_weights = True, activation = None, dropout = 0.0):
        '''
        Args:
            data (torch_geometric.data.Data): グラフデータ.
            decode_modelname
            train_pos_edge_adj_t (torch.SparseTensor[2, num_pos_edges]): trainデータのリンク.
            num_hidden_channels (int or None): 隠れ層の出力次元数. 全ての層で同じ値が適用される.
            num_layers (int or None): 隠れ層の数.
            alpha (float): convolution後に初期層を加える割合. (Default: 0.1)
            theta (float): .
            shared_weights (bool): . (Default: True)
            activation (obj`int` or None): activation functionを指定。None, "relu", "leaky_relu", or "tanh". (Default: None)
            dropout (float): 各層のDropoutの割合. (Default: 0.0)
        '''
        super(GCRNII, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decode_modelname = decode_modelname

        self.num_layers = num_layers
        self.train_pos_edge_adj_t = train_pos_edge_adj_t
        self.hidden_channels_str = (f'{num_hidden_channels}_'*num_layers)[:-1]

        self.lins = torch.nn.ModuleList()
        self.lins.append(GCNConv(data.x.size(1), num_hidden_channels))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers - 1):
            self.convs.append(GCN2Conv(num_hidden_channels, alpha, theta, layer+1, shared_weights = shared_weights, normalize = False))

        if self.decode_modelname in ['VGAE', 'Shifted-VGAE']:
            self.convs.append(GCN2Conv(num_hidden_channels, alpha, theta, layer+1, shared_weights = shared_weights, normalize = False))
        if self.decode_modelname == 'S_VAE':
            self.convs.append(GCNConv(num_hidden_channels, 1))

        self.batchnorms = torch.nn.ModuleList()
        for layer in range(num_layers - 2):
            self.batchnorms.append(torch.nn.BatchNorm1d(num_hidden_channels))

        self.activation = activation
        self.dropout = dropout    

    def forward(self, x, edge_index=None):
        '''
        Parameters:
            x (torch.tensor[num_nodes, input_channels]): input of the model (node features).
            edge_index (torch.tensor[2, num_edges]): tonsor of the pair of node indexes with edges.

        Returns:
            z (torch.tensor[num_nodes, output_channels]): node-wise latent features.
        '''
        if edge_index is None:
            edge_index = self.train_pos_edge_adj_t

        # 線形変換で次元削減して入力とする
        z = self.lins[0](x, edge_index)
        
        x_0 = z

        if self.num_layers==2:
            z = z.relu()

        if self.decode_modelname in ['VGAE', 'Shifted-VGAE', 'S_VAE']:
            for i, conv in enumerate(self.convs[:-2]):
                z = F.dropout(z, self.dropout, training = self.training)
                z = conv(z, x_0, edge_index)
                z = self.batchnorms[i](z)
                if self.activation == "relu":
                    z = z.relu()
                elif self.activation == "leaky_relu":
                    z = F.leaky_relu(z, negative_slope=0.01)
                elif self.activation == "tanh":
                    z = torch.tanh(z)

            if self.decode_modelname in ['VGAE', 'Shifted-VGAE']:
                return self.convs[-2](z, x_0, edge_index), self.convs[-1](z, x_0, edge_index)
            else:
                return self.convs[-2](z, x_0, edge_index), self.convs[-1](z, edge_index)

        else:
            for i, conv in enumerate(self.convs):
                z = F.dropout(z, self.dropout, training = self.training)
                z = conv(z, x_0, edge_index)
                if i < len(self.convs) - 1:
                    z = self.batchnorms[i](z)
                    if self.activation == "relu":
                        z = z.relu()
                    elif self.activation == "leaky_relu":
                        z = F.leaky_relu(z, negative_slope=0.01)
                    elif self.activation == "tanh":
                        z = torch.tanh(z)

            return z



class EvolveGCNO(torch.nn.Module):  
    '''Vanilla Graph Neural Network

    ノードの特徴量に対してVanilla GCNを適用するモデル

    Attributes:
        device (:obj:`int`): 'cpu', 'cuda'. 
        lins (torch.nn.ModuleList): 全結合層のリスト.
        convs (torch.nn.ModuleList): torch_geometric.nn.GCNConvのリスト.
        batchnorms (torch.nn.ModuleList): batch normalizationのリスト.
        activation (obj`int` or None): activation functionを指定。None, "relu", "leaky_relu", or "tanh". (Default: None)
        dropout (float): 各層のDropoutの割合. 
        num_layers (int): 隠れ層の数.
        hidden_channels_str (str): 各層の出力の次元を文字列として記録.
        train_pos_edge_adj_t (torch.SparseTensor[2, num_pos_edges]): trainデータのリンク.
    '''     
    def __init__(self, data_list, decode_modelname, train_pos_edge_adj_t, num_hidden_channels = None, num_layers = None, hidden_channels = None, activation = None, dropout = 0.0, future_prediction=True):
        '''
        Args:
            data (torch_geometric.data.Data): グラフデータ.
            decode_modelname
            train_pos_edge_adj_t (torch.SparseTensor[2, num_pos_edges]): trainデータのリンク.
            num_hidden_channels (int or None): 隠れ層の出力次元数. 全ての層で同じ値が適用される. (Default: None)
            num_layers (int or None): 隠れ層の数. (Default: None)
            hidden_channels (list of int, or None): 各隠れ層の出力の配列. 指定するとnum_hidden_channels とnum_layersは無効化される. (Default: None)
            activation (obj`int` or None): activation functionを指定。None, "relu", "leaky_relu", or "tanh". (Default: None)
            dropout (float): 各層のDropoutの割合. (Default: 0.0)
        '''
        super(EvolveGCNO, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decode_modelname = decode_modelname

        self.train_pos_edge_adj_t = train_pos_edge_adj_t

        self.lins = torch.nn.ModuleList()
        
        self.convs = torch.nn.ModuleList()

        self.recurrents = torch.nn.ModuleList()

        self.batchnorms = torch.nn.ModuleList()

        if hidden_channels is None:
            hidden_channels
            self.hidden_channels_str = (f'{num_hidden_channels}_'*num_layers)[:-1]

            self.num_layers = num_layers
            self.convs.append(GCNConv(data_list[-1].x.size(1), num_hidden_channels))
            self.recurrents.append(LSTM(input_size = num_hidden_channels, hidden_size = num_hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(num_hidden_channels, num_hidden_channels))
                self.recurrents.append(LSTM(input_size = num_hidden_channels, hidden_size = num_hidden_channels))

            for _ in range(num_layers - 1):
                self.batchnorms.append(torch.nn.BatchNorm1d(num_hidden_channels))

        else:            
            self.hidden_channels_str = ''
            for num in hidden_channels:
                self.hidden_channels_str += str(num)+'_'
            self.hidden_channels_str = self.hidden_channels_str[:-1]

            self.num_layers = len(hidden_channels)
            self.convs.append(GCNConv(data_list[-1].x.size(1), hidden_channels[0]))
            self.recurrents.append(LSTM(input_size = hidden_channels[0], hidden_size = hidden_channels[0]))
            for i in range(len(hidden_channels) - 1):
                self.convs.append(GCNConv(hidden_channels[i], hidden_channels[i+1]))
                self.recurrents.append(LSTM(input_size = hidden_channels[i+1], hidden_size = hidden_channels[i+1]))

            for i in range(len(hidden_channels) - 1):
                self.batchnorms.append(torch.nn.BatchNorm1d(hidden_channels[i]))

            if self.decode_modelname in ['VGAE', 'Shifted-VGAE']:
                self.convs.append(GCNConv(hidden_channels[-2], hidden_channels[-1]))
                self.recurrents.append(LSTM(input_size = hidden_channels[-1], hidden_size = hidden_channels[-1]))

        self.activation = activation
        self.dropout = dropout
        self.future_prediction = future_prediction

        if self.future_prediction is True:
            self.feature_recurrents = torch.nn.ModuleList()
            self.feature_recurrents.append(LSTM(input_size = num_hidden_channels, hidden_size = num_hidden_channels))

    def forward(self, x_seq, edge_index_seq):
        '''
        Parameters:
            x (torch.tensor[seq_length, num_nodes, input_channels]): input of the model (node features).
            edge_index (torch.tensor[seq_length, 2, num_edges]): tonsor of the pair of node indexes with edges.

        Returns:
            z (torch.tensor[seq_length, num_nodes, output_channels]): node-wise latent features.
        '''
                
        z_seq = []
        hx_layers = [None]*(len(self.convs)+1)
        for t, x in enumerate(x_seq):
            z = x
            for i, conv in enumerate(self.convs):
                z = F.dropout(z, self.dropout, training = self.training)

                W, (h_, c_) = self.recurrents[i](conv.weight[None, :, :], hx_layers[i])
                hx = (torch.zeros_like(c_).to(self.device), c_)
                hx_layers[i] = hx
                conv.weight = torch.nn.Parameter(W.squeeze())
                z = conv(z, edge_index_seq[t])
                if i < len(self.convs) - 1:
                    z = self.batchnorms[i](z)
                    if self.activation == "relu":
                        z = z.relu()
                    elif self.activation == "leaky_relu":
                        z = F.leaky_relu(z, negative_slope=0.01)
                    elif self.activation == "tanh":
                        z = torch.tanh(z)
            z_seq.append(z)

        if self.future_prediction is True:
            z_seq_tensor = torch.stack(z_seq[:-1],0)
            z , (h_, c_) = self.feature_recurrents[0](z_seq_tensor)
            hx = (torch.zeros_like(c_[-1].unsqueeze(0)).to(self.device), c_[-1].unsqueeze(0))
            z , _ = self.feature_recurrents[0](z[-1].unsqueeze(0), hx)
            z_seq[-1] = z.squeeze()

        return z_seq

class EvolveGCNIIO(torch.nn.Module):       
    '''GCNII

    ノードの特徴量に対してGCNIIを適用するモデル

    Attributes:
        device (:obj:`int`): 'cpu', 'cuda'. 
        lins (torch.nn.ModuleList): 全結合層のリスト.
        convs (torch.nn.ModuleList): torch_geometric.nn.GCNConvのリスト.
        batchnorms (torch.nn.ModuleList): batch normalizationのリスト.
        activation (obj`int` or None): activation functionを指定。None, "relu", "leaky_relu", or "tanh". (Default: None)
        dropout (float): 各層のDropoutの割合. 
        num_layers (int or None): 隠れ層の数.
        hidden_channels_str (str): 各層の出力の次元を文字列として記録
        train_pos_edge_adj_t (torch.SparseTensor[2, num_pos_edges]): trainデータのリンク.
    '''    

    def __init__(self, data_list, decode_modelname, train_pos_edge_adj_t, num_hidden_channels, num_layers, alpha=0.1, theta=0.5, shared_weights = True, activation = None, dropout = 0.0, future_prediction=True):
        '''
        Args:
            data (torch_geometric.data.Data): グラフデータ.
            decode_modelname
            train_pos_edge_adj_t (torch.SparseTensor[2, num_pos_edges]): trainデータのリンク.
            num_hidden_channels (int or None): 隠れ層の出力次元数. 全ての層で同じ値が適用される.
            num_layers (int or None): 隠れ層の数.
            alpha (float): convolution後に初期層を加える割合. (Default: 0.1)
            theta (float): .
            shared_weights (bool): . (Default: True)
            activation (obj`int` or None): activation functionを指定。None, "relu", "leaky_relu", or "tanh". (Default: None)
            dropout (float): 各層のDropoutの割合. (Default: 0.0)
        '''
        super(EvolveGCNIIO, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decode_modelname = decode_modelname

        self.num_layers = num_layers
        self.train_pos_edge_adj_t = train_pos_edge_adj_t
        self.hidden_channels_str = (f'{num_hidden_channels}_'*num_layers)[:-1]

        self.lins = torch.nn.ModuleList()
        self.recurrents = torch.nn.ModuleList()
        self.lins.append(GCNConv(data_list[-1].x.size(1), num_hidden_channels))
        self.recurrents.append(LSTM(input_size = num_hidden_channels, hidden_size = num_hidden_channels))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers - 1):
            self.convs.append(GCN2Conv(num_hidden_channels, alpha, theta, layer+1, shared_weights = shared_weights, normalize = False))
            self.recurrents.append(LSTM(input_size = num_hidden_channels, hidden_size = num_hidden_channels))

        self.batchnorms = torch.nn.ModuleList()
        for layer in range(num_layers - 2):
            self.batchnorms.append(torch.nn.BatchNorm1d(num_hidden_channels))

        self.activation = activation
        self.dropout = dropout    
        self.future_prediction = future_prediction

        if self.future_prediction is True:
            self.feature_recurrents = torch.nn.ModuleList()
            self.feature_recurrents.append(LSTM(input_size = num_hidden_channels, hidden_size = num_hidden_channels))

    def forward(self, x_seq, edge_index_seq):
        '''
        Parameters:
            x (torch.tensor[num_nodes, input_channels]): input of the model (node features).
            edge_index (torch.tensor[2, num_edges]): tonsor of the pair of node indexes with edges.

        Returns:
            z (torch.tensor[num_nodes, output_channels]): node-wise latent features.
        '''

        if self.future_prediction is True:
            x_seq = x_seq[:-1]
            edge_index_seq = edge_index_seq[:-1]

        z_seq = []
        hx_layers = [None]*(len(self.convs)+1)
        for t, x in enumerate(x_seq):
            W, (h_, c_) = self.recurrents[0](self.lins[0].weight[None, :, :], hx_layers[0])
            hx = (torch.zeros_like(c_).to(self.device), c_)
            hx_layers[0] = hx
            self.lins[0].weight1 = torch.nn.Parameter(W.squeeze())
            z = self.lins[0](x, edge_index_seq[t])
            x_0 = z
            for i, conv in enumerate(self.convs):
                z = F.dropout(z, self.dropout, training = self.training)

                W, (h_, c_) = self.recurrents[i+1](conv.weight1[None, :, :], hx_layers[i+1])
                hx = (torch.zeros_like(c_).to(self.device), c_)
                hx_layers[i+1] = hx
                conv.weight1 = torch.nn.Parameter(W.squeeze())
                z = conv(z, x_0, edge_index_seq[t])
                if i < len(self.convs) - 1:
                    z = self.batchnorms[i](z)
                    if self.activation == "relu":
                        z = z.relu()
                    elif self.activation == "leaky_relu":
                        z = F.leaky_relu(z, negative_slope=0.01)
                    elif self.activation == "tanh":
                        z = torch.tanh(z)
            z_seq.append(z)

        if self.future_prediction is True:
            z_seq_tensor = torch.stack(z_seq[:-1],0)
            z , (h_, c_) = self.feature_recurrents[0](z_seq_tensor)
            hx = (torch.zeros_like(c_[-1].unsqueeze(0)).to(self.device), c_[-1].unsqueeze(0))
            z , _ = self.feature_recurrents[0](z[-1].unsqueeze(0), hx)
            z_seq[-1] = z.squeeze()

        return z_seq