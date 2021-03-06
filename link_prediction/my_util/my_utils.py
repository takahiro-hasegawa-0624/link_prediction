"""
utilities for link prediction

    link preiction のための関数

Todo:
    隣接行列を図示する関数を作る
"""

import random
import numpy as np
import pandas as pd

import torch

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges, to_undirected

from torch_sparse import SparseTensor

import networkx as nx

# choice CPU or GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

# fix random variable
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def data_downloader(dataset = 'Cora', data_dir='../data', data_type='static'):
    '''
    グラフデータをダウンロードする.

    Parameters:
        dataset (:obj:`str`): データセット名.'Cora', 'CiteSeer', 'factset'

    Returens:
        data (torch_geometric.data.Data): グラフデータ.
    '''

    if dataset in ['Cora', 'CiteSeer', 'PubMed']:
        data = Planetoid(data_dir, dataset, transform=T.NormalizeFeatures())[0]

    elif 'Factset' in dataset:
        year = dataset[-4:]
        print(f'processing Factset in year {year}.')
        if data_type == 'dynamic':
            df = pd.read_csv(data_dir + f'/Factset/node_features_{year}_dynamic_processed.csv').drop_duplicates(ignore_index=True, subset='code')
        else:
            df = pd.read_csv(data_dir + f'/Factset/node_features_{year}_processed.csv').drop_duplicates(ignore_index=True, subset='code')
        N = len(df) # ノード数

        # sec_codeとノード番号の対応付け
        dic = {}
        for row in df.itertuples(): dic[row[1]] = row[0]

        edge = pd.read_csv(data_dir + f'/Factset/edges_{year}.csv', usecols=['REL_TYPE','SOURCE_COMPANY_TICKER','TARGET_COMPANY_TICKER']).rename(columns={'SOURCE_COMPANY_TICKER': 'source', 'TARGET_COMPANY_TICKER': 'target'})
        edge = edge[(edge['REL_TYPE']=='CUSTOMER') | (edge['REL_TYPE']=='SUPPLIER')]
        edge = edge[['source','target']].drop_duplicates(ignore_index=True, subset=['source', 'target'])

        for i in range(edge.shape[0]):
            if i in edge.index:
                source = edge.loc[i,'source']
                target = edge.loc[i,'target']
                edge = edge.drop(edge[(edge['source']==target) & (edge['target']==source)].index)

        edge = edge.applymap(lambda x: dic[x] if x in dic.keys() else np.nan)
        edge = edge.dropna(how='any').reset_index(drop=True)

        # 欠損値の処理
        df = df.iloc[:, 5:] # sec_codeは除く
        # df = df.dropna(thresh=100, axis=1) # NaNでないデータがthresh個以上なら削除しない
        df = df.fillna(0) # その他の列は平均で補完
        df = (df - df.mean()) / df.std()
        df = df.fillna(0)

        # X to tensor
        X = [[] for _ in range(N)]
        for row in df.itertuples(): X[row[0]] = row[1:]
        X = torch.tensor(X, dtype=torch.float)

        # edge_index to tensor
        edge_index = torch.tensor(edge.to_numpy().T, dtype=torch.long)

        # torch_geometric.data.Data 
        data = Data(x=X, edge_index=edge_index)

    print(f'dataset {dataset} has been downloaded.')
    print(f'is undirected: {data.is_undirected()}')
    print(f'contains self loops: {data.contains_self_loops()}')
    print(f'num_nodes: {data.num_nodes}')
    print(f'num_edges: {data.num_edges}\n')

    if data.is_undirected() is False:
        data.edge_index = to_undirected(data.edge_index)
        print('The graph has been transformed into undirected one.')

    return data

def data_processor(data, undirected=True, val_ratio=0.05, test_ratio=0.1):
    '''
    グラフデータをPytorch Geometric用に処理する.

    Parameters:
        data (torch_geometric.data.Data): グラフデータ.

    Returens:
        all_pos_edge_index (torch.Tensor[2, num_pos_edges]): train_test_split前の全リンク.
        train_pos_edge_adj_t (torch.SparseTensor[2, num_pos_edges]): trainデータのリンク.
        y_true (numpy.ndarray[num_nodes, num_nodes].flatten()): 全リンクの隣接行列をflattenしたもの.
        y_train (numpy.ndarray[num_nodes, num_nodes].flatten()): trainデータのリンクの隣接行列をflattenしたもの.
        mask (numpy.ndarray[num_nodes, num_nodes].flatten()): validation, testのpos_edge, neg_edgeとしてサンプリングしたリンクをFalse、それ以外をTrueとした隣接行列をflattenしたもの.
    '''

    # train_test_splitをする前に、エッジのTensorをコピーしておく
    all_pos_edge_index = data.edge_index

    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = train_test_split_edges(data, val_ratio=val_ratio, test_ratio=test_ratio)

    if (val_ratio + test_ratio ==1) and (data.train_pos_edge_index.size(1) > 0):
        data.test_pos_edge_index = torch.cat([data.test_pos_edge_index, data.train_pos_edge_index], dim=-1)
        data.train_pos_edge_index = torch.LongTensor([[],[]])

    print('train test split has been done.')
    print(data)
    print('')

    # GCN2Convに渡すエッジのSparseTensorを作成する
    # 参考: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/to_sparse_tensor.html#ToSparseTensor
    # edge_index全てではなく、train_pos_edge_indexに抽出されたエッジのみを変換する点に注意
    (row, col), N, E = data.train_pos_edge_index, data.num_nodes, data.train_pos_edge_index.size(1)
    perm = (col * N + row).argsort()
    row, col = row[perm], col[perm]

    value = None
    for key in ['edge_weight', 'edge_attr', 'edge_type']:
        if data[key] is not None:
            value = data[key][perm]
            break

    for key, item in data:
        if item.size(0) == E:
            data[key] = item[perm]

    train_pos_edge_adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N), is_sorted=True)

    print('train_pos_edge_adj_t is completed.\n')

    # 1. 全エッジ
    edge = pd.DataFrame(all_pos_edge_index.cpu().numpy().T, columns=['source', 'target'])
    G = nx.from_pandas_edgelist(edge, create_using=nx.Graph())

    #隣接行列を作成
    df_adj = pd.DataFrame(np.zeros([len(G.nodes()),len(G.nodes())]),index=G.nodes(),columns=G.nodes()).sort_index(axis=0).sort_index(axis=1)
    for i,j in G.edges(): 
        df_adj.loc[i,j] = 1
        
    y_true = torch.tensor(df_adj.to_numpy().flatten(), dtype=torch.float)

    print('y_true is completed.\n')

    # 2. trainに用いるエッジ
    edge = pd.DataFrame(data.train_pos_edge_index.cpu().numpy().T, columns=['source', 'target'])
    G_train = nx.from_pandas_edgelist(edge, create_using=nx.Graph())

    #隣接行列を作成
    df_adj_train = pd.DataFrame(np.zeros([len(G.nodes()),len(G.nodes())]),index=G.nodes(),columns=G.nodes()).sort_index(axis=0).sort_index(axis=1)
    for i,j in G_train.edges(): 
        df_adj_train.loc[i,j] = 1
        
    y_train = torch.tensor(df_adj_train.to_numpy().flatten(), dtype=torch.float)

    print('y_train is completed.\n')

    # 隣接行列が0の部分には、validation、testで用いるpositive、negativeのエッジが含まれる。これらのエッジをlossの計算から除くためのmaskを作成
    val_test_edge = torch.cat([data.test_neg_edge_index, data.test_pos_edge_index, data.val_neg_edge_index, data.val_pos_edge_index], dim=-1)
    mask = torch.ones([data.x.size(0), data.x.size(0)], dtype=torch.float)

    for i in range(val_test_edge.size(1)):
        mask[val_test_edge[0,i], val_test_edge[1,i]] = 0
        mask[val_test_edge[1,i], val_test_edge[0,i]] = 0

    mask = mask.flatten()

    return all_pos_edge_index, train_pos_edge_adj_t, y_true, y_train, mask

def get_link_labels(pos_edge_index, neg_edge_index):
    '''
    ノードの組の正解・不正解のラベルを持つ1階のTensorを返す.
    正解のノードに対応するindexのvalueが1、不正解のノードに対応するindexのvalueが0.
    binary cross entropy の正解ラベルとして使う.

    Parameters:
        pos_edge_index (torch.Tensor[2, num_pos_edges]): 実際にエッジが存在するノードの組
        neg_edge_index (torch.Tensor[2, num_neg_edges]): 実際にはエッジが存在しないノードの組

    Returns:
        link_labels (torch.Tensors[num_pos_edges+num_neg_edges]): pos_edge, neg_edgeの順にconcatされた正解ラベルのTensor
    '''

    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    
    return link_labels

def get_loss_weight(pos_edge_index, neg_edge_index, negative_sampling_ratio):
    '''
    ノードの組の正解・不正解に対するlossの重みを持つ1階のTensorを返す.
    不正解のノードに対応するindexのweightを1としたときの、正解のノードに対応するindexのweightを求める.
    binary cross entropy のweightとして使う.

    Parameters:
        pos_edge_index (torch.Tensor[2, num_pos_edges]): 実際にエッジが存在するノードの組
        neg_edge_index (torch.Tensor[2, num_neg_edges]): 実際にはエッジが存在しないノードの組
        negative_sampling_ratio (float or None): 正例に対する負例のサンプリング比率

    Returns:
        loss_weight (torch.Tensors[num_pos_edges+num_neg_edges]): pos_edge, neg_edgeの順にconcatされた、negative_edgeのlossに対するpositive_edgeのlossのweight
    '''

    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    loss_weight = torch.ones(E, dtype=torch.float)
    loss_weight[:pos_edge_index.size(1)] = float(negative_sampling_ratio)
    
    return loss_weight
    
def make_confusion_matrix(matrix, column_labels):
    # matrix numpy配列

    # columns 項目名リスト
    n = len(column_labels)

    act = ['y_true'] * n
    pred = ['y_pred'] * n

    df_confusion_matrix = pd.DataFrame(matrix, columns=[pred, column_labels], index=[act, column_labels])
    return df_confusion_matrix