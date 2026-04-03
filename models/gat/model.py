import os
import tinygrad
from tinygrad import Tensor,nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import kneighbors_graph 
from tinygrad import TinyJit
from tinygrad.nn.optim import AdamW

#loadings
data = np.load("/Users/vihantiwari/Documents/projects/gram_comp/data/1023_24-4.npz")
pos = data["pos"]
idcs_airfoil = data["idcs_airfoil"]
velocity_in = data["velocity_in"] #(5,100000,3) -> (timestep,point,vx,vy,vz)
velocity_out = data["velocity_out"]#(5,100000,3) -> (timestep,point,vx,vy,vz)
wing_points = pos[idcs_airfoil]

#construct the graph first using knn 
num_features = velocity_in 
knn = kneighbors_graph(pos,n_neighbors=5,mode="connectivity",metric="euclidean")
rows, cols = knn.nonzero()
edge_index = Tensor(np.array(knn.nonzero()),dtype="int64")
#node features = velocity at specific point, time  (100000,5x3=15) -> (points,timesteps*features)
node_features = velocity_in.transpose(1,0,2).reshape(100000,15)
print(node_features)
target_features = velocity_out.transpose(1,0,2).reshape(100000,15)


hidden_dim = 64
num_classes = 0

class GATLayers:
    def __init__(self,node_features,target_features,num_heads,concat,dropout=0.6):
        self.concat = concat
        self.num_heads = num_heads
        if concat:
            assert target_features % num_heads==0
            self.num_hidden = target_features//num_heads
        else:
            self.num_hidden = target_features

        self.linear = nn.Linear(node_features,self.num_heads * self.num_hidden,bias=False)
        self.attention = nn.Linear(self.num_hidden*2,1,bias=False)

    def __call__(self,h,adj_matrix):
        num_nodes = pos.shape[0]
        h = h.dropout(p=0.6)
        transform = self.linear(h).view(num_nodes,self.num_heads,self.num_hidden)
        transform_rows = transform.repeat_interleave(num_nodes,dim=0) 
        transform_cols = transform.repeat(num_nodes,dim=0)
        transform_final = transform_rows.cat(transform_cols,dim=1)
        transform_reshaped = transform_final.view(num_nodes,num_nodes,self.num_heads,2*self.num_hidden)
        raw_attention = self.attention(transform_reshaped)
        e_ij = raw_attention.leaky_relu(0.2)
        e_ij = e_ij.squeeze(-1)
        #masking
        assert adj_matrix.shape[0] == 1 or adj_matrix.shape[0] == num_nodes
        assert adj_matrix.shape[1] == 1 or adj_matrix.shape[1] == num_nodes
        assert adj_matrix.shape[2] == 1 or adj_matrix.shape[2] == self.num_heads
        e_ij = e_ij.masked_fill(adj_matrix==0,float("-1e9"))
        alpha_ij = e_ij.softmax(axis=1)
        alpha_ij = alpha_ij.dropout(p=0.6)
        att_final = Tensor.einsum("ijh,jhf->ihf",alpha_ij,transform)
        if self.concat:
            return att_final.reshape(num_nodes,self.num_heads*self.num_hidden)
        else:
            return att_final.sum(axis=1)

class GATModel:
    def __init__(self,node_features,hidden_features,target_features,heads=8,dropout=0.6):
        self.dropout = dropout
        self.layer1 = GATLayers(node_features,hidden_features,heads,concat=True)
        self.layer2 = GATLayers(hidden_features,target_features,heads,concat=False)

    def __call__(self,x,adj_matrix):
        x = self.layer1(x,adj_matrix)
        x = x.elu()
        return self.layer2(x,adj_matrix)

model = GATModel(num_features,hidden_dim,num_features,heads=8,dropout=0.6)

Tensor.training = True
params = nn.state.get_parameters(model)
optimizer = AdamW(params,lr=0.01)
