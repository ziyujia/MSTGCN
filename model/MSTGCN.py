import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Model of MSTGCN.
--------
Model input:  (*, T, V, F)
    T: num_of_timesteps
    V: num_of_vertices
    F: num_of_features
Model output: (*, C)
    C: num_of_classes
'''

################################################################################################
################################################################################################
# Attention Layers

class TemporalAttention(nn.Module):
    '''
    compute temporal attention scores
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_timesteps, num_of_timesteps)
    '''
    def __init__(self, num_of_timesteps, num_of_vertices, num_of_features):
        super(TemporalAttention, self).__init__()
        
        self.U_1 = nn.Parameter(torch.empty(num_of_vertices, 1))
        nn.init.kaiming_uniform_(self.U_1)
        
        self.U_2 = nn.Parameter(torch.empty(num_of_features, num_of_vertices))
        nn.init.kaiming_uniform_(self.U_2)
        
        self.U_3 = nn.Parameter(torch.empty(num_of_features,))
        nn.init.uniform_(self.U_3)
        
        self.b_e = nn.Parameter(torch.empty(1, num_of_timesteps, num_of_timesteps))
        nn.init.kaiming_uniform_(self.b_e)
        
        self.V_e = nn.Parameter(torch.empty(num_of_timesteps, num_of_timesteps))
        nn.init.kaiming_uniform_(self.V_e)
        
    def forward(self, x):
        # N, T, V, F = x.shape
        
        # shape of lhs is (batch_size, T, V)
        lhs = torch.matmul(x.permute([0,1,3,2]), self.U_1).squeeze(-1)
        lhs = torch.matmul(lhs, self.U_2)
        
        # shape of rhs is (batch_size, V, T)
        rhs = torch.matmul(self.U_3, x.permute([2,0,3,1]))
        rhs = rhs.permute([1,0,2])
        
        # shape of product is (batch_size, T, T)
        product = torch.matmul(lhs, rhs)
        
        S = torch.permute(torch.matmul(self.V_e, torch.permute(torch.sigmoid(product + self.b_e), [1,2,0])), [2,0,1])
        
        # normalize
        S = S - torch.max(S, dim=1, keepdim=True)[0]
        S_exp = torch.exp(S)
        S_normalized = S_exp / torch.sum(S_exp, dim=1, keepdim=True)
        return S_normalized
    

class SpatialAttention(nn.Module):
    '''
    compute spatial attention scores
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_vertices, num_of_vertices)
    '''
    def __init__(self, num_of_timesteps, num_of_vertices, num_of_features):
        super(SpatialAttention, self).__init__()
        
        self.W_1 = nn.Parameter(torch.empty(num_of_timesteps, 1))
        nn.init.kaiming_uniform_(self.W_1)
        
        self.W_2 = nn.Parameter(torch.empty(num_of_features, num_of_timesteps))
        nn.init.kaiming_uniform_(self.W_2)
        
        self.W_3 = nn.Parameter(torch.empty(num_of_features,))
        nn.init.uniform_(self.W_3)
        
        self.b_s = nn.Parameter(torch.empty(1, num_of_vertices, num_of_vertices))
        nn.init.kaiming_uniform_(self.b_s)
        
        self.V_s = nn.Parameter(torch.empty(num_of_vertices, num_of_vertices))
        nn.init.kaiming_uniform_(self.V_s)
        
    def forward(self, x):
        # N, T, V, F = x.shape
        
        # shape of lhs is (batch_size, V, T)
        lhs = torch.matmul(x.permute([0,2,3,1]), self.W_1).squeeze(-1)
        lhs = torch.matmul(lhs, self.W_2)
        
        # shape of rhs is (batch_size, T, V)
        rhs = torch.matmul(self.W_3, x.permute([1,0,3,2]))
        rhs = rhs.permute([1,0,2])
        
        # shape of product is (batch_size, V, V)
        product = torch.matmul(lhs, rhs)
        
        S = torch.permute(torch.matmul(self.V_s, torch.permute(torch.sigmoid(product + self.b_s), [1,2,0])), [2,0,1])
        
        # normalize
        S = S - torch.max(S, dim=1, keepdim=True)[0]
        S_exp = torch.exp(S)
        S_normalized = S_exp / torch.sum(S_exp, dim=1, keepdim=True)
        return S_normalized
    
    
################################################################################################
################################################################################################
# Adaptive Graph Learning Layer

def diff_loss(diff, S, Falpha):
    '''
    compute the 1st loss of L_{graph_learning}
    '''
    if len(S.shape) == 4:
        # batch input
        return torch.mean(torch.sum(torch.sum(diff**2, axis=3) * S, axis=(1,2))) *Falpha
    else:
        # single input
        return torch.sum(torch.sum(diff**2, axis=2) * S) *Falpha
    
    
def F_norm_loss(S, Falpha):
    '''
    compute the 2nd loss of L_{graph_learning}
    '''
    if len(S.shape) == 4:
        # batch input
        return Falpha * torch.sum(torch.mean(S**2, axis=0))
    else:
        # single input
        return Falpha * torch.sum(S**2)
    
    
class Graph_Learn(nn.Module):
    '''
    Graph structure learning (based on the middle time slice)
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_vertices, num_of_vertices)
    '''
    def __init__(self, num_of_features, alpha):
        super(Graph_Learn, self).__init__()
        self.alpha = alpha
        
        self.num_of_features = num_of_features
        self.a = nn.Parameter(torch.empty(num_of_features, 1))
        nn.init.kaiming_uniform_(self.a)
        
    def forward(self, x):
        N, T, V, F = x.shape
        
        outputs = torch.zeros((N, T, V, V)).cuda()
        diff_tmp = torch.zeros((N, V, V, F)).cuda()
        
        for time_step in range(T):
            # xt-shape:   (N,V,F) use the current slice
            xt = x[:,time_step,:,:]
            # diff-shape: (N,V,V,F)
            diff = torch.permute(torch.broadcast_to(xt, (V, N, V, F)).permute([2,1,0,3]) - xt, [1,0,2,3])
            # tmpS-shape: (N,V,V)
            tmpS = torch.exp(torch.matmul(torch.abs(diff), self.a)/self.num_of_features).squeeze(-1)
            # normalize
            S = tmpS / (torch.sum(tmpS, dim=1, keepdim=True)+1e-8)
            
            diff_tmp += torch.abs(diff)
            outputs[:,time_step,:,:] = S
            
        # compute extra losses
        S_mean = torch.mean(outputs, axis=0)
        diff_mean = torch.mean(diff_tmp, axis=0) / T
        loss1 = diff_loss(diff_mean, S_mean, self.alpha)
        loss2 = F_norm_loss(S_mean, self.alpha)
            
        return outputs, loss1, loss2
    
    
################################################################################################
################################################################################################
# GCN layers

class cheb_conv_with_Att_GL(nn.Module):
    '''
    K-order chebyshev graph convolution with attention after Graph Learn
    --------
    Input:  [x   (batch_size, num_of_timesteps, num_of_vertices, num_of_features),
             Att (batch_size, num_of_vertices, num_of_vertices),
             S   (batch_size, num_of_vertices, num_of_vertices)]
    Output: (batch_size, num_of_timesteps, num_of_vertices, num_of_filters)
    '''
    def __init__(self, input_filters, num_of_filters, k):
        super(cheb_conv_with_Att_GL, self).__init__()
        self.k = k
        self.num_of_filters = num_of_filters
        
        self.Theta = nn.Parameter(torch.empty(self.k, input_filters, num_of_filters))
        nn.init.kaiming_uniform_(self.Theta)
        
    def forward(self, x, Att, S):
        N, T, V, F = x.shape
        
        S = torch.minimum(S, torch.permute(S, [0,1,3,2]))
        
        # GCN
        outputs = torch.zeros((N, T, V, self.num_of_filters)).cuda()
        for time_step in range(T):
            # graph_signal-shape: (N,V,F)
            graph_signal = x[:,time_step,:,:]
            
            A = S[:,time_step,:,:]
            #Calculating Chebyshev polynomials (let lambda_max=2)
            D = torch.diag_embed(torch.sum(A, dim=1))
            L = D - A
            L_t = L - torch.eye(V).cuda()
            cheb_polynomials = [torch.eye(V).cuda(), L_t]
            for i in range(2, self.k):
                cheb_polynomials.append(2 * L_t * cheb_polynomials[-1] - cheb_polynomials[-2])

            for kk in range(self.k):
                T_k = cheb_polynomials[kk]      # shape of T_k is (V, V)
                T_k_with_att = T_k * Att        # shape of T_k_with_att is (N, V, V)
                theta_k = self.Theta[kk,:,:]    # shape of theta_k is (F, num_of_filters)
                
                # shape is (N, V, F)
                rhs = torch.matmul(torch.permute(T_k_with_att, [0,2,1]), graph_signal)
                outputs[:,time_step,:,:] += torch.matmul(rhs, theta_k)
            
        return torch.relu(outputs)


class cheb_conv_with_Att_static(nn.Module):
    '''
    K-order chebyshev graph convolution with static graph structure
    --------
    Input:  [x   (batch_size, num_of_timesteps, num_of_vertices, num_of_features),
             Att (batch_size, num_of_vertices, num_of_vertices)]
    Output: (batch_size, num_of_timesteps, num_of_vertices, num_of_filters)
    '''
    def __init__(self, input_filters, num_of_filters, k, cheb_polynomials):
        super(cheb_conv_with_Att_static, self).__init__()
        self.k = k
        self.num_of_filters = num_of_filters
        self.cheb_polynomials = torch.FloatTensor(cheb_polynomials).cuda()
        
        self.Theta = nn.Parameter(torch.empty(self.k, input_filters, num_of_filters))
        nn.init.kaiming_uniform_(self.Theta)
        
        self.drop = nn.Dropout(0.6)
        
    def forward(self, x, Att):
        N, T, V, F = x.shape
        
        # GCN
        outputs = torch.zeros((N, T, V, self.num_of_filters)).cuda()
        for time_step in range(T):
            # graph_signal-shape: (N,V,F)
            graph_signal = x[:,time_step,:,:]

            for kk in range(self.k):
                T_k = self.cheb_polynomials[kk]               # shape of T_k is (V, V)
                T_k_with_att = self.drop(T_k * Att)  # shape of T_k_with_att is (N, V, V)
                theta_k = self.Theta[kk,:,:]                  # shape of theta_k is (F, num_of_filters)
                
                # shape is (N, V, F)
                rhs = torch.matmul(torch.permute(T_k_with_att, [0,2,1]), graph_signal)
                outputs[:,time_step,:,:] += torch.matmul(rhs, theta_k)
        
        return torch.relu(outputs)


################################################################################################
################################################################################################
# Gradient Reverse Layer

class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, hp_lambda):
        ctx.hp_lambda = hp_lambda
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.hp_lambda
        return grad_input, None


class GradientReverseLayer(nn.Module):
    def __init__(self, hp_lambda=1.0):
        super(GradientReverseLayer, self).__init__()
        self.hp_lambda = hp_lambda

    def forward(self, x):
        return GradientReverseFunction.apply(x, self.hp_lambda)


################################################################################################
################################################################################################
# MSTGCN Block

class MSTGCN_block(nn.Module):
    '''
    packaged Spatial-temporal convolution Block
    -------
    x: input data;
    k: k-order cheb GCN
    i: block number
    '''
    def __init__(self, num_of_timesteps, num_of_vertices, num_of_features, 
                 k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                 cheb_polynomials, time_conv_kernel, GLalpha, i=0):
        super(MSTGCN_block, self).__init__()
        
        # attention layers
        self.temporal_Att = TemporalAttention(num_of_timesteps, num_of_vertices, num_of_features)
        self.spatial_Att  = SpatialAttention(num_of_timesteps, num_of_vertices, num_of_features)
        
        # multi-view GCN
        self.graph_learn = Graph_Learn(num_of_features, GLalpha)
        self.graph_drop = nn.Dropout(0.3)
        self.gcn_GL = cheb_conv_with_Att_GL(num_of_features, num_of_chev_filters, k)
        self.gcn_SD  = cheb_conv_with_Att_static(num_of_features, num_of_chev_filters, k, cheb_polynomials)
        
        # temporal convolution
        self.cnn_GL = nn.Conv2d(in_channels=num_of_chev_filters, out_channels=num_of_time_filters,
                                kernel_size=(time_conv_kernel, 1), stride=(time_conv_strides, 1), padding='same')
        self.cnn_SD = nn.Conv2d(in_channels=num_of_chev_filters, out_channels=num_of_time_filters,
                                kernel_size=(time_conv_kernel, 1), stride=(time_conv_strides, 1), padding='same')
        
        # layer norm
        self.norm_GL = nn.LayerNorm([num_of_time_filters], elementwise_affine=False) 
        self.norm_SD = nn.LayerNorm([num_of_time_filters], elementwise_affine=False)
        
    def forward(self, x):
        # x: (N, T, V, F)
        N, T, V, F = x.shape
        
        # attention
        TAtt = self.temporal_Att(x)
        x_TAtt = torch.bmm(x.permute([0,2,3,1]).reshape(N, V*F, T), TAtt)
        x_TAtt = x_TAtt.reshape(N, V, F, T).permute([0,3,1,2])
        SAtt = self.spatial_Att(x_TAtt)
        
        # GCN
        S, loss1, loss2 = self.graph_learn(x)
        S = self.graph_drop(S)
        x_GL = self.gcn_GL(x, SAtt, S) # (N, T, V, F')
        x_SD = self.gcn_SD(x, SAtt)    # (N, T, V, F')
        
        # temporal convolution
        x_GL = self.cnn_GL(x_GL.permute([0,3,1,2])).permute([0,2,3,1])
        x_SD = self.cnn_SD(x_SD.permute([0,3,1,2])).permute([0,2,3,1])
        
        # layer norm
        x_GL = self.norm_GL(torch.relu(x_GL))
        x_SD = self.norm_SD(torch.relu(x_SD))
        
        return x_GL, x_SD, loss1, loss2
    
    
################################################################################################
################################################################################################
# MSTGCN

class MSTGCN(nn.Module):
    # Input:  (*, num_of_timesteps, num_of_vertices, num_of_features)
    def __init__(self, num_of_timesteps, num_of_vertices, num_of_features, 
                 k, num_of_chev_filters, num_of_time_filters, time_conv_strides, cheb_polynomials,
                 time_conv_kernel, num_block, dense_size, GLalpha,
                 dropout, lambda_reversal, num_classes=5, num_domain=9):
        super(MSTGCN, self).__init__()
        
        self.num_block = num_block
        self.block1 = MSTGCN_block(num_of_timesteps, num_of_vertices, num_of_features, 
                                   k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                                   cheb_polynomials, time_conv_kernel, GLalpha, 1)
        if num_block>1:
            self.blocks = nn.ModuleList(
                [MSTGCN_block(num_of_timesteps, num_of_vertices, num_of_time_filters*2, 
                              k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                              cheb_polynomials, time_conv_kernel, GLalpha, i)
                 for i in range(num_block-1)])
        
        self.dropout = dropout
        if dropout != 0:
            self.drop = nn.Dropout(dropout)
            
        if type(dense_size) == int:
            # single layer
            self.dense_class = nn.Sequential(
                nn.Linear(2*num_of_time_filters*num_of_timesteps*num_of_vertices, dense_size),
                nn.Linear(dense_size, num_classes))
            self.dense_domain = nn.Sequential(
                nn.Linear(2*num_of_time_filters*num_of_timesteps*num_of_vertices, dense_size),
                nn.Linear(dense_size, num_domain))
        elif len(dense_size) == 1:
            # single layer
            self.dense_class = nn.Sequential(
                nn.Linear(2*num_of_time_filters*num_of_timesteps*num_of_vertices, dense_size[0]),
                nn.Linear(dense_size[0], num_classes))
            self.dense_domain = nn.Sequential(
                nn.Linear(2*num_of_time_filters*num_of_timesteps*num_of_vertices, dense_size[0]),
                nn.Linear(dense_size[0], num_domain))
        elif len(dense_size) > 1:
            # multiple layers
            self.dense_class = nn.Sequential(
                nn.Linear(2*num_of_time_filters*num_of_timesteps*num_of_vertices, dense_size[0]))
            for i in range(1, len(dense_size)):
                self.dense_class.append(nn.Linear(dense_size[i-1], dense_size[i]))
            self.dense_class.append(nn.Linear(dense_size[-1], num_classes))
            self.dense_domain = nn.Sequential(
                nn.Linear(2*num_of_time_filters*num_of_timesteps*num_of_vertices, dense_size[0]))
            for i in range(1, len(dense_size)):
                self.dense_domain.append(nn.Linear(dense_size[i-1], dense_size[i]))
            self.dense_domain.append(nn.Linear(dense_size[-1], num_domain))
                
        self.GRL = GradientReverseLayer(lambda_reversal)
        
    def forward(self, x):
        x_GL, x_SD, loss1, loss2 = self.block1(x)
        block_out = torch.cat((x_GL, x_SD), dim=-1)
        if self.num_block>1:
            for block in self.blocks:
                x_GL, x_SD, loss1_tmp, loss2_tmp = block(block_out)
                block_out = torch.cat((x_GL, x_SD), dim=-1)
                loss1 += loss1_tmp
                loss2 += loss2_tmp
        block_out = torch.flatten(block_out, start_dim=1)
        
        if self.dropout != 0:
            block_out = self.drop(block_out)
            
        # Global dense layer
        class_out = self.dense_class(block_out)
        
        # GRL & G_d
        block_out_flip = self.GRL(block_out)
        domain_out = self.dense_domain(block_out_flip)  
        
        return class_out, domain_out, loss1, loss2
    
    
    def predict_class(self, x):
        # only predict stage class
        x_GL, x_SD, loss1, loss2 = self.block1(x)
        block_out = torch.cat((x_GL, x_SD), dim=-1)
        if self.num_block>1:
            for block in self.blocks:
                x_GL, x_SD, loss1_tmp, loss2_tmp = block(block_out)
                block_out = torch.cat((x_GL, x_SD), dim=-1)
                loss1 += loss1_tmp
                loss2 += loss2_tmp
        block_out = torch.flatten(block_out, start_dim=1)
        if self.dropout != 0:
            block_out = self.drop(block_out)
        class_out = self.dense_class(block_out)
        return class_out


################################################################################################
# MSTGCN test

def build_MSTGCN_test():
    import numpy as np
    import torchinfo
    
    # an example to test
    fold = 10
    lambda_GRL = 0.1
    context = 5
    dropout = 0.6
    GLalpha = 0.0001
    num_block = 1
    channels = 26
    cheb_k = 3
    num_of_chev_filters = 10
    num_of_time_filters = 11
    time_conv_strides = 1
    time_conv_kernel = 3
    dense_size = np.array([64, 32])
    cheb_poly_DC = np.random.rand(cheb_k, channels, channels)

    model = MSTGCN(context, channels, 64, cheb_k, num_of_chev_filters, num_of_time_filters,
                   time_conv_strides, cheb_poly_DC, time_conv_kernel, num_block, dense_size, 
                   GLalpha,  dropout, lambda_GRL, num_classes=5, num_domain=fold-1)
    torchinfo.summary(model, input_size=(4, context, channels, 64))

if __name__ == '__main__':
    build_MSTGCN_test()
