import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer


class GAT_with_LSTM(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, lstm_hid=6):
        """Dense version of GAT."""
        super(GAT_with_LSTM, self).__init__()
        self.dropout = dropout
        # LSTM to process time series
        self.lstm = nn.LSTM(2, 2 * lstm_hid, num_layers=2, batch_first=True)
        # multi-head attention setting
        self.attentions = [GraphAttentionLayer(nfeat * lstm_hid, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        # final attention layer use averaging instead of concatenation
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj, return_final_att=False):
        lstm_output, _ = self.lstm(x)   # input: (batch, seq_len, input_size) output: (batch, seq_len, hidden_size)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(lstm_output.flatten(start_dim=1), adj) for att in self.attentions], dim=1)      # size (N_samples, n_hid * n_heads)
        # x = F.dropout(x, self.dropout, training=self.training)
        if not return_final_att:
            x = F.elu(self.out_att(x, adj))
            return F.log_softmax(x, dim=1)  # size (N_samples, n_class)
        else:
            out, att = self.out_att(x, adj, return_att=True)
            x = F.elu(out)
            return F.log_softmax(x, dim=1), att


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        # multi-head attention setting
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        # final attention layer use averaging instead of concatenation
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj, return_final_att=False):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)      # size (N_samples, n_hid * n_heads)
        x = F.dropout(x, self.dropout, training=self.training)
        if not return_final_att:
            x = F.elu(self.out_att(x, adj))
            return F.log_softmax(x, dim=1)  # size (N_samples, n_class)
        else:
            out, att = self.out_att(x, adj, return_att=True)
            x = F.elu(out)
            return F.log_softmax(x, dim=1), att



class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


# if __name__ == '__main__':
#     # 6 samples
#     input = torch.randn(6, 10)
#     adj = torch.zeros((6, 6), dtype=torch.int8)
#     for i in range(6):
#         for j in range(6):
#             if torch.randn(1)[0] > 0.2:
#                 adj[i][j] = 1
#     gat = GAT(10, 8, 5, dropout=0.6, alpha=0.2, nheads=3)
#     out, att = gat(input, adj, return_final_att=True)
#     print(att)
