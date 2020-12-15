import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class TPSE(nn.Module):
    def __init__(self, training, batch_size, input_size, rnn_size, hidden_size, output_size):
        super().__init__()
        
        self.rnn = nn.GRU(input_size, rnn_size, batch_first=True)
        self.fc1 = nn.Linear(rnn_size, hidden_size)
        
    def forward(self, x):
        self.rnn.flatten_parameters()
        _, x = self.rnn(x)
        x = x.squeeze(0)
        x = self.fc1(x)
        x = x.tanh()
        x = x.repeat(1, 4) # Resize for multi-headed attention 
        x = x.unsqueeze(1)

        return x
        

class GST(nn.Module):

    def __init__(self, refencoder, styletoken):

        super().__init__()
        self.encoder = ReferenceEncoder(**refencoder)
        self.styletoken = StyleTokenLayer(**styletoken)

    def forward(self, x):
        x = self.encoder(x)
        x = self.styletoken(x)

        return x


class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self, output_channels, kernel_size, stride, padding, embedding_dim, n_mels):

        super().__init__()
        self.n_mels = n_mels
        output_channel_count = len(output_channels) # output_channel_count = K
        filters = [1] + output_channels
        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding) for i in range(output_channel_count)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=output_channels[i]) for i in range(output_channel_count)])

        channel_2d_size = self.calculate_channels(self.n_mels, 3, 2, 1, output_channel_count)
        self.gru = nn.GRU(input_size=output_channels[-1] * channel_2d_size,
                          hidden_size=embedding_dim // 2,
                          batch_first=True)

    def forward(self, x):
        B = x.size(0)
        # x = x.view(B, 1, -1, self.n_mels)  # [B, 1, Ty, n_mels]
        x = x.unsqueeze(1) # [B, 1, N, T]
        x = x.transpose(2, 3) # [B, 1, T, N]
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)  # [B, 128, T//2^K, N//2^K]

        x = x.transpose(1, 2)  # [B, T//2^K, 128, N//2^K]
        B, T = x.size(0), x.size(1)
        x = x.reshape(B, T, -1)  # [B, T//2^K, 128*N//2^K]

        self.gru.flatten_parameters()
        _, x = self.gru(x)  # out --- [1, N, E//2]

        return x.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class StyleTokenLayer(nn.Module):
    '''
    inputs --- [N, E//2]
    '''

    def __init__(self, embedding_dim, heads, tokens):

        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(tokens, embedding_dim // heads))
        d_q = embedding_dim // 2
        d_k = embedding_dim // heads
        # self.attention = MultiHeadAttention(heads, d_model, d_q, d_v)
        self.attention = MultiHeadAttention(query_dim=d_q, key_dim=d_k, num_units=embedding_dim, num_heads=heads)

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, x):
        N = x.size(0)
        x = x.unsqueeze(1)  # [N, 1, E//2]
        keys = self.embed.tanh().unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]
        x = self.attention(x, keys)

        return x


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, x, key):
        x = self.W_query(x)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        x = torch.stack(torch.split(x, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        x = torch.matmul(x, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        x = x / (self.key_dim ** 0.5)
        x = F.softmax(x, dim=3)

        # out = score * V
        x = torch.matmul(x, values)  # [h, N, T_q, num_units/h]
        x = torch.cat(torch.split(x, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return x
