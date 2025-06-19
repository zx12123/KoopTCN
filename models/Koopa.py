# import math
# import torch
# import torch.nn as nn
#
#
# # ZX：傅里叶滤波器
# class FourierFilter(nn.Module):
#     """
#     Fourier Filter: to time-variant and time-invariant term
#     """
#
#     def __init__(self, mask_spectrum):
#         super(FourierFilter, self).__init__()
#         self.mask_spectrum = mask_spectrum
#
#     # ZX：forward方法：对输入数据进行傅里叶变换，应用掩膜，得到时间变化的成分和时间不变的成分。
#     def forward(self, x):
#         xf = torch.fft.rfft(x, dim=1)#傅里叶变换
#         mask = torch.ones_like(xf)
#         mask[:, self.mask_spectrum, :] = 0
#         x_var = torch.fft.irfft(xf * mask, dim=1)#傅里叶逆变换
#         x_inv = x - x_var
#
#         return x_var, x_inv
#
#
# # ZX:多层感知机（MLP）目的：实现一个多层感知机，用于编码和解码数据的高维表示
# class MLP(nn.Module):
#     '''
#     Multilayer perceptron to encode/decode high dimension representation of sequential data
#     '''
#
#     # 主要参数：输入和输出特征维度、隐藏层参数、激活函数等。
#     def __init__(self,
#                  f_in,  # 输入特征的维度
#                  f_out,  # 输出特征的维度。
#                  hidden_dim=128,  # 每个隐藏层中神经元的数量
#                  hidden_layers=2,  # 隐藏层的数量
#                  dropout=0.05,  # 在训练过程中随机丢弃节点的比例。每次训练中大约 5% 的节点会被随机丢弃。
#                  activation='tanh'):  # tanh 是双曲正切激活函数，能够引入非线性特性，从而使模型更能够捕捉复杂的模式。
#
#         super(MLP, self).__init__()
#         self.f_in = f_in
#         self.f_out = f_out
#         self.hidden_dim = hidden_dim
#         self.hidden_layers = hidden_layers
#         self.dropout = dropout
#         # 将构造函数参数传给实例变量，以便在整个 MLP 类中使用。
#
#         if activation == 'relu':
#             self.activation = nn.ReLU()
#         elif activation == 'tanh':
#             self.activation = nn.Tanh()
#         else:
#             raise NotImplementedError
#         # 根据传入的activation参数，选择合适的激活函数
#
#         # 构建网络层：
#         layers = [nn.Linear(self.f_in, self.hidden_dim),
#                   self.activation, nn.Dropout(self.dropout)]
#         # layers列表用于存储网络的层。首先添加输入层到第一个隐藏层的线性变换和激活函数，再使用Dropout层以防止过拟合。
#         for i in range(self.hidden_layers - 2):
#             layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
#                        self.activation, nn.Dropout(dropout)]
#         # 然后，使用循环为剩余的隐藏层添加线性层、激活函数和Dropout层。循环的次数为self.hidden_layers - 2，因为已经添加了第一层。
#         layers += [nn.Linear(hidden_dim, f_out)]  # 添加从最后一个隐藏层到输出层的线性变换。
#         self.layers = nn.Sequential(*layers)
#         # 使用 nn.Sequential 将所有层组合在一起，形成最终的多层感知机模型。这使得可以通过一次前向传播将输入数据传递通过所有层。
#
#     def forward(self, x):
#         # x:     B x S x f_in
#         # y:     B x S x f_out
#         y = self.layers(x)
#         return y
#
#
#
#
# # KP层（KPLayer & KPLayerApprox）：这两个类实现了通过DMD找到线性系统的一个步骤转移功能。
# # KPLayerApprox通过多步预测来扩展功能。在前向传播中，它们计算历时快照间的转移，并进行预测。
# class KPLayer(nn.Module):
#     """
#     A demonstration of finding one step transition of linear system by DMD iteratively
#     """
#
#     def __init__(self):
#         super(KPLayer, self).__init__()  # 通过 super() 调用父类（nn.Module）的构造函数，确保父类的初始化逻辑被正确执行。
#
#         self.K = None  # B E E
#
#     def one_step_forward(self, z, return_rec=False, return_K=False):
#         B, input_len, E = z.shape
#         assert input_len > 1, 'snapshots number should be larger than 1'
#         x, y = z[:, :-1], z[:, 1:]
#
#         # solve linear system：使用最小二乘法（torch.linalg.lstsq）求解线性回归，得到转换矩阵 K。K 的形状是 (B, E, E)。
#         self.K = torch.linalg.lstsq(x, y).solution  # B E E
#         if torch.isnan(self.K).any():  # 检查求得的转移矩阵 K 是否包含 NaN 值。如果有，则输出警告并将 K 替换为单位矩阵，确保后续计算的稳定性。
#             print('Encounter K with nan, replace K by identity matrix')
#             self.K = torch.eye(self.K.shape[1]).to(self.K.device).unsqueeze(0).repeat(B, 1, 1)
#
#         z_pred = torch.bmm(z[:, -1:], self.K)
#         if return_rec:
#             z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)
#             return z_rec, z_pred
#
#         return z_pred
#
#     def forward(self, z, pred_len=1):
#         assert pred_len >= 1, 'prediction length should not be less than 1'
#         z_rec, z_pred = self.one_step_forward(z, return_rec=True)
#         z_preds = [z_pred]
#         for i in range(1, pred_len):
#             z_pred = torch.bmm(z_pred, self.K)
#             z_preds.append(z_pred)
#         z_preds = torch.cat(z_preds, dim=1)
#         return z_rec, z_preds
#
#
# class KPLayerApprox(nn.Module):
#     """
#     Find koopman transition of linear system by DMD with multistep K approximation
#     """
#
#     def __init__(self):
#         super(KPLayerApprox, self).__init__()
#
#         self.K = None  # B E E
#         self.K_step = None  # B E E
#
#     def forward(self, z, pred_len=1):
#         # z:       B L E, koopman invariance space representation
#         # z_rec:   B L E, reconstructed representation
#         # z_pred:  B S E, forecasting representation
#         B, input_len, E = z.shape
#         assert input_len > 1, 'snapshots number should be larger than 1'
#         x, y = z[:, :-1], z[:, 1:]
#
#         # solve linear system
#         self.K = torch.linalg.lstsq(x, y).solution  # B E E
#
#         if torch.isnan(self.K).any():
#             print('Encounter K with nan, replace K by identity matrix')
#             self.K = torch.eye(self.K.shape[1]).to(self.K.device).unsqueeze(0).repeat(B, 1, 1)
#
#         z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)  # B L E
#
#         if pred_len <= input_len:
#             self.K_step = torch.linalg.matrix_power(self.K, pred_len)
#             if torch.isnan(self.K_step).any():
#                 print('Encounter multistep K with nan, replace it by identity matrix')
#                 self.K_step = torch.eye(self.K_step.shape[1]).to(self.K_step.device).unsqueeze(0).repeat(B, 1, 1)
#             z_pred = torch.bmm(z[:, -pred_len:, :], self.K_step)
#         else:
#             self.K_step = torch.linalg.matrix_power(self.K, input_len)
#             if torch.isnan(self.K_step).any():
#                 print('Encounter multistep K with nan, replace it by identity matrix')
#                 self.K_step = torch.eye(self.K_step.shape[1]).to(self.K_step.device).unsqueeze(0).repeat(B, 1, 1)
#             temp_z_pred, all_pred = z, []
#             for _ in range(math.ceil(pred_len / input_len)):
#                 temp_z_pred = torch.bmm(temp_z_pred, self.K_step)
#                 all_pred.append(temp_z_pred)
#             z_pred = torch.cat(all_pred, dim=1)[:, :pred_len, :]
#
#         return z_rec, z_pred
#
#
# #时间变换Koopman预测器（TimeVarKP）：TimeVarKP用于利用局部变化来预测时间变化的成分
# class TimeVarKP(nn.Module):
#     """
#     Koopman Predictor with DMD (analysitical solution of Koopman operator)
#     Utilize local variations within individual sliding window to predict the future of time-variant term
#     """
#
#     def __init__(self,
#                  enc_in=8,
#                  input_len=96,
#                  pred_len=96,
#                  seg_len=24,
#                  dynamic_dim=128,
#                  encoder=None,
#                  decoder=None,
#                  multistep=False,
#                  ):
#         super(TimeVarKP, self).__init__()
#         self.input_len = input_len
#         self.pred_len = pred_len
#         self.enc_in = enc_in
#         self.seg_len = seg_len
#         self.dynamic_dim = dynamic_dim
#         self.multistep = multistep
#         self.encoder, self.decoder = encoder, decoder
#         self.freq = math.ceil(self.input_len / self.seg_len)  # segment number of input
#         self.step = math.ceil(self.pred_len / self.seg_len)  # segment number of output
#         self.padding_len = self.seg_len * self.freq - self.input_len
#         # Approximate mulitstep K by KPLayerApprox when pred_len is large
#         self.dynamics = KPLayerApprox() if self.multistep else KPLayer()
#
#     def forward(self, x):
#         # x: B L C
#         B, L, C = x.shape
#
#         res = torch.cat((x[:, L - self.padding_len:, :], x), dim=1)
#
#         res = res.chunk(self.freq, dim=1)  # F x B P C, P means seg_len
#         res = torch.stack(res, dim=1).reshape(B, self.freq, -1)  # B F PC
#
#         res = self.encoder(res)  # B F H
#         x_rec, x_pred = self.dynamics(res, self.step)  # B F H, B S H
#
#         x_rec = self.decoder(x_rec)  # B F PC
#         x_rec = x_rec.reshape(B, self.freq, self.seg_len, self.enc_in)
#         x_rec = x_rec.reshape(B, -1, self.enc_in)[:, :self.input_len, :]  # B L C
#
#         x_pred = self.decoder(x_pred)  # B S PC
#         x_pred = x_pred.reshape(B, self.step, self.seg_len, self.enc_in)
#         x_pred = x_pred.reshape(B, -1, self.enc_in)[:, :self.pred_len, :]  # B S C
#
#         return x_rec, x_pred
#
#
# #时间不变Koopman预测器（TimeInvKP）：TimeInvKP用于利用历史快照和预测窗口来预测时间不变的成分。
# class TimeInvKP(nn.Module):
#     """
#     Koopman Predictor with learnable Koopman operator
#     Utilize lookback and forecast window snapshots to predict the future of time-invariant term
#     """
#
#     def __init__(self,
#                  input_len=96,
#                  pred_len=96,
#                  dynamic_dim=128,
#                  encoder=None,
#                  decoder=None):
#         super(TimeInvKP, self).__init__()
#         self.dynamic_dim = dynamic_dim
#         self.input_len = input_len
#         self.pred_len = pred_len
#         self.encoder = encoder
#         self.decoder = decoder
#
#         K_init = torch.randn(self.dynamic_dim, self.dynamic_dim)
#         U, _, V = torch.svd(K_init)  # stable initialization
#         self.K = nn.Linear(self.dynamic_dim, self.dynamic_dim, bias=False)
#         self.K.weight.data = torch.mm(U, V.t())
#
#     def forward(self, x):
#         # x: B L C
#         res = x.transpose(1, 2)  # B C L
#         res = self.encoder(res)  # B C H
#         res = self.K(res)  # B C H
#         res = self.decoder(res)  # B C S
#         res = res.transpose(1, 2)  # B S C
#
#         return res
#
#
# # Koopman预测模型（Model）：
# class Model(nn.Module):
#     '''
#     Koopman Forecasting Model
#     '''
#
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.mask_spectrum = configs.mask_spectrum
#         self.enc_in = configs.enc_in
#         self.input_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.seg_len = configs.seg_len
#         self.num_blocks = configs.num_blocks
#         self.dynamic_dim = configs.dynamic_dim
#         self.hidden_dim = configs.hidden_dim
#         self.hidden_layers = configs.hidden_layers
#         self.multistep = configs.multistep
#
#         self.disentanglement = FourierFilter(self.mask_spectrum)
#
#         # shared encoder/decoder to make koopman embedding consistent
#         self.time_inv_encoder = MLP(f_in=self.input_len, f_out=self.dynamic_dim, activation='relu',
#                                     hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
#         self.time_inv_decoder = MLP(f_in=self.dynamic_dim, f_out=self.pred_len, activation='relu',
#                                     hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
#         self.time_inv_kps = self.time_var_kps = nn.ModuleList([
#             TimeInvKP(input_len=self.input_len,
#                       pred_len=self.pred_len,
#                       dynamic_dim=self.dynamic_dim,
#                       encoder=self.time_inv_encoder,
#                       decoder=self.time_inv_decoder)
#             for _ in range(self.num_blocks)])
#
#         # shared encoder/decoder to make koopman embedding consistent
#         self.time_var_encoder = MLP(f_in=self.seg_len * self.enc_in, f_out=self.dynamic_dim, activation='tanh',
#                                     hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
#         self.time_var_decoder = MLP(f_in=self.dynamic_dim, f_out=self.seg_len * self.enc_in, activation='tanh',
#                                     hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
#         self.time_var_kps = nn.ModuleList([
#             TimeVarKP(enc_in=configs.enc_in,
#                       input_len=self.input_len,
#                       pred_len=self.pred_len,
#                       seg_len=self.seg_len,
#                       dynamic_dim=self.dynamic_dim,
#                       encoder=self.time_var_encoder,
#                       decoder=self.time_var_decoder,
#                       multistep=self.multistep)
#             for _ in range(self.num_blocks)])
#
#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
#         # x_enc: B L C
#
#         # Series Stationarization adopted from NSformer
#         mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
#         x_enc = x_enc - mean_enc
#         std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
#         x_enc = x_enc / std_enc
#
#
#         # # Koopman Forecasting
#         residual, forecast = x_enc, None
#         for i in range(self.num_blocks):
#             time_var_input, time_inv_input = self.disentanglement(residual)
#             time_inv_output = self.time_inv_kps[i](time_inv_input)
#             time_var_backcast, time_var_output = self.time_var_kps[i](time_var_input)
#             residual = residual - time_var_backcast
#             if forecast is None:
#                 forecast = (time_inv_output + time_var_output)
#             else:
#                 forecast += (time_inv_output + time_var_output)
#
#         # Series Stationarization adopted from NSformer
#         res = forecast * std_enc + mean_enc
#
#         return res
#

#KoopTCN
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 添加TCN层
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.1):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(
                in_channels, out_channels,
                kernel_size, stride=1,
                dilation=dilation_size,
                padding=(kernel_size - 1) * dilation_size,
                dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) \
            if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.dropout(self.relu(out))
        out = self.conv2(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        # 裁剪以匹配序列长度
        out = out[:, :, :res.shape[-1]]
        return self.relu(out + res)


# 修改后的 FourierFilter 类
class FourierFilter(nn.Module):
    def __init__(self, mask_spectrum, enc_in):
        super(FourierFilter, self).__init__()
        self.mask_spectrum = mask_spectrum
        self.enc_in = enc_in
        # 修改 num_channels 以匹配输入特征数量
        self.tcn_time_inv = TemporalConvNet(num_inputs=self.enc_in, num_channels=[self.enc_in, self.enc_in ])
        self.tcn_time_var = TemporalConvNet(num_inputs=self.enc_in, num_channels=[self.enc_in, self.enc_in ])
        # 添加输出通道属性
        self.out_channels_inv = self.enc_in  # 对应 num_channels 最后一个元素
        self.out_channels_var = self.enc_in
    def forward(self, x):
        # x: B x L x C
        xf = torch.fft.rfft(x, dim=1)         # 傅里叶变换
        mask = torch.ones_like(xf)
        mask[:, self.mask_spectrum, :] = 0
        x_var = torch.fft.irfft(xf * mask, dim=1)  # 变化成分
        x_inv = x - x_var                           # 不变成分

        # TCN 提取时序特征
        x_inv = self.tcn_time_inv(x_inv.transpose(1, 2)).transpose(1, 2)
        x_var = self.tcn_time_var(x_var.transpose(1, 2)).transpose(1, 2)

        return x_inv, x_var


# 多层感知机（MLP）——支持自适应维度转置
class MLP(nn.Module):
    def __init__(self,
                 f_in,
                 f_out,
                 hidden_dim=128,
                 hidden_layers=2,
                 dropout=0.05,
                 activation='relu'):
        super(MLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError(f"Unsupported activation: {activation}")

        # 构建网络层
        layers = [nn.Linear(self.f_in, self.hidden_dim),
                  self.activation, nn.Dropout(self.dropout)]
        for _ in range(self.hidden_layers - 2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                       self.activation, nn.Dropout(self.dropout)]
        layers += [nn.Linear(self.hidden_dim, self.f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x: B x S x E
        B, S, E = x.shape
        # 如果最后一维与期望不符，且第二维和期望 f_in 相符，则自动转置
        if E != self.f_in:
            if S == self.f_in:
                x = x.transpose(1, 2)  # B x E x S
                B, S, E = x.shape
            else:
                raise ValueError(f"MLP输入维度错误：期望最后一维={self.f_in}，"
                                 f"却得到{E}。可尝试检查前一层输出形状。")
        # 再次检查
        assert E == self.f_in, f"MLP 维度仍不匹配：期望 {self.f_in}, 得到 {E}"

        x = x.reshape(B * S, E)        # (B*S) x E
        y = self.layers(x)             # (B*S) x f_out
        y = y.reshape(B, S, -1)        # B x S x f_out
        return y


# Koosman 注意力动力学层
class KPAttentionLayer(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8):
        super(KPAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.K_proj = nn.Linear(embed_dim, embed_dim)  # 新增线性变换层
        self.K = None

    def one_step_forward(self, z, return_rec=False):
        B, L, E = z.shape
        x, y = z[:, :-1], z[:, 1:]

        # 调整维度为 (sequence_length, batch_size, embed_dim)
        x_t = x.transpose(0, 1)  # (L-1, B, E)
        y_t = y.transpose(0, 1)  # (L-1, B, E)

        # 计算注意力输出
        attn_out, _ = self.attention(x_t, x_t, y_t)
        attn_out = attn_out.transpose(0, 1)  # (B, L-1, E)

        # 估计Koopman转移矩阵
        x_pseudo_inv = torch.linalg.pinv(x)  # (B, E, L-1)

        # 修改矩阵乘法顺序
        self.K = torch.bmm(x_pseudo_inv, attn_out)  # (B, E, E)
        self.K = self.K_proj(self.K)  # 添加可学习的线性变换

        # 维度验证
        assert self.K.shape == (B, E, E), \
            f"K矩阵维度错误: 期望({B}, {E}, {E}), 实际{self.K.shape}"

        # 重构与预测
        z_pred = torch.bmm(z[:, -1:], self.K)  # (B,1,E) * (B,E,E) => (B,1,E)

        if return_rec:
            z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)
            return z_rec, z_pred
        return z_pred

    def forward(self, z, pred_len=1):
        z_rec, z_pred = self.one_step_forward(z, return_rec=True)
        preds = [z_pred]
        for _ in range(1, pred_len):
            z_pred = torch.bmm(z_pred, self.K)
            preds.append(z_pred)
        preds = torch.cat(preds, dim=1)
        return z_rec, preds


# 时间变化 KP 预测器
class TimeVarKP(nn.Module):
    def __init__(self, enc_in=8, input_len=96, pred_len=96, seg_len=24,
                 dynamic_dim=128, encoder=None, decoder=None, multistep=False,
                 hidden_dim=128, hidden_layers=2):
        super(TimeVarKP, self).__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.seg_len = seg_len
        self.dynamic_dim = dynamic_dim
        self.multistep = multistep
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

        self.freq = math.ceil(self.input_len / self.seg_len)
        self.step = math.ceil(self.pred_len / self.seg_len)
        self.padding_len = self.seg_len * self.freq - self.input_len

        # 调整 f_in 参数以匹配实际的输入维度
        self.encoder = MLP(
            f_in=self.seg_len * self.enc_in, f_out=self.dynamic_dim,
            hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers,
            activation='tanh')
        self.decoder = MLP(
            f_in=self.dynamic_dim, f_out=self.seg_len * self.enc_in,
            hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers,
            activation='tanh')
        self.dynamics = KPAttentionLayer(embed_dim=self.dynamic_dim)

    def forward(self, x):
        # x: B x L x C
        B, L, C = x.shape
        # 前填充，使长度能被 seg_len 整除
        pad = x[:, L - self.padding_len:, :]
        res = torch.cat((pad, x), dim=1)
        # 分段
        chunks = res.chunk(self.freq, dim=1)
        res = torch.stack(chunks, dim=1).reshape(B, self.freq, -1)  # B x F x (seg_len*C)
        # 编码到 Koopman 嵌入
        res = self.encoder(res)  # B x F x H
        # DMD / Attention 预测
        backcast, forecast = self.dynamics(res, pred_len=self.step)
        # 解码 backcast（重构）和 forecast（预测）
        back = self.decoder(backcast).reshape(B, self.freq, self.seg_len, self.enc_in)
        back = back.reshape(B, -1, self.enc_in)[:, :self.input_len, :]
        fore = self.decoder(forecast).reshape(B, self.step, self.seg_len, self.enc_in)
        fore = fore.reshape(B, -1, self.enc_in)[:, :self.pred_len, :]
        return back, fore


# 时间不变 KP 预测器
class TimeInvKP(nn.Module):
    def __init__(self, input_len=96, pred_len=96, dynamic_dim=128, encoder=None, decoder=None):
        super(TimeInvKP, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_len = input_len
        self.pred_len = pred_len

        # 使用 SVD 初始化可学习 Koopman 算子
        K_init = torch.randn(dynamic_dim, dynamic_dim)
        U, _, V = torch.svd(K_init)
        self.K = nn.Linear(dynamic_dim, dynamic_dim, bias=False)
        self.K.weight.data = torch.mm(U, V.t())

    def forward(self, x):
        # x: B x L x C
        res = x.transpose(1, 2)       # B x C x L
        res = self.encoder(res)      # B x C x H
        res = self.K(res)            # B x C x H
        res = self.decoder(res)      # B x C x L
        return res.transpose(1, 2)   # B x L x C


# 顶层 Koopman 预测模型
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.mask_spectrum = configs.mask_spectrum
        self.enc_in = configs.enc_in
        self.input_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seg_len = configs.seg_len
        self.num_blocks = configs.num_blocks
        self.dynamic_dim = configs.dynamic_dim
        self.hidden_dim = configs.hidden_dim
        self.hidden_layers = configs.hidden_layers
        self.multistep = configs.multistep

        self.disentanglement = FourierFilter(self.mask_spectrum, self.enc_in)

        # 获取TCN的输出通道数
        tcn_var_out = self.disentanglement.out_channels_var  # 128

        # 时间不变分量的共享编码/解码
        self.time_inv_encoder = MLP(
            f_in=self.input_len,
            f_out=self.dynamic_dim,
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
            activation='relu')
        self.time_inv_decoder = MLP(
            f_in=self.dynamic_dim,
            f_out=self.input_len,
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
            activation='relu')
        self.time_inv_kps = nn.ModuleList([
            TimeInvKP(input_len=self.input_len, pred_len=self.input_len,
                      dynamic_dim=self.dynamic_dim,
                      encoder=self.time_inv_encoder,
                      decoder=self.time_inv_decoder)
            for _ in range(self.num_blocks)
        ])

        # 时间变化分量的编码/解码调整输入维度
        # 修改 num_channels 以匹配输入特征数量
        self.time_var_encoder = MLP(
            f_in=self.seg_len * self.enc_in,  # 原始输入特征维度 1*9=9
            f_out=self.dynamic_dim,
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
            activation='tanh')
        self.time_var_decoder = MLP(
            f_in=self.dynamic_dim,
            f_out=self.seg_len * self.enc_in,  # 恢复原始维度 1*9=9
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
            activation='tanh')

        # 确保TimeVarKP使用原始特征维度
        self.time_var_kps = nn.ModuleList([
            TimeVarKP(enc_in=self.enc_in,  # 使用configs.enc_in=9
                      input_len=self.input_len,
                      pred_len=self.pred_len,
                      seg_len=self.seg_len,
                      dynamic_dim=self.dynamic_dim,
                      encoder=self.time_var_encoder,
                      decoder=self.time_var_decoder,
                      multistep=self.multistep,
                      hidden_dim=self.hidden_dim,
                      hidden_layers=self.hidden_layers)
            for _ in range(self.num_blocks)
        ])

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: B x L x C
        # 标准化
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_norm = (x_enc - mean_enc)
        std_enc = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_norm = x_norm / std_enc

        # 多层 Koopman 解耦与预测
        residual = x_norm
        forecast = None
        for i in range(self.num_blocks):
            inv_part, var_part = self.disentanglement(residual)
            inv_rec = self.time_inv_kps[i](inv_part)
            var_back, var_for = self.time_var_kps[i](var_part)
            residual = residual - var_back
            if forecast is None:
                forecast = inv_rec + var_for
            else:
                forecast = forecast + (inv_rec + var_for)

        # 反标准化
        output = forecast * std_enc + mean_enc
        return output


