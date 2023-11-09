import torch.nn as nn
import torch
import math

def conv1x1x1(inplanes, planes, stride=1):
    return nn.Conv3d(inplanes, planes, 1, stride=stride, bias=False)  # inplane 输入通道数， plane输出通道数


def conv1x3x3(inplanes, planes, stride=1, padding=(0, 1, 1)):
    return nn.Conv3d(inplanes, planes, (1, 3, 3), stride=stride, padding=padding, bias=False)


def conv3x1x1(inplanes, planes, stride=1, padding=(1, 0, 0)):
    return nn.Conv3d(inplanes, planes, (3, 1, 1), stride=stride, padding=padding, bias=False)


class eca_block(nn.Module):
    # 初始化, in_channel代表特征图的输入通道数, b和gama代表公式中的两个系数
    def __init__(self, in_channel, b=1, gama=2):
        # 继承父类初始化
        super(eca_block, self).__init__()
        # 根据输入通道数自适应调整卷积核大小
        kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
        # 如果卷积核大小是奇数，就使用它
        if kernel_size % 2:
           kernel_size = kernel_size
        # 如果卷积核大小是偶数，就把它变成奇数
        else:
            kernel_size = kernel_size+1

        # 卷积时，为例保证卷积前后的size不变，需要0填充的数量
        padding = kernel_size // 2

        # 全局平均池化，输出的特征图的宽高=1
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max_pool = nn.AdaptiveMaxPool3d(output_size=(4, 1,1))
        # 1D卷积，输入和输出通道数都=1，卷积核大小是自适应的
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size,
                              bias=False, padding=padding)
        # sigmoid激活函数，权值归一化
        self.sigmoid = nn.Sigmoid()
        self.gama = gama
        self.b = b
        self.relu = nn.ReLU()
    # 前向传播
    def forward(self, inputs):
        # 获得输入图像的shape
        b, c, t, h, w = inputs.shape

        # 全局平均池化 [b,c,t,h,w]==>[b,c,t,1,1]
        avg_pool = self.avg_pool(inputs)
        max_pool = self.max_pool(inputs)
        #[b,c,t,1,1]==>[b,c,t]
        avg_pool = avg_pool.squeeze(-1)
        avg_pool = avg_pool.squeeze(-1)
        max_pool = max_pool.squeeze(-1)
        max_pool = max_pool.squeeze(-1)
        #[b,c,t]==>[b,t,c]   [b,c,t]==>[t,b,c]
        avg_pool = avg_pool.permute(2, 0, 1)
        max_pool = max_pool.permute(2, 0, 1)
        # [b,t,c]==>[b,1,c] [t,b,c]==>[t,1,c]
        max_pool = torch.mean(max_pool, dim=1, keepdim=True)
        avg_pool = torch.mean(avg_pool, dim=1, keepdim=True)
        avg_pool = self.conv(avg_pool)
        max_pool = self.conv(max_pool)
        max_pool = self.relu(max_pool)
        avg_pool = self.relu(avg_pool)
        avg_pool = self.conv(avg_pool)
        max_pool = self.conv(max_pool)
        x = max_pool + avg_pool
        x = x.permute(1, 2, 0)
        x = self.sigmoid(x)
        x = x.view(1, c, t, 1, 1)
        # 将输入特征图和通道权重相乘[b,c,t,h,w]*[b,c,1,1,1]==>[b,c,t,h,w]  [b,c,t,h,w]*[1,c,t,1,1]==>[b,c,t,h,w]
        outputs = x * inputs

        #outputs =inputs*(x*self.gama+self.b)
        return outputs

# ---------------------------------------------------- #
# （2）空间注意力机制
class spatial_attention(nn.Module):
    # 初始化，卷积核大小为7*7
    def __init__(self, kernel_size=7):
        # 继承父类初始化方法
        super(spatial_attention, self).__init__()

        # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        padding = kernel_size // 2
        # 7*7卷积融合通道信息 [b,2,h,w]==>[b,1,h,w],因为卷积数入前是两个池化拼接后的结果，所以维度是2
        self.conv = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        # sigmoid函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 在通道维度上最大池化 [b,1，t,h,w]  keepdim保留原有深度
        # 返回值是在某维度的最大值和对应的索引
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)
        #x_maxpool1, _ =torch.max(x_maxpool, dim=2, keepdim=True)
        # 在通道维度上平均池化 [b,1,t,h,w]
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        #x_avgpool1 = torch.mean(x_avgpool, dim=2, keepdim=True)
        # 池化后的结果在通道维度上堆叠 [b,2,t,h,w]
        x = torch.cat([x_maxpool, x_avgpool], dim=1)
        #x1 = torch.cat([x_maxpool1, x_avgpool1], dim=1)
        # 卷积融合通道信息 [b,2,t,h,w]==>[b,1,t,h,w]
        x = self.conv(x)
        #x1 = self.conv(x1)
        # 空间权重归一化
        x = self.sigmoid(x)
        #x1 = self.sigmoid(x1)
        # 输入特征图和空间权重相乘
        outputs = inputs * x
        #outputs = outputs * x1
        return outputs


# ---------------------------------------------------- #
# （3）CBAM注意力机制
class cbam(nn.Module):
    # 初始化，in_channel和ratio=4代表通道注意力机制的输入通道数和第一个全连接下降的通道数
    # kernel_size代表空间注意力机制的卷积核大小
    def __init__(self, in_channel, kernel_size=7):  #########################################################
        # 继承父类初始化方法
        super(cbam, self).__init__()

        # 实例化通道注意力机制
        self.channel_attention = eca_block(in_channel=in_channel)
        # 实例化空间注意力机制
        self.spatial_attention = spatial_attention(kernel_size=kernel_size)

    # 前向传播
    def forward(self, inputs):
        # 先将输入图像经过通道注意力机制
        x = self.channel_attention(inputs)
        # 然后经过空间注意力机制
        x = self.spatial_attention(x)

        return x


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class Degenerate_Bottlenec(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Degenerate_Bottlenec, self).__init__()
        self.conv1 = conv1x1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv1x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.attentions = cbam(64)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.attentions(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


'''
因为res2 res3都使用了退化快，但是两个网络的输入通道数是不同的，所以将两个网络进行区分设计。比较麻烦。
'''


class Degenerate_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Degenerate_Bottleneck, self).__init__()
        self.conv1 = conv1x1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv1x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.attentions = cbam(128)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.attentions(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Non_degenerate_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Non_degenerate_Bottleneck, self).__init__()
        self.conv1 = conv3x1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv1x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


#
# class Non_degenerate_Bottleneck1(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Non_degenerate_Bottleneck1, self).__init__()
#         self.conv1 = conv3x1x1(inplanes, planes)
#         self.bn1 = nn.BatchNorm3d(planes)
#         self.conv2 = conv1x3x3(planes, planes, stride)
#         self.attentions = cbam(8)
#         self.bn2 = nn.BatchNorm3d(planes)
#         self.conv3 = conv1x1x1(planes, planes * self.expansion)
#         self.bn3 = nn.BatchNorm3d(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#
#         out = self.attentions(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
# class Non_degenerate_Bottleneck2(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Non_degenerate_Bottleneck2, self).__init__()
#         self.conv1 = conv3x1x1(inplanes, planes)
#         self.bn1 = nn.BatchNorm3d(planes)
#         self.conv2 = conv1x3x3(planes, planes, stride)
#         self.attentions = cbam(16)
#         self.bn2 = nn.BatchNorm3d(planes)
#         self.conv3 = conv1x1x1(planes, planes * self.expansion)
#         self.bn3 = nn.BatchNorm3d(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.attentions(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
class slowfast(nn.Module):

    def __init__(self, layers, num_classes=400):
        super(slowfast, self).__init__()

        self.slow_conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.fast_conv1 = nn.Conv3d(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.slow_bn1 = nn.BatchNorm3d(64)
        self.fast_bn1 = nn.BatchNorm3d(8)
        self.sa = spatial_attention(7)
        self.sa1 = spatial_attention(7)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.fast_res2 = self._make_layer(Non_degenerate_Bottleneck, 8, 8, layers[0])
        self.fast_res3 = self._make_layer(Non_degenerate_Bottleneck, 32, 16, layers[1], stride=(1, 2, 2))
        self.fast_res4 = self._make_layer(Non_degenerate_Bottleneck, 64, 32, layers[2], stride=(1, 2, 2))
        self.fast_res5 = self._make_layer(Non_degenerate_Bottleneck, 128, 64, layers[3], stride=(1, 2, 2))

        self.slow_res2 = self._make_layer(Degenerate_Bottlenec, 64 + 8 * 2, 64, layers[0])
        self.slow_res3 = self._make_layer(Degenerate_Bottleneck, 256 + 32 * 2, 128, layers[1], stride=(1, 2, 2))
        self.slow_res4 = self._make_layer(Non_degenerate_Bottleneck, 512 + 64 * 2, 256, layers[2], stride=(1, 2, 2))
        self.slow_res5 = self._make_layer(Non_degenerate_Bottleneck, 1024 + 128 * 2, 512, layers[3], stride=(1, 2, 2))

        self.tconv1 = nn.Conv3d(8, 8 * 2, kernel_size=(5, 1, 1), stride=(8, 1, 1), padding=(2, 0, 0), bias=False)
        self.tconv2 = nn.Conv3d(32, 32 * 2, kernel_size=(5, 1, 1), stride=(8, 1, 1), padding=(2, 0, 0), bias=False)
        self.tconv3 = nn.Conv3d(64, 64 * 2, kernel_size=(5, 1, 1), stride=(8, 1, 1), padding=(2, 0, 0), bias=False)
        self.tconv4 = nn.Conv3d(128, 128 * 2, kernel_size=(5, 1, 1), stride=(8, 1, 1), padding=(2, 0, 0), bias=False)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.convlstm = ConvLSTM(input_dim=1024, hidden_dim=1024, kernel_size=(3, 3), num_layers=3, batch_first=True)
        self.fc = nn.Linear(1024, num_classes)
        self.fc1 = nn.Linear(32, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=(1, 1, 1)):
        downsample = None
        if stride != (1, 1, 1) or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(inplanes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # the input in the paper should be [N,C,T*τ,H,W]
        # the fast way's frame interval is 2 and the low way 16
        # so we only input [N,C,T/2*τ,H,W] to save memory and time.  假设τ=3 T变化2或者16  N为batch_size

        fast_out, lateral = self._fast_net(x)
        slow_out = self._slow_net(x[:, :, ::8, ...], lateral)

        # w1 = nn.Parameter(torch.tensor(fast_out.clone()))
        # w2 = nn.Parameter(torch.tensor(slow_out.clone()))
        # print('w1=',fast_out.shape)
        # print('w2=', slow_out.shape)
        # fusion_out = torch.cat([w1*fast_out, w2*slow_out], dim=1)
        fusion_out = torch.cat([fast_out, slow_out], dim=1)
        # fusion_out = torch.add(fast_out, slow_out)
        # torch.save(w1,'/home/yao/SlowFast-Network-master/final_w1')
        # torch.save(w2, '/home/yao/SlowFast-Network-master/final_w2')
        fusion_out = fusion_out.permute(0, 2, 1, 3, 4)
        layer_output_list, last_state_list = self.convlstm(fusion_out)

        last_layer_output = layer_output_list[-1]

        fusion_out = last_layer_output[:, -1, ...]
        fusion_out = self.avgpool1(fusion_out).view(fusion_out.size(0), -1)
        output = self.fc(fusion_out)

        return output

    def _slow_net(self, x, lateral):
        out = self.slow_conv1(x)
        out = self.sa(out)
        out = self.slow_bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = torch.cat([out, lateral[0]], dim=1)

        out = self.slow_res2(out)
        out = torch.cat([out, lateral[1]], dim=1)

        out = self.slow_res3(out)
        out = torch.cat([out, lateral[2]], dim=1)

        out = self.slow_res4(out)
        out = torch.cat([out, lateral[3]], dim=1)

        slow_out = self.slow_res5(out)
        B, C, T, H, W = slow_out.size()
        slow_out = slow_out.permute(0, 2, 3, 4, 1).reshape(-1, 2048)  # （T*B*H*W，C）
        fc2 = nn.Linear(in_features=2048, out_features=768).to('cuda')  # 改变C的通道数
        slow_out = fc2(slow_out).reshape(B, 768, T, H, W)  # （T*B*H*W，C）变为（B，C，T,H，W）
        # slow_out = self.avgpool(out).view(x.size(0), -1)

        return slow_out

    def _fast_net(self, x):
        lateral = []
        out = self.fast_conv1(x)
        out = self.sa1(out)
        out = self.fast_bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        lateral.append(self.tconv1(out))

        out = self.fast_res2(out)
        lateral.append(self.tconv2(out))
        out = self.fast_res3(out)
        lateral.append(self.tconv3(out))
        out = self.fast_res4(out)
        lateral.append(self.tconv4(out))
        out = self.fast_res5(out)

        # fast_out = self.avgpool(out).view(x.size(0), -1)
        # print('fast_out=',out.shape)
        fast_out = out.permute(0, 1, 3, 4, 2).reshape(-1, 32)
        fast_out = self.fc1(fast_out)
        fast_out = fast_out.reshape(-1, 256, 7, 7, 4).permute(0, 1, 4, 2, 3)
        return fast_out, lateral


def SlowFastNet(num_classes=400):
    model = slowfast([3, 4, 6, 3], num_classes=num_classes)
    return model


if __name__ == '__main__':
    model = SlowFastNet()
    device = torch.device('cuda')
    model.cuda(device=device)

    from torchsummary import summary

    summary(model, (3, 48, 224, 224))

    from tools import print_model_parm_flops

    print_model_parm_flops(model, torch.randn(1, 3, 48, 224, 224))
