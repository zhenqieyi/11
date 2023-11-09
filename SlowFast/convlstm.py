import torch.nn as nn
import torch

'''
0.1 在块网络使用convlstm，效果与tcn lstm同位置没有区别
0.2 修改了convlstm中的两个参数，分别为卷积和大小与convlstm层数，分别改为（5，5）  3  因为属于较深层网络，所以卷积和不适以较大
0.2 在快慢网络分别使用convlstm  num=3 在曼网络进行了一些修改，因为不修改的话因为曼网络维度太高，回报显存，所以采取的措施是县使用全连阶层降低
    曼网络维度，再使用convlstm
'''
def conv1x1x1(inplanes, planes, stride=1):
    return nn.Conv3d(inplanes, planes, 1, stride=stride, bias=False)#inplane 输入通道数， plane输出通道数


def conv1x3x3(inplanes, planes, stride=1, padding=(0, 1, 1)):
    return nn.Conv3d(inplanes, planes, (1, 3, 3), stride=stride, padding=padding, bias=False)


def conv3x1x1(inplanes, planes, stride=1, padding=(1, 0, 0)):
    return nn.Conv3d(inplanes, planes, (3, 1, 1), stride=stride, padding=padding, bias=False)

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


class Degenerate_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Degenerate_Bottleneck, self).__init__()
        self.conv1 = conv1x1x1(inplanes, planes)
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


class slowfast(nn.Module):

    def __init__(self, layers, num_classes=101):
        super(slowfast, self).__init__()

        self.slow_conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.fast_conv1 = nn.Conv3d(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.slow_bn1 = nn.BatchNorm3d(64)
        self.fast_bn1 = nn.BatchNorm3d(8)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.fast_res2 = self._make_layer(Non_degenerate_Bottleneck, 8, 8, layers[0])
        self.fast_res3 = self._make_layer(Non_degenerate_Bottleneck, 32, 16, layers[1], stride=(1, 2, 2))
        self.fast_res4 = self._make_layer(Non_degenerate_Bottleneck, 64, 32, layers[2], stride=(1, 2, 2))
        self.fast_res5 = self._make_layer(Non_degenerate_Bottleneck, 128, 64, layers[3], stride=(1, 2, 2))

        self.slow_res2 = self._make_layer(Degenerate_Bottleneck, 64 + 8 * 2, 64, layers[0])
        self.slow_res3 = self._make_layer(Degenerate_Bottleneck, 256 + 32 * 2, 128, layers[1], stride=(1, 2, 2))
        self.slow_res4 = self._make_layer(Non_degenerate_Bottleneck, 512 + 64 * 2, 256, layers[2], stride=(1, 2, 2))
        self.slow_res5 = self._make_layer(Non_degenerate_Bottleneck, 1024 + 128 * 2, 512, layers[3], stride=(1, 2, 2))

        self.tconv1 = nn.Conv3d(8, 8 * 2, kernel_size=(5, 1, 1), stride=(8, 1, 1), padding=(2, 0, 0), bias=False)
        self.tconv2 = nn.Conv3d(32, 32 * 2, kernel_size=(5, 1, 1), stride=(8, 1, 1), padding=(2, 0, 0), bias=False)
        self.tconv3 = nn.Conv3d(64, 64 * 2, kernel_size=(5, 1, 1), stride=(8, 1, 1), padding=(2, 0, 0), bias=False)
        self.tconv4 = nn.Conv3d(128, 128 * 2, kernel_size=(5, 1, 1), stride=(8, 1, 1), padding=(2, 0, 0), bias=False)
        self.convlstm = ConvLSTM(input_dim=256, hidden_dim=256, kernel_size=(3, 3), num_layers=3,batch_first=True)
        self.convlstms = ConvLSTM(input_dim=1024, hidden_dim=1024, kernel_size=(3, 3), num_layers=3, batch_first=True)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.avgpool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256 + 1024, num_classes)
        self.fc1 = nn.Linear(2048, 1024)
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
        fusion_out = torch.cat([fast_out, slow_out], dim=1)
        output = self.fc(fusion_out)

        return output

    def _slow_net(self, x, lateral):
        out = self.slow_conv1(x)
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

        out = self.slow_res5(out)
        B, C, T, H, W = out.size()
        out = out.permute(0,2,3,4,1).reshape(-1, 2048)  # （T*B*H*W，C）
        fc1 = nn.Linear(in_features=2048, out_features=1024).to('cuda')#改变C的通道数
        out = fc1(out).reshape(B, 1024, T,H, W)  # （T*B*H*W，C）变为（B，C，T,H，W）
        out = out.permute(0, 2, 1, 3, 4)#(b,t,c,h,w)
        layer_output_list, last_state_list = self.convlstms(out)

        last_layer_output = layer_output_list[-1]

        slow_out = last_layer_output[:, -1, ...]

        slow_out = self.avgpool2(slow_out).view(x.size(0), -1)
        #slow_out = self.avgpool(out).view(x.size(0), -1)
        #print('slow_out',slow_out.shape)
        return slow_out

    def _fast_net(self, x):
        lateral = []
        out = self.fast_conv1(x)
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
        out = out.permute(0,2,1,3,4)
        layer_output_list, last_state_list= self.convlstm(out)


        last_layer_output = layer_output_list[-1]

        fast_out= last_layer_output[:,-1,...]

        fast_out = self.avgpool2(fast_out).view(x.size(0), -1)
        return fast_out, lateral


def SlowFastNet(num_classes=101):
    model = slowfast([3, 4, 6, 3], num_classes=num_classes)
    return model


if __name__ == '__main__':
    model = SlowFastNet()
    device = torch.device('cuda')
    model.cuda(device=device)

    from torchsummary import summary
    summary(model, (3, 48, 224, 224))

    from tools import print_model_parm_flops
    print_model_parm_flops(model,torch.randn(1,3,48,224,224))
