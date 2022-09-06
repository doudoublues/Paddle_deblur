import glob
import os
import cv2
import time
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import sys


class NonLocalBlock(nn.Layer):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2D(channel, self.inter_channel, kernel_size=1, stride=1, bias_attr=False)
        self.conv_theta = nn.Conv2D(channel, self.inter_channel, kernel_size=1, stride=1, bias_attr=False)
        self.conv_g = nn.Conv2D(channel, self.inter_channel, kernel_size=1, stride=1, bias_attr=False)
        self.softmax = nn.Softmax(axis=1)
        self.conv_mask = nn.Conv2D(self.inter_channel, channel, kernel_size=1, stride=1, bias_attr=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.shape
        # 获取phi特征，维度为[N, C/2, H * W]，注意是要保留batch和通道维度的，是在HW上
        x_phi = self.conv_phi(x)
        x_phi = paddle.reshape(x_phi, (b, c, -1))
        # 获取theta特征，维度为[N, H * W, C/2]
        x_theta = self.conv_theta(x)
        x_theta = paddle.transpose(paddle.reshape(x_theta, (b, c, -1)), (0, 2, 1))
        # 获取g特征，维度为[N, H * W, C/2]
        x_g = self.conv_g(x)
        # x_g = paddle.reshape(x_g, (b, c, -1)).permute(0, 2, 1).contiguous()
        x_g = paddle.transpose(paddle.reshape(x_g, (b, c, -1)), (0, 2, 1))
        # 对phi和theta进行矩阵乘，[N, H * W, H * W]
        # print(x_theta.shape, x_phi.shape) # [1, 8192, 64] [1, 64, 8192]
        mul_theta_phi = paddle.matmul(x_theta, x_phi)
        # softmax拉到0~1之间
        # print(mul_theta_phi.shape) # [1, 8192, 8192]
        mul_theta_phi = self.softmax(mul_theta_phi)
        # 与g特征进行矩阵乘运算，[N, H * W, C/2]
        mul_theta_phi_g = paddle.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = paddle.transpose(mul_theta_phi_g, (0, 2, 1))
        mul_theta_phi_g = paddle.reshape(mul_theta_phi_g, (b, self.inter_channel, h, w))
        # 1X1卷积扩充通道数
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x # 残差连接
        return out

class NonLocalModule(nn.Layer):

    def __init__(self, in_channels, **kwargs):
        super(NonLocalModule, self).__init__()

    def init_modules(self):
        for m in self.sublayers():
            if len(m.sublayers()) > 0:
                continue
            if isinstance(m, nn.Conv2D):
                m.weight=m.create_parameter(m.weight.shape, default_initializer=nn.initializer.KaimingNormal())
                if len(list(m.parameters())) > 1:
                    m.bias.set_value(paddle.zeros(m.bias.shape))
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.set_value(paddle.zeros(m.weight.shape))
                m.bias.set_value(paddle.zeros(m.bias.shape))
            elif isinstance(m, nn.GroupNorm):
                m.weight.set_value(paddle.zeros(m.weight.shape))
                m.bias.set_value(paddle.zeros(m.bias.shape))
            elif len(list(m.parameters())) > 0:
                raise ValueError("UNKOWN NONLOCAL LAYER TYPE:", m)


class NonLocal(NonLocalModule):
    """Spatial NL block for image classification.
       [https://github.com/facebookresearch/video-nonlocal-net].
    """

    def __init__(self, inplanes, use_scale=False, **kwargs):
        planes = inplanes // 2
        self.use_scale = use_scale

        super(NonLocal, self).__init__(inplanes)
        self.t = nn.Conv2D(inplanes, planes, kernel_size=1,
                           stride=1, bias_attr=True)
        self.p = nn.Conv2D(inplanes, planes, kernel_size=1,
                           stride=1, bias_attr=True)
        self.g = nn.Conv2D(inplanes, planes, kernel_size=1,
                           stride=1, bias_attr=True)
        self.softmax = nn.Softmax(axis=2)
        self.z = nn.Conv2D(planes, inplanes, kernel_size=1,
                           stride=1, bias_attr=True)
        self.bn = nn.BatchNorm2D(inplanes)

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        b, c, h, w = t.shape

        t = paddle.transpose(paddle.reshape(t, (b, c, -1)), (0, 2, 1))
        p = paddle.reshape(p, (b, c, -1))
        g = paddle.transpose(paddle.reshape(g, (b, c, -1)), (0, 2, 1))
        # print(t.shape, p.shape)
        att = paddle.bmm(t, p)
        # print(att.shape)
        if self.use_scale:
            att = paddle.divide(att, paddle.to_tensor(c**0.5))
        # print(att.shape) # [4, 128, 64, 64] # [4, 64, 128, 128]
        att = self.softmax(att)
        x = paddle.bmm(att, g)

        x = paddle.transpose(x, (0, 2, 1))
        x = paddle.reshape(x, (b, c, h, w))

        x = self.z(x)
        x = self.bn(x) + residual
        # x = x + residual

        return x


class BATransform(nn.Layer):

    def __init__(self, in_channels, s, k):
        super(BATransform, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2D(in_channels, k, 1),
                                   nn.BatchNorm2D(k),
                                   nn.ReLU())
        self.conv_p = nn.Conv2D(k, s * s * k, [s, 1])
        self.conv_q = nn.Conv2D(k, s * s * k, [1, s])
        self.conv2 = nn.Sequential(nn.Conv2D(in_channels, in_channels, 1),
                                   nn.BatchNorm2D(in_channels),
                                   nn.ReLU())
        self.s = s
        self.k = k
        self.in_channels = in_channels

    def extra_repr(self):
        return 'BATransform({in_channels}, s={s}, k={k})'.format(**self.__dict__)

    def resize_mat(self, x, t):
        n, c, s, s1 = x.shape
        assert s == s1
        if t <= 1:
            return x
        x = paddle.reshape(x, (n * c, -1, 1, 1))
        x = x * paddle.eye(t, t, dtype=x.dtype)
        x = paddle.reshape(x, (n * c, s, s, t, t))
        x = paddle.concat(paddle.split(x, 1, axis=1), axis=3)
        x = paddle.concat(paddle.split(x, 1, axis=2), axis=4)
        x = paddle.reshape(x, (n, c, s * t, s * t))
        return x

    def forward(self, x):
        out = self.conv1(x)
        rp = F.adaptive_max_pool2d(out, (self.s, 1))
        cp = F.adaptive_max_pool2d(out, (1, self.s))
        p = paddle.reshape(self.conv_p(rp), (x.shape[0], self.k, self.s, self.s))
        q = paddle.reshape(self.conv_q(cp), (x.shape[0], self.k, self.s, self.s))
        p = F.sigmoid(p)
        q = F.sigmoid(q)
        p = p / paddle.sum(p, axis=3, keepdim=True)
        q = q / paddle.sum(q, axis=2, keepdim=True)

        p = paddle.reshape(p, (x.shape[0], self.k, 1, self.s, self.s))
        p = paddle.expand(p, (x.shape[0], self.k, x.shape[1] // self.k, self.s, self.s))

        p = paddle.reshape(p, (x.shape[0], x.shape[1], self.s, self.s))

        q = paddle.reshape(q, (x.shape[0], self.k, 1, self.s, self.s))
        q = paddle.expand(q, (x.shape[0], self.k, x.shape[1] // self.k, self.s, self.s))

        q = paddle.reshape(q, (x.shape[0], x.shape[1], self.s, self.s))

        p = self.resize_mat(p, x.shape[2] // self.s)
        q = self.resize_mat(q, x.shape[2] // self.s)
        y = paddle.matmul(p, x)
        y = paddle.matmul(y, q)

        y = self.conv2(y)
        return y


class BATBlock(NonLocalModule):

    def __init__(self, in_channels, r=2, s=4, k=4, dropout=0.2, **kwargs):
        super().__init__(in_channels)

        inter_channels = in_channels // r
        self.conv1 = nn.Sequential(nn.Conv2D(in_channels, inter_channels, 1),
                                   nn.BatchNorm2D(inter_channels),
                                   nn.ReLU())
        self.batransform = BATransform(inter_channels, s, k)
        self.conv2 = nn.Sequential(nn.Conv2D(inter_channels, in_channels, 1),
                                   nn.BatchNorm2D(in_channels),
                                   nn.ReLU())
        self.dropout = nn.Dropout2D(p=dropout)

    def forward(self, x):
        xl = self.conv1(x)
        y = self.batransform(xl)
        y = self.conv2(y)
        y = self.dropout(y)
        return y + x

    def init_modules(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                m.weight=m.create_parameter(m.weight.shape, default_initializer=nn.initializer.KaimingNormal())
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.set_value(paddle.ones(m.weight.shape))
                m.bias.set_value(paddle.zeros(m.bias.shape))

class AIDR(nn.Layer):

    def __init__(self, in_channels=3, out_channels=3, num_c=48):
        super(AIDR, self).__init__()
        self.en_block1 = nn.Sequential(
            nn.Conv2D(in_channels, num_c, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(num_c, num_c, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2D(2))

        self.en_block2 = nn.Sequential(
            nn.Conv2D(num_c, num_c, 3, padding=1,bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2D(2))

        self.en_block3 = nn.Sequential(
            nn.Conv2D(num_c, num_c, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2D(2))

        self.en_block4 = nn.Sequential(
            nn.Conv2D(num_c, num_c, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2D(2))

        self.en_block5 = nn.Sequential(
            nn.Conv2D(num_c, num_c, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            NonLocalBlock(num_c),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2D(2),
            nn.Conv2D(num_c, num_c, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            NonLocalBlock(num_c),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block1 = nn.Sequential(
            nn.Conv2D(num_c*2, num_c*2, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            NonLocalBlock(num_c*2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(num_c*2, num_c*2, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2,mode='nearest'))

        self.de_block2 = nn.Sequential(
            nn.Conv2D(num_c*3, num_c*2, 3, padding=1,bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(num_c*2, num_c*2, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2,mode='nearest'))

        self.de_block3 = nn.Sequential(
            nn.Conv2D(num_c*3, num_c*2, 3, padding=1,bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(num_c*2, num_c*2, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block4 = nn.Sequential(
            nn.Conv2D(num_c*3, num_c*2, 3, padding=1,bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(num_c*2, num_c*2, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2,mode='nearest'))

        self.de_block5 = nn.Sequential(
            nn.Conv2D(num_c*2 + in_channels, 64, 3,padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(64, 32, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(32, out_channels, 3, padding=1, bias_attr=True))

    def forward(self, x):
        pool1 = self.en_block1(x)
        pool2 = self.en_block2(pool1)
        pool3 = self.en_block3(pool2)
        pool4 = self.en_block4(pool3)
        upsample5 = self.en_block5(pool4)
        concat5 = paddle.concat((upsample5, pool4), axis=1)
        upsample4 = self.de_block1(concat5)
        concat4 = paddle.concat((upsample4, pool3), axis=1)
        upsample3 = self.de_block2(concat4)
        concat3 = paddle.concat((upsample3, pool2), axis=1)
        upsample2 = self.de_block3(concat3)
        concat2 = paddle.concat((upsample2, pool1), axis=1)
        upsample1 = self.de_block4(concat2)
        concat1 = paddle.concat((upsample1, x), axis=1)
        out = self.de_block5(concat1)
        return out


def load_pretrained_model(model, pretrained_model):
    if pretrained_model is not None:
        print('Loading pretrained model from {}'.format(pretrained_model))

        if os.path.exists(pretrained_model):
            para_state_dict = paddle.load(pretrained_model)

            model_state_dict = model.state_dict()
            keys = model_state_dict.keys()
            num_params_loaded = 0
            for k in keys:
                if k not in para_state_dict:
                    print("{} is not in pretrained model".format(k))
                elif list(para_state_dict[k].shape) != list(
                        model_state_dict[k].shape):
                    print(
                        "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                        .format(k, para_state_dict[k].shape,
                                model_state_dict[k].shape))
                else:
                    model_state_dict[k] = para_state_dict[k]
                    num_params_loaded += 1
            model.set_dict(model_state_dict)
            print("There are {}/{} variables loaded into {}.".format(
                num_params_loaded, len(model_state_dict),
                model.__class__.__name__))

        else:
            raise ValueError(
                'The pretrained model directory is not Found: {}'.format(
                    pretrained_model))
    else:
        print(
            'No pretrained model to load, {} will be trained from scratch.'.
            format(model.__class__.__name__))



def infer(img, model):
    hor = [False, True]
    ver = [False, True]
    trans = [False, True]
    result = np.zeros(img.shape, dtype='float32')
    h_pad = 0
    w_pad = 0
    height, width, c = img.shape
    if height % 64 != 0:
        h_pad = (height // 64 + 1) * 64 - height
    if width % 64 != 0:
        w_pad = (width // 64 + 1) * 64 - width

    img = np.pad(img, ((0, h_pad), (0, w_pad), (0, 0)), "reflect")
    # for h in hor:
    #     for v in ver:
    #         for t in trans:
    #             if h :
    #                 img = img[:, ::-1, :]
    #             if v:
    #                 img = img[::-1, :, :]
    #             if t:
    #                 img = img.transpose(1,0,2)

    #             img_tensor = paddle.to_tensor(img)
    #             img_tensor /= 255.0
                
    #             img_tensor = paddle.transpose(img_tensor, [2, 0, 1])
    #             model.eval()
    #             img_tensor = img_tensor.unsqueeze(0)
    #             img_out = model(img_tensor)
    #             img_out = img_out.squeeze(0).numpy()
                
    #             img_out = img_out.transpose(1, 2, 0)
                
    #             #输出的翻转顺序跟输入相反
    #             if t:
    #                 img_out = img_out.transpose(1,0,2)
    #                 img = img.transpose(1,0,2)
    #             if v:
    #                 img_out = img_out[::-1, :, :]
    #                 img = img[::-1, :, :]
    #             if h:
    #                 img_out = img_out[:, ::-1, :]
    #                 img = img[:, ::-1, :]

    #             #最后再slice
                
    #             img_out = img_out[:height, :width, :]
    #             result += img_out
    # result /= 8
    img_tensor = paddle.to_tensor(img)
    img_tensor /= 255.0
    
    img_tensor = paddle.transpose(img_tensor, [2, 0, 1])
    model.eval()
    img_tensor = img_tensor.unsqueeze(0)
    img_out = model(img_tensor)
    img_out = img_out.squeeze(0).numpy()
    
    img_out = img_out.transpose(1, 2, 0)
    result = img_out[:height, :width, :]
    return result


def main(src_image_dir, save_dir):
    model = AIDR(num_c=96)
    load_pretrained_model(model, 'model.pdparams')
    im_files = sorted(glob.glob(os.path.join(src_image_dir, "*.png")))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    total_time = 0
    with paddle.no_grad():
        for i, im in enumerate(im_files):
            print(im)
            img = cv2.imread(im)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            final_result = np.zeros(img.shape, dtype='float32')
            height, width, c = img.shape
            h1 = int(0.52 * height)
            h2 = int(0.48 * height)
            w1 = int(0.52 * width)
            w2 = int(0.48 * width)

            img1 = img[0:h1, 0:w1]
            img2 = img[0:h1, w2:]
            img3 = img[h2:, 0:w1]
            img4 = img[h2:, w2:]
            
            result1 = infer(img1, model)
            result2 = infer(img2, model)
            result3 = infer(img3, model)
            result4 = infer(img4, model)

            final_result[0:h1, 0:w1] = result1
            final_result[0:h1, w2:w1] = (final_result[0:h1, w2:w1] + result2[0:h1, 0:w1-w2]) / 2.
            final_result[0:h1, w1:] = result2[0:h1, w1-w2:]
            final_result[h2:h1, 0:w1] = (final_result[h2:h1, 0:w1] + result3[0:h1-h2, 0:w1]) / 2.
            final_result[h1:, 0:w1] = result3[h1-h2:, 0:w1]
            final_result[h2:, w2:w1] = (final_result[h2:, w2:w1] + result4[0:, 0:w1-w2]) / 2.
            final_result[h2:h1, w2:] = (final_result[h2:h1, w2:] + result4[0:h1-h2, 0:]) / 2.
            final_result[h1:, w1:,] = result4[h1-h2:, w1-w2:]
            
            final_result = final_result * 255.0
            final_result = final_result.round().clip(0,255).astype(np.uint8)

            
            final_result = cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, im.split('/')[-1]), final_result)


if __name__=='__main__':
    assert len(sys.argv) == 3
    src_image_dir = sys.argv[1]
    save_dir = sys.argv[2]
    main(src_image_dir, save_dir)

