import numpy as np
from numpy import sin, cos, tan, pi, arcsin, arctan
from functools import lru_cache
import torch
from torch import nn
from torch.nn.parameter import Parameter

@lru_cache(None)
def get_xy(delta_phi, delta_theta):
    return np.array([
        [
            (-tan(delta_theta), 1 / cos(delta_theta) * tan(delta_phi)),
            (0, tan(delta_phi)),
            (tan(delta_theta), 1 / cos(delta_theta) * tan(delta_phi)),
        ],
        [
            (-tan(delta_theta), 0),
            (1, 1),
            (tan(delta_theta), 0),
        ],
        [
            (-tan(delta_theta), -1 / cos(delta_theta) * tan(delta_phi)),
            (0, -tan(delta_phi)),
            (tan(delta_theta), -1 / cos(delta_theta) * tan(delta_phi)),
        ]
    ])


@lru_cache(None)
def cal_index(h, w, img_r, img_c):
    '''
        Calculate Kernel Sampling Pattern
        only support 3x3 filter
        return 9 locations: (3, 3, 2)
    '''
    # pixel -> rad
    phi = -((img_r + 0.5) / h * pi - pi / 2)
    theta = (img_c + 0.5) / w * 2 * pi - pi

    delta_phi = pi / h
    delta_theta = 2 * pi / w

    xys = get_xy(delta_phi, delta_theta)
    x = xys[..., 0]
    y = xys[..., 1]
    rho = np.sqrt(x ** 2 + y ** 2)
    v = arctan(rho)
    new_phi = arcsin(cos(v) * sin(phi) + y * sin(v) * cos(phi) / rho)
    new_theta = theta + arctan(x * sin(v) / (rho * cos(phi) * cos(v) - y * sin(phi) * sin(v)))
    # rad -> pixel
    new_r = (-new_phi + pi / 2) * h / pi - 0.5
    new_c = (new_theta + pi) * w / 2 / pi - 0.5
    # indexs out of image, equirectangular leftmost and rightmost pixel is adjacent
    new_c = (new_c + w) % w
    new_result = np.stack([new_r, new_c], axis=-1)
    new_result[1, 1] = (img_r, img_c)
    return new_result


@lru_cache(None)
def _gen_filters_coordinates(h, w, stride):
    co = np.array([[cal_index(h, w, i, j) for j in range(0, w, stride)] for i in range(0, h, stride)])
    return np.ascontiguousarray(co.transpose([4, 0, 1, 2, 3]))


def gen_filters_coordinates(h, w, stride=1):
    '''
    return np array of kernel lo (2, H/stride, W/stride, 3, 3)
    '''
    assert (isinstance(h, int) and isinstance(w, int))
    return _gen_filters_coordinates(h, w, stride).copy()


def gen_grid_coordinates(h, w, stride=1):
    coordinates = gen_filters_coordinates(h, w, stride).copy()
    coordinates[0] = (coordinates[0] * 2 / h) - 1
    coordinates[1] = (coordinates[1] * 2 / w) - 1
    coordinates = coordinates[::-1]
    coordinates = coordinates.transpose(1, 3, 2, 4, 0)
    sz = coordinates.shape
    coordinates = coordinates.reshape(1, sz[0] * sz[1], sz[2] * sz[3], sz[4])

    return coordinates.copy()


class SphereConv2D(nn.Module):
    '''  SphereConv2D
    Note that this layer only support 3x3 filter
    '''

    def __init__(self, in_c, out_c, stride=1, bias=True, mode='bilinear'):
        super(SphereConv2D, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.mode = mode
        self.weight = Parameter(torch.Tensor(out_c, in_c, 3, 3))
        if bias:
            self.bias = Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)
        self.grid_shape = None
        self.grid = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        if self.grid_shape is None or self.grid_shape != tuple(x.shape[2:4]):
            self.grid_shape = tuple(x.shape[2:4])
            coordinates = gen_grid_coordinates(x.shape[2], x.shape[3], self.stride)
            with torch.no_grad():
                self.grid = torch.FloatTensor(coordinates).to(x.device)
                self.grid.requires_grad = True

        with torch.no_grad():
            grid = self.grid.repeat(x.shape[0], 1, 1, 1)

        x = nn.functional.grid_sample(x, grid, mode=self.mode, align_corners=True)
        x = nn.functional.conv2d(x, self.weight, self.bias, stride=3)
        return x


class SphereMaxPool2D(nn.Module):
    '''  SphereMaxPool2D
    Note that this layer only support 3x3 filter
    '''

    def __init__(self, stride=1, mode='bilinear'):
        super(SphereMaxPool2D, self).__init__()
        self.stride = stride
        self.mode = mode
        self.grid_shape = None
        self.grid = None
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)

    def forward(self, x):
        if self.grid_shape is None or self.grid_shape != tuple(x.shape[2:4]):
            self.grid_shape = tuple(x.shape[2:4])
            coordinates = gen_grid_coordinates(x.shape[2], x.shape[3], self.stride)
            with torch.no_grad():
                self.grid = torch.FloatTensor(coordinates).to(x.device)
                self.grid.requires_grad = True

        with torch.no_grad():
            grid = self.grid.repeat(x.shape[0], 1, 1, 1)

        return self.pool(nn.functional.grid_sample(x, grid, mode=self.mode, align_corners=True))


class Sphere_CNN(nn.Module):
    def __init__(self, out_put_dim, out_h=128, out_w=256, patch_size=(8, 8)):
        super(Sphere_CNN, self).__init__()
        self.output_dim = out_put_dim
        # self.coord_conv = AddCoordsTh(x_dim=128, y_dim=256, with_r=False)

        # Image pipeline
        self.image_conv1 = SphereConv2D(3, 64, stride=2, bias=False)
        # self.image_conv1 = SphereConv2D(5, 64, stride=2, bias=False)
        self.image_norm1 = nn.BatchNorm2d(64)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.image_conv2 = SphereConv2D(64, 128, stride=2, bias=False)
        self.image_norm2 = nn.BatchNorm2d(128)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.image_conv3 = SphereConv2D(128, 256, stride=2, bias=False)
        self.image_norm3 = nn.BatchNorm2d(256)
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.image_conv3_5 = SphereConv2D(256, 512, stride=2, bias=False)
        self.image_norm3_5 = nn.BatchNorm2d(512)
        self.leaky_relu3_5 = nn.LeakyReLU(0.2, inplace=True)

        # Joint pipeline

        # self.image_conv4 = nn.Conv2d(512, 256, 4, 2, 1, bias=False)
        # self.image_norm4 = nn.BatchNorm2d(256)
        # self.leaky_relu4 = nn.LeakyReLU(0.2, inplace=True)
        #
        # self.image_conv5 = nn.Conv2d(256, 64, 4, 2, 1, bias=False)
        # self.image_norm5 = nn.BatchNorm2d(64)
        # self.leaky_relu5 = nn.LeakyReLU(0.2, inplace=True)

        # self.fc1 = nn.Linear(64 * 4 * 2, self.output_dim)
        # self.flatten = nn.Flatten()
        # self.activation = nn.Tanh()

        self.image_h = out_h
        self.image_w = out_w
        self.pool2d = nn.AvgPool2d(patch_size)

    def upsampling(self, x):
        m = nn.UpsamplingBilinear2d(size=(self.image_h, self.image_w))
        x = m(x)
        return x

    def forward(self, image):
        x = image  # (b,c,h,w)

        # x = self.coord_conv(x)  # (b,5,h,w)   #是否加入坐标卷积

        x1 = self.leaky_relu1(self.image_norm1(self.image_conv1(x)))  # (b,64,64,128)

        x2 = self.leaky_relu2(self.image_norm2(self.image_conv2(x1)))  # (b,128,32,64)

        x3 = self.leaky_relu3(self.image_norm3(self.image_conv3(x2)))  # (b,256,16,32)

        x4 = self.leaky_relu3_5(self.image_norm3_5(self.image_conv3_5(x3)))  # (b,512,8,16)

        # x5 = self.leaky_relu4(self.image_norm4(self.image_conv4(x4))) #256
        #
        # x6 = self.leaky_relu5(self.image_norm5(self.image_conv5(x5))) #64
        #
        # x7 = self.activation(self.fc1(self.flatten(x6)))

        if self.output_dim == 192:
            x1 = self.upsampling(x1)
            x2 = self.upsampling(x2)
            features = torch.cat((x1, x2), dim=1)
        elif self.output_dim == 384:
            x2 = self.upsampling(x2)
            x3 = self.upsampling(x3)
            features = torch.cat((x2, x3), dim=1)
        elif self.output_dim == 448:
            x1 = self.upsampling(x1)
            x2 = self.upsampling(x2)
            x3 = self.upsampling(x3)
            features = torch.cat((x1, x2, x3), dim=1)
        elif self.output_dim == 320:
            x1 = self.upsampling(x1)
            x3 = self.upsampling(x4)
            features = torch.cat((x1, x3), dim=1)
        elif self.output_dim == 576:
            x1 = self.upsampling(x1)
            x4 = self.upsampling(x4)
            features = torch.cat((x1, x4), dim=1)

        if self.output_dim != 512:
            features = self.pool2d(features)
            enc_inputs = features.flatten(2).permute(0, 2, 1)
        else:
            enc_inputs = x4.flatten(2).permute(0, 2, 1)
        return enc_inputs


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    # test cnn
    cnn = SphereConv2D(3, 5, 1)
    out = cnn(torch.randn(2, 3, 10, 10))
    print('SphereConv2D(3, 5, 1) output shape: ', out.size())
    # test pool
    # create sample image
    h, w = 100, 200
    img = np.ones([h, w, 3])
    for r in range(h):
        for c in range(w):
            img[r, c, 0] = img[r, c, 0] - r / h
            img[r, c, 1] = img[r, c, 1] - c / w
    plt.imsave('./image/demo_original.png', img)
    img = img.transpose([2, 0, 1])
    img = np.expand_dims(img, 0)  # (B, C, H, W)
    # pool
    pool = SphereMaxPool2D(1)
    out = pool(torch.from_numpy(img).float())
    out = torch.clamp(out, min=0, max=1)
    out = np.squeeze(out.numpy(), 0).transpose([1, 2, 0])

    plt.imsave('./image/demo_pool_1.png', out)
    print('Save image after pooling with stride 1: demo_pool_1.png')
    # pool with tride 3
    pool = SphereMaxPool2D(3)
    out = pool(torch.from_numpy(img).float())
    out = np.squeeze(out.numpy(), 0).transpose([1, 2, 0])
    plt.imsave('./image/demo_pool_3.png', out)
    print('Save image after pooling with stride 3: demo_pool_3.png')
