# Based on https://github.com/jahongir7174/YOLOv8
# If you reuse this script, please do credit them as well

import math
import torch
import torchvision
from src.utils.boxOps import boxCxcywh2Xyxy

def make_anchors(x, strides, offset=0.5):
    assert x is not None
    anchor_tensor, stride_tensor = [], []
    dtype, device = x[0].dtype, x[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_tensor), torch.cat(stride_tensor)

def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv

def non_max_suppression(predictions, confidence_threshold=0.001, iou_threshold=0.7, max_det=300):
    max_wh = 7680
    max_nms = 30000

    bs = predictions.shape[0]  # batch size
    nc = predictions.shape[1] - 4  # number of classes
    xc = predictions[:, 4:4 + nc].amax(1) > confidence_threshold  # candidates

    # Settings
    output = [torch.zeros((0, 6), device=predictions.device)] * bs
    for index, x in enumerate(predictions):  # image index, image inference
        x = x.transpose(0, -1)[xc[index]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # matrix nx6 (box, confidence, cls)
        box, cls = x.split((4, nc), 1)
        box = boxCxcywh2Xyxy(box)  # (cx, cy, w, h) to (x1, y1, x2, y2)
        if nc > 1:
            i, j = (cls > confidence_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > confidence_threshold]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes, scores
        indices = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        indices = indices[:max_det]  # limit detections

        output[index] = x[indices]

    return output

class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)
        self.relu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))

class Residual(torch.nn.Module):
    def __init__(self, ch, add=True):
        super().__init__()
        self.add_m = add
        self.conv1 = Conv(ch, ch, k=3, p=1)
        self.conv2 = Conv(ch, ch, k=3, p=1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return x + y if self.add_m else y

class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch)
        self.conv2 = Conv((2 + n) * out_ch // 2, out_ch)
        self.res_m = torch.nn.ModuleList(Residual(out_ch // 2, add) for _ in range(n))

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv2(torch.cat(tensors=y, dim=1))

class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2)
        self.conv2 = Conv(in_ch * 2, out_ch)
        self.res_m = torch.nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat(tensors=[x, y1, y2, self.res_m(y2)], dim=1))

class DarkNet(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        self.p1.append(Conv(width[0], width[1], k=3, s=2, p=1))
        # p2/4
        self.p2.append(Conv(width[1], width[2], k=3, s=2, p=1))
        self.p2.append(CSP(width[2], width[2], depth[0]))
        # p3/8
        self.p3.append(Conv(width[2], width[3], k=3, s=2, p=1))
        self.p3.append(CSP(width[3], width[3], depth[1]))
        # p4/16
        self.p4.append(Conv(width[3], width[4], k=3, s=2, p=1))
        self.p4.append(CSP(width[4], width[4], depth[2]))
        # p5/32
        self.p5.append(Conv(width[4], width[5], k=3, s=2, p=1))
        self.p5.append(CSP(width[5], width[5], depth[0]))
        self.p5.append(SPP(width[5], width[5]))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5

class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[0], add=False)
        self.h2 = CSP(width[3] + width[4], width[3], depth[0], add=False)
        self.h3 = Conv(width[3], width[3], k=3, s=2, p=1)
        self.h4 = CSP(width[3] + width[4], width[4], depth[0], add=False)
        self.h5 = Conv(width[4], width[4], k=3, s=2, p=1)
        self.h6 = CSP(width[4] + width[5], width[5], depth[0], add=False)

    def forward(self, x):
        p3, p4, p5 = x
        p4 = self.h1(torch.cat(tensors=[self.up(p5), p4], dim=1))
        p3 = self.h2(torch.cat(tensors=[self.up(p4), p3], dim=1))
        p4 = self.h4(torch.cat(tensors=[self.h3(p3), p4], dim=1))
        p5 = self.h6(torch.cat(tensors=[self.h5(p4), p5], dim=1))
        return p3, p4, p5

class Head(torch.nn.Module):
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 4  # number of outputs per anchor
        self.stride = torch.zeros(len(filters))  # strides computed during build

        box = max(64, filters[0] // 4)
        cls = max(80, filters[0], self.nc)

        self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, box, k=3, p=1),
                                                           Conv(box, box, k=3, p=1),
                                                           torch.nn.Conv2d(box, out_channels=4,
                                                                           kernel_size=1)) for x in filters)
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, cls, k=3, p=1),
                                                           Conv(cls, cls, k=3, p=1),
                                                           torch.nn.Conv2d(cls, out_channels=self.nc,
                                                                           kernel_size=1)) for x in filters)

    def forward(self, x):
        shape = x[0].shape
        for i, (box, cls) in enumerate(zip(self.box, self.cls)):
            x[i] = torch.cat(tensors=(box(x[i]), cls(x[i])), dim=1)
        if self.training:
            return x
        if self.shape != shape:
            self.shape = shape
            self.anchors, self.strides = (i.transpose(0, 1) for i in make_anchors(x, self.stride))

        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
        box, cls = x.split(split_size=(4, self.nc), dim=1)

        a, b = box.chunk(2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)

        return torch.cat(tensors=(box * self.strides, cls.sigmoid()), dim=1)

    def initialize_biases(self):
        # Initialize biases
        # WARNING: requires stride availability
        for box, cls, s in zip(self.box, self.cls, self.stride):
            # box
            box[-1].bias.data[:] = 1.0
            # cls (.01 objects, 80 classes, 640 image)
            cls[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)

class YOLOv8(torch.nn.Module):
    '''
    Note model output varies based on model state:
        - model.train(): i.e. model.training == True,  will return a list of anchor outputs for all stages (P3, P4, P5)
        - model.eval() : i.e. model.training == False, will return a dict of {labels: [batch, class], scores:[batch, class] , boxes: [cx, cy, w, h]}
    '''
    def __init__(self, args):
        num_classes = args.numClass
        width = args.yolo.width
        depth = args.yolo.depth
        width[0] = args.inChans #Correct to fit different image modalities

        #For inference
        self.conf_thres = args.yolo.conf_thres
        self.iou_thres = args.yolo.iou_thres
        self.max_dets = args.yolo.max_dets

        super().__init__()
        self.net = DarkNet(width, depth)
        self.fpn = DarkFPN(width, depth)

        img_dummy = torch.zeros(1, width[0], 256, 256)
        self.head = Head(num_classes, (width[3], width[4], width[5]))
        self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        x = self.net(x)
        x = self.fpn(x)
        if self.training:
            return self.head(list(x))
        else:
            x = torch.stack(non_max_suppression(self.head(list(x)), self.conf_thres, self.iou_thres, self.max_dets), 0)
            return {"labels": x[:,:,5],
                    "scores": x[:,:,4],
                    "boxes" : x[:,:,:4],
                     }

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self

