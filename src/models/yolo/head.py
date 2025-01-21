import torch
import torch.nn.functional as F
from torch import nn

from . import box_ops


class Head(nn.Module):
    def __init__(self, predictor, anchors, strides, 
                 match_thresh, giou_ratio, loss_weights, 
                 score_thresh, nms_thresh, detections):
        super().__init__()
        self.predictor = predictor
        self.register_buffer("anchors", torch.Tensor(anchors))
        self.strides = strides
        
        self.match_thresh = match_thresh
        self.giou_ratio = giou_ratio
  
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections = detections
        
        self.merge = False
        #self.min_size = 2
        
    def forward(self, features, image_shapes=None, scale_factors=None, max_size=None):
        preds = self.predictor(features)
        return self.inference(preds, image_shapes, scale_factors, max_size)
    
    def inference(self, preds, image_shapes, scale_factors, max_size):
        ids, ps, boxes = [], [], []
        for pred, stride, wh in zip(preds, self.strides, self.anchors): # 3.54s
            pred = torch.sigmoid(pred)
            n, y, x, a = torch.where(pred[..., 4] > self.score_thresh)
            p = pred[n, y, x, a]
            
            xy = torch.stack((x, y), dim=1)
            xy = (2 * p[:, :2] - 0.5 + xy) * stride
            wh = 4 * p[:, 2:4] ** 2 * wh[a]
            box = torch.cat((xy, wh), dim=1)
            
            ids.append(n)
            ps.append(p)
            boxes.append(box)
            
        ids = torch.cat(ids)
        ps = torch.cat(ps)
        boxes = torch.cat(boxes)
        
        boxes = box_ops.cxcywh2xyxy(boxes)
        logits = ps[:, [4]] * ps[:, 5:]
        indices, labels = torch.where(logits > self.score_thresh) # 4.94s
        ids, boxes, scores = ids[indices], boxes[indices], logits[indices, labels]
        
        results = []
        for i, im_s in enumerate(image_shapes): # 20.97s
            keep = torch.where(ids == i)[0] # 3.11s
            box, label, score = boxes[keep], labels[keep], scores[keep]
            #ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1] # 0.27s
            #keep = torch.where((ws >= self.min_size) & (hs >= self.min_size))[0] # 3.33s
            #boxes, objectness, logits = boxes[keep], objectness[keep], logits[keep] # 0.36s
            
            if len(box) > 0:
                box[:, 0].clamp_(0, im_s[1]) # 0.39s
                box[:, 1].clamp_(0, im_s[0]) #~
                box[:, 2].clamp_(0, im_s[1]) #~
                box[:, 3].clamp_(0, im_s[0]) #~
                
                keep = box_ops.batched_nms(box, score, label, self.nms_thresh, max_size) # 4.43s
                keep = keep[:self.detections]
                
                nms_box, nms_label = box[keep], label[keep]
                if self.merge: # slightly increase AP, decrease speed ~14%
                    mask = nms_label[:, None] == label[None]
                    iou = (box_ops.box_iou(nms_box, box) * mask) > self.nms_thresh # 1.84s
                    weights = iou * score[None] # 0.14s
                    nms_box = torch.mm(weights, box) / weights.sum(1, keepdim=True) # 0.55s
                    
                box, label, score = nms_box / scale_factors[i], nms_label, score[keep] # 0.30s
            results.append({"bbox": box, "class": label, "conf": score}) # boxes format: (xmin, ymin, xmax, ymax)
            
        return results
    