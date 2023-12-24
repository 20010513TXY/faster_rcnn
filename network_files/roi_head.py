from typing import Optional, List, Dict, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from . import det_utils
from . import boxes as box_ops


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Arguments:
        class_logits : 预测类别概率信息，shape=[num_anchors, num_classes]
        box_regression : 预测边目标界框回归信息
        labels : 真实类别信息
        regression_targets : 真实目标边界框信息

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    # 计算类别损失信息
    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    # 返回标签类别大于0的索引
    # sampled_pos_inds_subset = torch.nonzero(torch.gt(labels, 0)).squeeze(1)
    # 正样本的索引：tensor([  49,   84,  250,  258,  511,  814, 1022, 1023, 1122, 1189, 1200, 1301,
    #         1328, 1470, 1512, 1534, 1535, 2047, 2154, 2202, 2205, 2234, 2324, 2345,
    #         2506, 2559, 2630, 2717, 3069, 3070, 3071, 3256, 3327, 3344, 3348, 3399,
    #         3418, 3432, 3436, 3471, 3491, 3555, 3558, 3562, 3582, 3583, 3654, 3891,
    #         4094, 4095], device='cuda:0')
    sampled_pos_inds_subset = torch.where(torch.gt(labels, 0))[0]

    # 返回标签类别大于0位置的类别信息
    # 正样本对应的类别：tensor([ 2,  2,  2,  2,  2,  4,  4,  4, 15, 18, 15, 15, 15, 18, 18, 18, 15,  4,
    #         15, 15, 15, 15, 15, 15, 15, 15, 17, 17, 17, 17, 15,  7,  7,  7,  7,  7,
    #          7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 15, 15, 19, 15],
    #        device='cuda:0')
    labels_pos = labels[sampled_pos_inds_subset]

    # shape=[num_proposal, num_classes]
    N, num_classes = class_logits.shape  #torch.Size([4096, 21])
    box_regression = box_regression.reshape(N, -1, 4)
    #torch.Size([4096, 21, 4]) 有4096个proposal，有21个类别，每个类别都对应1个目标回归参数（4维）

    # 计算边界框损失信息
    box_loss = det_utils.smooth_l1_loss(
        # 获取指定索引proposal的指定类别box信息
        box_regression[sampled_pos_inds_subset, labels_pos], #取出正样本对应类别的目标回归参数(预测框)
        regression_targets[sampled_pos_inds_subset], #正样本的真实gtbox
        beta=1 / 9,
        size_average=False,
    ) / labels.numel()
    # 分别得到类别损失信息和边界框损失信息
    return classification_loss, box_loss


class RoIHeads(torch.nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(self,
                 box_roi_pool,   # Multi-scale RoIAlign pooling
                 box_head,       # TwoMLPHead
                 box_predictor,  # FastRCNNPredictor
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,  # default: 0.5, 0.5
                 batch_size_per_image, positive_fraction,  # default: 512, 0.25
                 bbox_reg_weights,  # None
                 # Faster R-CNN inference
                 score_thresh,        # default: 0.05
                 nms_thresh,          # default: 0.5
                 detection_per_img):  # default: 100
        super(RoIHeads, self).__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,  # default: 0.5
            bg_iou_thresh,  # default: 0.5
            allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,  # default: 512
            positive_fraction)     # default: 0.25

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool    # Multi-scale RoIAlign pooling
        self.box_head = box_head            # TwoMLPHead
        self.box_predictor = box_predictor  # FastRCNNPredictor

        self.score_thresh = score_thresh  # default: 0.05
        self.nms_thresh = nms_thresh      # default: 0.5
        self.detection_per_img = detection_per_img  # default: 100

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        """
        为每个proposal匹配对应的gt_box，并划分到正负样本中
        Args:
            proposals:每个图片的proposal+真实的gtbox
            gt_boxes:每个图片真实的gtbox
            gt_labels:每个图片真实的gtbox对应的类别

        Returns:

        """
        matched_idxs = []
        labels = []
        # 遍历每张图像的proposals, gt_boxes, gt_labels信息
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            if gt_boxes_in_image.numel() == 0:  # 该张图像中没有gt框，为背景
                # background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                # 计算proposal与每个gt_box的iou重合度
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)

                # 计算proposal与每个gt_box匹配的iou最大值，并记录索引，
                # iou < low_threshold索引值为 -1， low_threshold <= iou < high_threshold索引值为 -2
                # 记录正负样本的位置 正样本的位置为对应的gtbox（比如第i个proposal是正样本，与gtbox[j]有最大iou值（且iou>0.5），则matched_idxs_in_image[i]=j）;负样本位置为-1
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)  #tensor([-1, -1, -1,  ...,  1,  2,  3], device='cuda:0')

                # 限制最小值，防止匹配标签时出现越界的情况
                # 注意-1, -2对应的gt索引会调整到0,获取的标签类别为第0个gt的类别（实际上并不是）,后续会进一步处理
                # 将负样本位置处的值变为0
                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)  #tensor([0, 0, 0,  ..., 1, 2, 3], device='cuda:0')
                # 获取proposal匹配到的gt对应标签 （负样本会对应gt_labels中的第0个值）
                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # label background (below the low threshold)
                # 将gt索引为-1的类别设置为0，即背景，负样本
                # 将负样本对应的类别改为0，即背景
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_in_image[bg_inds] = 0

                # label ignore proposals (between low and high threshold)
                # 将gt索引为-2的类别设置为-1, 即废弃样本
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        # matched_idxs：记录了每个图片的每个样本对应的gtbox（负样本对应的也是第0个gtbox）  labels:记录了每个图片的每个样本对应的gtbox的类别
        # matched_idxs = [tensor([0, 0, 0,  ..., 0, 0, 0], device='cuda:0'), tensor([0, 0, 0,  ..., 0, 0, 0], device='cuda:0'), tensor([0, 0, 0,  ..., 0, 1, 2], device='cuda:0'), tensor([0, 2, 0,  ..., 1, 2, 3], device='cuda:0'), tensor([0, 0, 0,  ..., 0, 0, 1], device='cuda:0'), tensor([0, 0, 0,  ..., 0, 0, 0], device='cuda:0'), tensor([0, 0, 0,  ..., 0, 1, 2], device='cuda:0'), tensor([0, 0, 0,  ..., 0, 1, 2], device='cuda:0')]
        # labels = [tensor([0, 8, 0,  ..., 0, 0, 8], device='cuda:0'), tensor([0, 0, 0,  ..., 0, 0, 1], device='cuda:0'), tensor([ 0,  0,  0,  ..., 15, 15, 15], device='cuda:0'), tensor([ 0, 18,  0,  ..., 18, 18, 16], device='cuda:0'), tensor([ 0,  0,  0,  ...,  0, 12, 12], device='cuda:0'), tensor([ 0,  0,  0,  ...,  0,  0, 19], device='cuda:0'), tensor([ 0,  0,  0,  ..., 17, 17, 17], device='cuda:0'), tensor([ 0,  0,  0,  ..., 12, 12, 12], device='cuda:0')]
        return matched_idxs, labels


    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        # BalancedPositiveNegativeSampler
        # # 得到一个batch中所有图片上被选择的正样本和负样本  对应两个蒙版列表，一个记录被选择的正样本，一个记录被选择的负样本
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        # 遍历每张图片的正负样本索引
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            # 记录所有采集样本索引（包括正样本和负样本）
            # img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        # 得到一个batch中所有被选择的样本
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        """
        将每个图片的gt_boxes拼接到proposal后面
        Args:
            proposals: 一个batch中每张图像rpn预测的boxes
            gt_boxes:  一个batch中每张图像对应的真实目标边界框

        Returns:

        """
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]
        return proposals

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        assert targets is not None
        assert all(["boxes" in t for t in targets])
        assert all(["labels" in t for t in targets])

    def select_training_samples(self,
                                proposals,  # type: List[Tensor]
                                targets     # type: Optional[List[Dict[str, Tensor]]]
                                ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        """
        划分正负样本，统计对应gt的标签以及边界框回归信息
        list元素个数为batch_size
        Args:
            proposals: rpn预测的boxes
            targets:

        Returns:

        """

        # 检查target数据是否为空
        self.check_targets(targets)
        # 如果不加这句，jit.script会不通过(看不懂)
        assert targets is not None

        dtype = proposals[0].dtype
        device = proposals[0].device

        # 获取图片真实的boxes以及labels信息
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to proposal
        # 将gt_boxes拼接到proposal后面  正样本很少，将gtbox添加到proposal充当正样本
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        # 为每个proposal匹配对应的gt_box，并划分到正负样本中
        # matched_idxs：记录了每个图片的每个样本对应的gtbox（负样本对应的也是第0个gtbox）  labels:记录了每个图片的每个样本对应的gtbox的类别
        # matched_idxs = [tensor([0, 0, 0,  ..., 0, 0, 0], device='cuda:0'), tensor([0, 0, 0,  ..., 0, 0, 0], device='cuda:0'), tensor([0, 0, 0,  ..., 0, 1, 2], device='cuda:0'), tensor([0, 2, 0,  ..., 1, 2, 3], device='cuda:0'), tensor([0, 0, 0,  ..., 0, 0, 1], device='cuda:0'), tensor([0, 0, 0,  ..., 0, 0, 0], device='cuda:0'), tensor([0, 0, 0,  ..., 0, 1, 2], device='cuda:0'), tensor([0, 0, 0,  ..., 0, 1, 2], device='cuda:0')]
        # labels = [tensor([0, 8, 0,  ..., 0, 0, 8], device='cuda:0'), tensor([0, 0, 0,  ..., 0, 0, 1], device='cuda:0'), tensor([ 0,  0,  0,  ..., 15, 15, 15], device='cuda:0'), tensor([ 0, 18,  0,  ..., 18, 18, 16], device='cuda:0'), tensor([ 0,  0,  0,  ...,  0, 12, 12], device='cuda:0'), tensor([ 0,  0,  0,  ...,  0,  0, 19], device='cuda:0'), tensor([ 0,  0,  0,  ..., 17, 17, 17], device='cuda:0'), tensor([ 0,  0,  0,  ..., 12, 12, 12], device='cuda:0')]
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        # 按给定数量和比例采样正负样本  并不是使用所有的样本，而是需要对样本进行采样
        sampled_inds = self.subsample(labels) # 得到一个batch中所有被选择的样本  测试时：每个图片512个样本
        matched_gt_boxes = []
        num_images = len(proposals)

        # 遍历每张图像
        for img_id in range(num_images):
            # 获取每张图像的正负样本索引
            img_sampled_inds = sampled_inds[img_id]
            # 第0张图片的索引：tensor([   8,   10,   15,   19,  1654, 1655, 1656, 1657, 1658], device='cuda:0')
            # 获取对应正负样本的proposals信息
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            # tensor([[150.3034, 472.5213, 248.5049, 483.5397],
            #         [740.9252, 696.5301, 803.1105, 720.0422],
            #         ...,
            #         [  0.0000, 466.2577,  51.4920, 527.6074]], device='cuda:0')
            # 获取对应正负样本的真实类别信息
            labels[img_id] = labels[img_id][img_sampled_inds]
            # tensor([ 0,  0,  0,  0,  ...  0,  0,  0, 19], device='cuda:0')
            # 获取对应正负样本的gt索引信息
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
            # tensor([0, 0, 0, 0, ... 0, 0], device='cuda:0')

            gt_boxes_in_image = gt_boxes[img_id] #第img_id张图片的gtbox  tensor([[252.2100, 377.1772, 879.1320, 605.4054]], device='cuda:0')
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            # 获取对应正负样本的gt box信息
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        # 根据gt和proposal计算边框回归参数（针对gt的）  得到调整后的proposal
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        # proposals：256个被选择的proposal  labels：256个被选择的proposal对应的gtbox类别
        # regression_targets：256个被选择的proposal经过gtbox调整后
        return proposals, labels, regression_targets

    def postprocess_detections(self,
                               class_logits,    # type: Tensor
                               box_regression,  # type: Tensor
                               proposals,       # type: List[Tensor]
                               image_shapes     # type: List[Tuple[int, int]]
                               ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        """
        对网络的预测数据进行后处理，包括
        （1）根据proposal以及预测的回归参数计算出最终bbox坐标
        （2）对预测类别结果进行softmax处理
        （3）裁剪预测的boxes信息，将越界的坐标调整到图片边界上
        （4）移除所有背景信息
        （5）移除低概率目标
        （6）移除小尺寸目标
        （7）执行nms处理，并按scores进行排序
        （8）根据scores排序返回前topk个目标
        Args:
            class_logits: 网络预测类别概率信息
            box_regression: 网络预测的边界框回归参数
            proposals: rpn输出的proposal
            image_shapes: 打包成batch前每张图像的宽高

        Returns:

        """
        device = class_logits.device
        # 预测目标类别数
        num_classes = class_logits.shape[-1]

        # 获取每张图像的预测bbox数量
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        # 根据proposal以及预测的回归参数计算出最终bbox坐标
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        # 对预测类别结果进行softmax处理
        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image
        # 根据每张图像的预测bbox数量分割结果
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        # 遍历每张图像预测信息
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            # 裁剪预测的boxes信息，将越界的坐标调整到图片边界上
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove prediction with the background label
            # 移除索引为0的所有信息（0代表背景）
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            # 移除低概率目标，self.scores_thresh=0.05
            # gt: Computes input > other element-wise.
            # inds = torch.nonzero(torch.gt(scores, self.score_thresh)).squeeze(1)
            inds = torch.where(torch.gt(scores, self.score_thresh))[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            # 移除小目标
            keep = box_ops.remove_small_boxes(boxes, min_size=1.)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximun suppression, independently done per class
            # 执行nms处理，执行后的结果会按照scores从大到小进行排序返回
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)

            # keep only topk scoring predictions
            # 获取scores排在前topk个预测目标
            keep = keep[:self.detection_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(self,
                features,       # type: Dict[str, Tensor]
                proposals,      # type: List[Tensor]
                image_shapes,   # type: List[Tuple[int, int]]
                targets=None    # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """

        # 检查targets的数据类型是否正确
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"
                assert t["labels"].dtype == torch.int64, "target labels must of int64 type"

        if self.training:
            # 划分正负样本，统计对应gt的标签以及边界框回归信息
            # proposals：经nms处理后的proposal  targets：图像的xml信息
            # proposals:正负样本的proposal  labels：正负样本的对应gtbox的类别, regression_targets:proposal调整之后的坐标
            proposals, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None

        # 将采集样本通过Multi-scale RoIAlign pooling层
        # box_features_shape: [num_proposals, channel, height, width]
        box_features = self.box_roi_pool(features, proposals, image_shapes)

        # 通过roi_pooling后的两层全连接层
        # box_features_shape: [num_proposals, representation_size]
        box_features = self.box_head(box_features)

        # 接着分别预测目标类别和边界框回归参数
        class_logits, box_regression = self.box_predictor(box_features)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            # 分别得到类别损失信息和边界框损失信息
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )
        # 训练时,得到类别损失和边界框损失信息
        return result, losses
