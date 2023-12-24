import torch
import math
from typing import List, Tuple
from torch import Tensor


class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        # type: (int, float) -> None
        """
        roi时：batch_size_per_image：512  positive_fraction：0.25
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentage of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        """
        Arguments:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        # 遍历每张图像的matched_idxs matched_idxs：每个anchor所对应的正/负/丢弃样本标签 如果是正，则为1，如果是负，则为0，如果是丢弃，则为-1
        for matched_idxs_per_image in matched_idxs:
            # 在roi训练过程中，matched_idxs对应的是每个proposal的labels
            # >= 1的为正样本, nonzero返回非零元素索引
            # positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            # 得到所有正样本对应的位置 torch.Size([10])  tensor([9159, 9174, 9189, 9204, 9219, 9759, 9774, 9789, 9804, 9819], device='cuda:0')
            positive = torch.where(torch.ge(matched_idxs_per_image, 1))[0]  #roi训练时，matched_idxs_per_image≥1,则为正样本
            # = 0的为负样本
            # negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)
            # 得到所有负样本对应的位置 torch.Size([14359])  tensor([0, 1, 2, ..., 14997, 14998, 14999], device='cuda:0')
            negative = torch.where(torch.eq(matched_idxs_per_image, 0))[0] #roi训练时，matched_idxs_per_image≤0,则为负样本

            # 指定正样本的数量
            num_pos = int(self.batch_size_per_image * self.positive_fraction) #128
            # protect against not enough positive examples
            # 如果正样本数量不够就直接采用所有正样本
            num_pos = min(positive.numel(), num_pos) #min(10,128) = 10
            # 指定负样本数量
            num_neg = self.batch_size_per_image - num_pos #256-10=246
            # protect against not enough negative examples
            # 如果负样本数量不够就直接采用所有负样本
            num_neg = min(negative.numel(), num_neg) #min(14359,246) = 246

            # randomly select positive and negative examples
            # Returns a random permutation of integers from 0 to n - 1.
            # 随机选择指定数量的正负样本
            # 从正样本中随机选择num_pos个正样本，用于计算损失
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            # 从负样本中随机选择num_neg个负样本，用于计算损失
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1] #得到随机选择的num_pos个正样本的下标
            neg_idx_per_image = negative[perm2] #得到随机选择的num_neg个负样本的下标

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )

            pos_idx_per_image_mask[pos_idx_per_image] = 1 #在正样本蒙版上的被选择的正样本下标处标为1
            neg_idx_per_image_mask[neg_idx_per_image] = 1 #在负样本蒙版上的被选择的负样本下标处标为1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)
        # 得到一个batch中所有图片上被选择的正样本和负样本
        return pos_idx, neg_idx


@torch.jit._script_if_tracing
def encode_boxes(reference_boxes, proposals, weights):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Encode a set of proposals with respect to some
    reference boxes

    Arguments:
        reference_boxes (Tensor): reference boxes(gt)
        proposals (Tensor): boxes to be encoded(anchors)
        weights:
    """

    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    # unsqueeze()
    # Returns a new tensor with a dimension of size one inserted at the specified position.
    proposals_x1 = proposals[:, 0].unsqueeze(1) #anchor的xmin
    proposals_y1 = proposals[:, 1].unsqueeze(1) #anchor的ymin
    proposals_x2 = proposals[:, 2].unsqueeze(1) #anchor的xmax
    proposals_y2 = proposals[:, 3].unsqueeze(1) #anchor的ymax

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1) #gtbox的xmin
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1) #gtbox的ymin
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1) #gtbox的xmax
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1) #gtbox的ymax

    # implementation starts here
    # parse widths and heights
    ex_widths = proposals_x2 - proposals_x1 #anchor的宽
    ex_heights = proposals_y2 - proposals_y1
    # parse coordinate of center point
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths #anchor的中心x坐标
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1 #gtbox的宽
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths #gtbox的中心x坐标
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights
    # 这里是根据真实的gtbox来调整anchor样本，然后得到调整后的anchor
    # （并不是通过预测回归参数与anchor相结合，得到proposal）
    # tx* = (x* - xa)/wa  ty* = (y* - ya)/ha tw* = log(w*/wa)  th* = log(h*/ha)
    # tx*:anchor与对应gtbox关于中心坐标x的回归参数  x*：gtbox的x坐标  xa：anchor的x坐标   wa：anchor的宽度
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    # 得到每个anchor关于gtbox的调整参数
    return targets


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        # type: (Tuple[float, float, float, float], float) -> None
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        """
        结合anchors和与之对应的gt计算regression参数
        Args:
            reference_boxes: List[Tensor] 每个proposal/anchor对应的gt_boxes
            proposals: List[Tensor] anchors/proposals

        Returns: regression parameters

        """
        # 统计每张图像的anchors个数，方便后面拼接在一起处理后在分开
        # reference_boxes和proposal数据结构相同
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0) #将一个batch中的图片样本所对应的gtbox拼接到一起
        proposals = torch.cat(proposals, dim=0) #将一个batch中的图片所对应的anchor也拼接到一起
        # 每个proposal对应一个gtbox
        # targets_dx, targets_dy, targets_dw, targets_dh
        # 得到每个anchor关于gtbox的调整参数
        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)

        return targets

    def decode(self, rel_codes, boxes):
        # type: (Tensor, List[Tensor]) -> Tensor
        """
        # rel_codes：一个batch中所有的anchor的回归参数预测(8*21420, 4)
        # boxes：一个batch中所有生成的anchors
        Args:
            rel_codes: bbox regression parameters
            boxes: anchors/proposals

        Returns:

        """
        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)
        boxes_per_image = [b.size(0) for b in boxes] #每张图片对应的anchor：[14250, 14250, 14250, 14250, 14250, 14250, 14250, 14250]
        concat_boxes = torch.cat(boxes, dim=0)  #torch.Size([114000, 4])

        box_sum = 0 #每个batch中所有anchor：114000
        for val in boxes_per_image:
            box_sum += val

        # 将预测的bbox回归参数应用到对应anchors上得到预测bbox的坐标
        # 得到了每个proposal的坐标
        pred_boxes = self.decode_single(
            rel_codes, concat_boxes
        )

        # 防止pred_boxes为空时导致reshape报错
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4) #torch.Size([114000, 1, 4])

        return pred_boxes

    def decode_single(self, rel_codes, boxes):
        """
        rel_codes：一个batch中，所有anchor的预测回归参数
        boxes：一个batch中所有的anchor

        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes (bbox regression parameters)
            boxes (Tensor): reference boxes (anchors/proposals)
        """
        boxes = boxes.to(rel_codes.dtype)

        # xmin, ymin, xmax, ymax
        widths = boxes[:, 2] - boxes[:, 0]   # anchor/proposal宽度
        heights = boxes[:, 3] - boxes[:, 1]  # anchor/proposal高度
        ctr_x = boxes[:, 0] + 0.5 * widths   # anchor/proposal中心x坐标
        ctr_y = boxes[:, 1] + 0.5 * heights  # anchor/proposal中心y坐标

        wx, wy, ww, wh = self.weights  # RPN中为[1,1,1,1], fastrcnn中为[10,10,5,5]
        dx = rel_codes[:, 0::4] / wx   # 中心坐标x回归参数   0::4用切片方式可以获得两个维度，如[114000,1]
        dy = rel_codes[:, 1::4] / wy   # 中心坐标y回归参数
        dw = rel_codes[:, 2::4] / ww   # 宽度回归参数
        dh = rel_codes[:, 3::4] / wh   # 高度回归参数

        # limit max value, prevent sending too large values into torch.exp()
        # self.bbox_xform_clip=math.log(1000. / 16)   4.135
        dw = torch.clamp(dw, max=self.bbox_xform_clip) #将dw中的每个元素都限制在不超过self.bbox_xform_clip的范围内
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        # 将anchor坐标与调整（回归）参数相结合，得到proposal的中心x、y和高、宽
        # tx = (x-xa)/wa  ty = (y-ya)/ha    x = (tx * wa) + xa   y = (ty * ha) + ya
        # x、y:最终proposal坐标  tx、ty：x、y的回归参数  xa、ya:anchor的中心坐标 wa、ha:anchor的宽、高
        # tw = log(w/wa)  th = log(h/ha)    w = e^tw * wa  h = e^th * ha
        # w、h:最终proposal的宽高  th、tw：h、w的回归参数
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        # 根据得到proposal的中心x、y,以及高宽，得到proposal的xmin、ymin、xmax、ymax
        # xmin  torch.Size([114000, 1])
        pred_boxes1 = pred_ctr_x - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        # ymin
        pred_boxes2 = pred_ctr_y - torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        # xmax
        pred_boxes3 = pred_ctr_x + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        # ymax
        pred_boxes4 = pred_ctr_y + torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h

        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
        # torch.Size([114000, 4]) 在xmin、ymin、xmax、ymax的第二个维度进行拼接
        # 得到一个batch中所有proposal的xmin、ymin、xmax、ymax
        return pred_boxes


class Matcher(object):
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    __annotations__ = {
        'BELOW_LOW_THRESHOLD': int,
        'BETWEEN_THRESHOLDS': int,
    }

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        # type: (float, float, bool) -> None
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold  # 0.7
        self.low_threshold = low_threshold    # 0.3
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        match_quality_matrix：anchor与gtbox的iou矩阵
        计算anchors与每个gtboxes匹配的iou最大值，并记录索引，
        iou<low_threshold索引值为-1， low_threshold<=iou<high_threshold索引值为-2
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        # M x N 的每一列代表一个anchors与所有gt的匹配iou值
        # matched_vals代表每列的最大值，即每个anchors与所有gt匹配的最大iou值
        # 由于现在这张图片只有一个gtbox，所以每列数据只有一个值，所以matched_vals的内容和match_quality_matrix的内容一样，而且最大值都在每列的第0个位置，所以matches全为0
        # 如果图片不只一个gtbox，则结果不是这样
        # matches记录每列最大值所在的位置（从0开始）
        matched_vals, matches = match_quality_matrix.max(dim=0)  # match_quality_matrix在维度0的方向求最大值 the dimension to reduce.
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None

        # Assign candidate matches with low quality to negative (unassigned) values
        # 计算iou小于low_threshold的索引
        below_low_threshold = matched_vals < self.low_threshold
        # 计算iou在low_threshold与high_threshold之间的索引值
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold
        )
        # iou小于low_threshold的matches索引置为-1
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD  # -1

        # iou在[low_threshold, high_threshold]之间的matches索引置为-2
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS    # -2

        if self.allow_low_quality_matches:
            # 是否启用第一条正样本匹配准则：当有gtbox没有被anchor匹配到时，将与gtbox有最大iou的anchor作为正样本
            assert all_matches is not None
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)
        # matches中，所有正样本下标为匹配到的gtbox的下标，所有负样本下标为-1，所有丢弃样本下标为-2
        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # For each gt, find the prediction with which it has highest quality
        # highest_quality_foreach_gt为匹配到的最大iou值
        # 找到iou矩阵中每行最大的值  找到与每个gtboxes有iou最大的anchor
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)  # the dimension to reduce.

        # Find highest quality match available, even if it is low, including ties
        # gt_pred_pairs_of_highest_quality = torch.nonzero(
        #     match_quality_matrix == highest_quality_foreach_gt[:, None]
        # )
        # 寻找每个gt boxes与其iou最大的anchor索引，一个gt匹配到的最大iou可能有多个anchor
        gt_pred_pairs_of_highest_quality = torch.where(
            torch.eq(match_quality_matrix, highest_quality_foreach_gt[:, None])
        )
        # gt_pred_pairs_of_highest_quality = (tensor([0, 1], device='cuda:0'), tensor([10488,  9387], device='cuda:0'))
        # 表示：与第一个gtbox有最大IOU的是第10488个anchor   与第二个gtbox有最大IOU的是第9387个anchor
        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        # gt_pred_pairs_of_highest_quality[:, 0]代表是对应的gt index(不需要)
        # pre_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        pre_inds_to_update = gt_pred_pairs_of_highest_quality[1] #tensor([10488,  9387], device='cuda:0')
        # 如果某个anchor与gtbox有最大的iou值，则将这个anchor作为正样本，将它的下标更新到matches中，这样就不会有gtbox没有被anchor匹配到
        matches[pre_inds_to_update] = all_matches[pre_inds_to_update]


def smooth_l1_loss(input, target, beta: float = 1. / 9, size_average: bool = True):
    """
    #input：正样本的预测回归参数
    #target：正样本的真实回归参数
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)  #被选择的正样本的预测回归参数与真实回归参数之间的差距
    # cond = n < beta
    cond = torch.lt(n, beta)
    # tensor([[False, False, False, False],
    #         [False, False, False, False],
    #         ...,
    #         [False,  True, False, False],
    #         [ True, False, False, False]], device='cuda:0')
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta) #如果cond中的元素为True，选择0.5 * n ** 2 / beta，否则选择n - 0.5 * beta
    if size_average:
        return loss.mean()
    return loss.sum()
