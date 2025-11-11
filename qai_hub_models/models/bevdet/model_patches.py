# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import copy

import torch
from mmengine.model import BaseModule
from torch import nn

from qai_hub_models.extern.mmcv import patch_mmcv_no_extensions
from qai_hub_models.utils.optimization import optimized_cumsum

with patch_mmcv_no_extensions():
    from mmcv.cnn import ConvModule, build_conv_layer


def bev_pool_v2(
    depth: torch.Tensor,
    feat: torch.Tensor,
    ranks_depth: torch.Tensor,
    ranks_feat: torch.Tensor,
    ranks_bev: torch.Tensor,
    bev_feat_shape: tuple[int, int, int, int],
    interval_starts: torch.Tensor,
) -> torch.Tensor:
    """
    Optimized bev_bool_v2 without CUDA ops.

    Parameters
    ----------
        depth (torch.Tensor): Depth tensor with shape [num_cam, d, H, W].
        feat (torch.Tensor): Feature tensor with shape [num_cam, H, W, C].
        ranks_bev (torch.Tensor): Rank of the voxel that a point is belong to
            with shape (N_Points).
        ranks_depth (torch.Tensor): Reserved index of points in the depth space
            with shape (N_Points).
        ranks_feat (torch.Tensor): Reserved index of points in the feature space
            with shape (N_Points).
            interval_starts (torch.Tensor): Interval starts.
        bev_feat_shape (tuple[int, int, int, int]):
            Shape of BEV feature (D, bev_H, bev_W, C)
        interval_starts (torch.Tensor): Interval starts.

    Returns
    -------
        x (torch.Tensor):
            BEV feature with shape of bev_feat_shape with shape (D, C, bev_H, bev_W)
    """
    _, H, W, c = feat.shape

    out = torch.zeros(bev_feat_shape)

    D, bev_H, bev_W, c = out.shape

    # Flatten tensors
    depth = depth.contiguous().view(-1)
    feat = feat.contiguous().view(-1, c)
    out = out.reshape(-1, c)

    # Apply 3D indices to sort/permute directly
    feat_selected = feat[ranks_feat.reshape(-1, 1)[..., 0], :].reshape(W, H, -1, c)
    depth_selected = depth[ranks_depth].reshape(W, H, -1, 1)
    weighted_feat = feat_selected * depth_selected

    change_points = interval_starts[1:] + 1

    # Calculate lengths without torch.diff
    starts = torch.cat([torch.tensor([0], dtype=torch.int), change_points])
    ends = torch.cat([change_points, torch.tensor([W], dtype=torch.int)])
    lengths = ends - starts

    weighted_feat = optimized_cumsum(weighted_feat).reshape(-1, c)
    ends = optimized_cumsum(lengths.reshape(5, 32, 70, 1))
    ends = ends.reshape(-1) - 1  # -1 because ends are exclusive

    # euivalent to torch.diff(weighted_feat, end)
    segment_sums = weighted_feat[ends, :]
    n = segment_sums.size(0)
    diffs = segment_sums[1:, :] - segment_sums[: n - 1, :]
    first_elem = segment_sums[0:1, :]  # Keep dims for ONNX compatibility
    segment_sums = torch.cat([first_elem, diffs], dim=0)

    scatter_indices = ranks_bev[interval_starts]

    out[scatter_indices, :] = segment_sums

    return out.reshape(D, bev_H, bev_W, c).permute(0, 3, 1, 2).contiguous()


class LSSViewTransformerOptimized(BaseModule):
    """
    Optimized Lift-Splat-Shoot view transformer with BEVPoolv2 implementation.
    https://github.com/HuangJunJie2017/BEVDet/blob/26144be7c11c2972a8930d6ddd6471b8ea900d13/mmdet3d/models/necks/view_transformer.py#L18

    Parameters
    ----------
        grid_config (dict): Config of grid alone each axis in format of
            (lower_bound, upper_bound, interval). axis in {x,y,z,depth}.
        input_size (tuple(int,int)): Size of input images in format of (height,
            width).
        downsample (int): Down sample factor from the input size to the feature
            size.
        in_channels (int): Channels of input feature.
        out_channels (int): Channels of transformed feature.
    """

    def __init__(
        self,
        grid_config: dict,
        input_size: tuple[int, int],
        downsample: int = 16,
        in_channels: int = 512,
        out_channels: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.create_grid_infos(**grid_config)
        self.frustum = self.create_frustum(grid_config["depth"], input_size, downsample)
        self.out_channels = out_channels
        self.depth_net = nn.Conv2d(
            in_channels, self.D + self.out_channels, kernel_size=1, padding=0
        )

    def create_grid_infos(
        self,
        x: tuple[float, float, float],
        y: tuple[float, float, float],
        z: tuple[float, float, float],
        **kwargs,
    ):
        """
        Generate the grid information including the lower bound, interval,
        and size.

        Parameters
        ----------
            x (tuple(float)): Config of grid alone x axis in format of
                (lower_bound, upper_bound, interval).
            y (tuple(float)): Config of grid alone y axis in format of
                (lower_bound, upper_bound, interval).
            z (tuple(float)): Config of grid alone z axis in format of
                (lower_bound, upper_bound, interval).
        """
        self.grid_lower_bound = torch.tensor([cfg[0] for cfg in [x, y, z]])
        self.grid_interval = torch.tensor([cfg[2] for cfg in [x, y, z]])
        self.grid_size = torch.tensor([(cfg[1] - cfg[0]) / cfg[2] for cfg in [x, y, z]])

    def create_frustum(
        self, depth_cfg: tuple[float], input_size: tuple[int, int], downsample: int
    ):
        """
        Generate the frustum template for each image.

        Parameters
        ----------
            depth_cfg (tuple(float)): Config of grid alone depth axis in format
                of (lower_bound, upper_bound, interval).
            input_size (tuple(int,int)): Size of input images in format of (height,
                width).
            downsample (int): Down sample scale factor from the input size to
                the feature size.
        """
        H_in, W_in = input_size
        H_feat, W_feat = H_in // downsample, W_in // downsample
        d = (
            torch.arange(*depth_cfg, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, H_feat, W_feat)
        )
        self.D = d.shape[0]
        x = (
            torch.linspace(0, W_in - 1, W_feat, dtype=torch.float)
            .view(1, 1, W_feat)
            .expand(self.D, H_feat, W_feat)
        )
        y = (
            torch.linspace(0, H_in - 1, H_feat, dtype=torch.float)
            .view(1, H_feat, 1)
            .expand(self.D, H_feat, W_feat)
        )

        # D x H x W x 3
        return torch.stack((x, y, d), -1)

    def get_lidar_coor(
        self,
        sensor2keyegos: torch.Tensor,
        img2cams: torch.Tensor,
        inv_post_rots: torch.Tensor,
        post_trans: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Calculate the locations of the frustum points in the lidar
        coordinate system.
        Optimization split the tensors on cams to make it memory efficient.
        changed matmul to mul and reduce_sum

        Parameters
        ----------
            sensor2keyegos: torch.Tensor of shape [B, N, 4, 4] as float32
                transformation matix to convert from camera sensor
                to ego-vehicle at front camera coordinate frame
            img2cams: torch.Tensor of shape [B, N, 3, 3] as float32
                Inverse of Camera intrinsic matrix
                used to project 2D image coordinates to 3D points
            inv_post_rots: torch.Tensor with shape [B, N, 3, 3] as float32
                inverse post rotation matrix in camera coordinate system
            post_trans: torch.Tensor with shape [B, N, 1, 3] as float32
                post translation tensor in camera coordinate system

        Returns
        -------
            List[torch.Tensor]: List of Point coordinates in shape for N_cams
                (1, 3, D, H*W)
        """
        B, N, _, _ = sensor2keyegos.shape
        D, _H, _W, _ = self.frustum.shape

        combine = sensor2keyegos[..., :3, :3].matmul(img2cams)
        post_trans = post_trans.permute(0, 2, 3, 1).reshape(1, 3, 1, N)
        inv_post_rots = inv_post_rots.permute(0, 2, 3, 1).reshape(3, 3, 1, N)
        sensor2keyegos = sensor2keyegos.permute(0, 2, 3, 1)[:, :3, 3:4]
        combine = combine.permute(0, 2, 3, 1).reshape(3, 3, 1, N)

        # post-transformation
        # B x N x D x H x W x 3
        _post_trans = torch.split(post_trans, 1, dim=-1)
        _inv_post_rots = torch.split(inv_post_rots, 1, dim=-1)
        _sensor2ego_ = torch.split(sensor2keyegos, 1, dim=-1)
        _combine = torch.split(combine, 1, dim=-1)

        points_cat = []
        for i in range(N):
            frustum = self.frustum.reshape(B, D, -1, 3).permute(0, 3, 1, 2)
            points = (_inv_post_rots[i] * (frustum - _post_trans[i])).sum(1)

            # cam_to_ego
            points[0:2] *= points[2:3]
            points = (_combine[i] * points.unsqueeze(0)).sum(1).unsqueeze(0)
            points += _sensor2ego_[i]
            points_cat.append(points)

        return points_cat

    def voxel_pooling_prepare_v2(
        self, coor: list[torch.Tensor], feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Data preparation for voxel pooling.
        Optimization Change dynamic to static.

        Parameters
        ----------
            coor (list(torch.Tensor)): Coordinate of points in the lidar space in
                shape (B, 3, D, HW).
            feat (torch.Tensor): Feature tensor with shape [num_cam, C, H, W].

        Returns
        -------
            ranks_bev (torch.Tensor): Rank of the voxel that a point is belong to
            in shape (N_Points).
            ranks_depth (torch.Tensor): Reserved index of points in the depth space
            in shape (N_Points).
            ranks_feat (torch.Tensor): Reserved index of points in the feature space
            in shape (N_Points).
            interval_starts (torch.Tensor): Interval starts.
        """
        B, _, D, HW = coor[0].shape
        N = len(coor)
        num_points = B * N * D * HW
        # record the index of selected points for acceleration purpose
        # Begin Qualcomm modification:
        # Changed range to arange to support export
        ranks_depth = torch.arange(0, num_points, dtype=torch.int32)
        ranks_feat = torch.arange(0, num_points // D, dtype=torch.int32)

        ranks_feat = ranks_feat.reshape(N, 1, 1, HW)
        ranks_feat = ranks_feat.expand(-1, -1, D, -1)
        ranks_depth = ranks_depth.reshape(N, 1, D, HW)

        # split the num_cams for memory efficiency
        ranks_feat_ = torch.split(ranks_feat, 1, dim=0)
        ranks_depth_ = torch.split(ranks_depth, 1, dim=0)

        rd_list = []
        rf_list = []
        rb_list = []

        # convert coordinate into the voxel space
        for _coor, ranks_feat, ranks_depth in zip(
            coor, ranks_feat_, ranks_depth_, strict=False
        ):
            _coor -= self.grid_lower_bound.reshape(1, 3, 1, 1)
            _coor = (_coor / self.grid_interval.reshape(1, 3, 1, 1)).int()

            # filter out points that are outside box
            # Combined seperate comparison.
            kept = (_coor >= 0) & (_coor < self.grid_size.reshape(1, 3, 1, 1))
            kept = kept.sum(dim=1, keepdim=True)
            kept = kept == torch.full(kept.shape, 3)

            # changed gather to mul for optimization.
            _coor = _coor * kept
            ranks_depth = ranks_depth * kept
            ranks_feat = ranks_feat * kept

            # get tensors from the same voxel next to each other
            # Combine seperate mul.
            weights = torch.tensor(
                [
                    self.grid_size[2],
                    self.grid_size[0],
                    self.grid_size[1] * self.grid_size[0],
                ]
            )
            ranks_bev = (_coor * weights.reshape(1, 3, 1, 1)).sum(dim=1, keepdim=True)

            rd_list.append(ranks_depth)
            rf_list.append(ranks_feat)
            rb_list.append(ranks_bev)

        ranks_depth = torch.concat(rd_list, dim=1).reshape(-1)
        ranks_feat = torch.concat(rf_list, dim=1).reshape(-1)
        ranks_bev = torch.concat(rb_list, dim=1).reshape(-1)
        # End Qualcomm modification.

        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = (
            ranks_bev[order],
            ranks_depth[order],
            ranks_feat[order],
        )

        # Begin Qualcomm modification:
        # change not equal to 1-equal and reshape kept for better performance
        _, _, H, W = feat.shape
        kept = torch.cat(
            [
                torch.tensor([1]).to(torch.float32),
                1 - (ranks_bev[1:] == ranks_bev[:-1]).to(torch.float32),
            ]
        ).reshape(W, H, -1, 1)

        # Fixed Dynamic to Static by replacing torch.where
        # Fixed kept as 11200
        cumsum = optimized_cumsum(kept)
        mask = (cumsum <= torch.full(cumsum.shape, 11200)).to(kept.dtype) * kept
        index = ((cumsum - 1) * mask).reshape(-1).long()
        positions = torch.arange(kept.view(-1).size(0)).long()
        interval_starts = torch.zeros(11200, dtype=torch.int64)
        interval_starts[index] = positions * mask.reshape(-1).int()  # 11200
        # End Qualcomm modification.

        return (
            ranks_bev.int().contiguous(),
            ranks_depth.int().contiguous(),
            ranks_feat.int().contiguous(),
            interval_starts.contiguous(),
        )

    def forward(self, xlist: list[torch.Tensor]) -> torch.Tensor:
        """Transform image-view feature into bird-eye-view feature.

        Parameters
        ----------
            xlist (list(torch.Tensor)):
                img-view feature: torch.Tensor of shape [N, C, H, W] as float32
                sensor2keyegos: torch.Tensor of shape [B, N, 4, 4] as float32
                    transformation matix to convert from camera sensor
                    to ego-vehicle at front camera coordinate frame
                inv_intrins: torch.Tensor of shape [B, N, 3, 3] as float32
                    Inverse of Camera intrinsic matrix
                    used to project 2D image coordinates to 3D points
                inv_post_rots: torch.Tensor with shape [B, N, 3, 3] as float32
                    inverse post rotation matrix in camera coordinate system
                post_trans: torch.Tensor with shape [B, N, 1, 3] as float32
                    post translation tensor in camera coordinate system

        Returns
        -------
            bev_feat (torch.Tensor):
                Bird-eye-view feature in shape (B, C, H_BEV, W_BEV)
        """
        x = xlist[0]
        B = 1

        x = self.depth_net(x)

        depth_digit: torch.Tensor = x[:, : self.D, ...]
        feat = x[:, self.D : self.D + self.out_channels, ...]
        depth = depth_digit.softmax(dim=1)
        # Begin Qualcomm modification:
        # Split the input based on num_cam for memory efficiency.
        coor = self.get_lidar_coor(*xlist[1:5])
        # End Qualcomm modification

        (
            ranks_bev,
            ranks_depth,
            ranks_feat,
            interval_starts,
        ) = self.voxel_pooling_prepare_v2(coor, feat)

        feat = feat.permute(0, 2, 3, 1)
        bev_feat_shape = (
            B * int(self.grid_size[2]),
            int(self.grid_size[1]),
            int(self.grid_size[0]),
            feat.shape[-1],
        )

        # Begin Qualcomm modification:
        # add bev_pool_v2 function without cuda ops
        return bev_pool_v2(
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            bev_feat_shape,
            interval_starts,
        )
        # End Qualcomm modification


class SeparateHead(BaseModule):
    """
    SeparateHead for CenterHead.
    Removed class functions that are not used by model to avoid dependency issue

    Parameters
    ----------
        in_channels (int): Input channels for conv_layer.
        heads (dict): Conv information.
        head_conv (int, optional): Output channels.
            Default: 64.
        final_kernel (int, optional): Kernel size for the last conv layer.
            Default: 1.
        conv_cfg (dict, optional): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict, optional): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str, optional): Type of bias. Default: 'auto'.
    """

    def __init__(
        self,
        in_channels: int,
        heads: dict,
        head_conv: int = 64,
        final_kernel: int = 1,
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        bias: str = "auto",
        init_cfg: dict | None = None,
        **kwargs,
    ):
        if norm_cfg is None:
            norm_cfg = dict(type="BN2d")
        if conv_cfg is None:
            conv_cfg = dict(type="Conv2d")
        super().__init__(init_cfg=init_cfg)
        self.heads = heads
        for head in self.heads:
            classes, num_conv = self.heads[head]

            conv_layers_list = []
            c_in = in_channels
            for _i in range(num_conv - 1):
                conv_layers_list.append(
                    ConvModule(
                        c_in,
                        head_conv,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=bias,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                    )
                )
                c_in = head_conv

            conv_layers_list.append(
                build_conv_layer(
                    conv_cfg,
                    head_conv,
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True,
                )
            )
            conv_layers = nn.Sequential(*conv_layers_list)

            self.__setattr__(head, conv_layers)

    def forward(self, x: torch.Tensor) -> dict:
        """Forward function for SepHead.

        Parameters
        ----------
            x (torch.Tensor): Input feature map with the shape of
                [B, C, H, W].

        Returns
        -------
            dict[str: torch.Tensor]: contains the following keys:

                -reg (torch.Tensor): 2D regression value with the
                    shape of [B, 2, H, W].
                -height (torch.Tensor): Height value with the
                    shape of [B, 1, H, W].
                -dim (torch.Tensor): Size value with the shape
                    of [B, 3, H, W].
                -rot (torch.Tensor): Rotation value with the
                    shape of [B, 2, H, W].
                -vel (torch.Tensor): Velocity value with the
                    shape of [B, 2, H, W].
                -heatmap (torch.Tensor): Heatmap with the shape of
                    [B, N, H, W].
        """
        ret_dict = {}
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict


class CenterHead(BaseModule):
    """
    CenterHead for CenterPoint.

    Removed class functions that are not used by model to avoid dependency issue.

    Moved get_bbox class function to app.py, to make bbox_coder out of model,
    with the bbox_coder, the PSNR value is < 30 for bboxs amd scores in cpu,
    the topk and atan2 in bbox_coder.decode causes accuracy issue while export.
    https://github.com/HuangJunJie2017/BEVDet/blob/26144be7c11c2972a8930d6ddd6471b8ea900d13/mmdet3d/models/dense_heads/centerpoint_head.py#L244

    Parameters
    ----------
        in_channels (list[int], optional): Channels of the input
            feature map. Default: [128].
        tasks (list[dict], optional): Task information including class number
            and class names. Default: None.
        common_heads (dict, optional): Conv information for common heads.
            Default: dict().
        separate_head (dict, optional): Config of separate head. Default: dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3)
        share_conv_channel (int, optional): Output channels for share_conv
            layer. Default: 64.
        num_heatmap_convs (int, optional): Number of conv layers for heatmap
            conv layer. Default: 2.
        conv_cfg (dict, optional): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict, optional): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str, optional): Type of bias. Default: 'auto'.
    """

    def __init__(
        self,
        in_channels: list[int] | None = None,
        tasks=None,
        common_heads: dict | None = None,
        separate_head: dict | None = None,
        share_conv_channel: int = 64,
        num_heatmap_convs: int = 2,
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        bias: str = "auto",
        init_cfg: dict | None = None,
        **kwargs,
    ):
        if norm_cfg is None:
            norm_cfg = dict(type="BN2d")
        if conv_cfg is None:
            conv_cfg = dict(type="Conv2d")
        if separate_head is None:
            separate_head = dict(type="SeparateHead", init_bias=-2.19, final_kernel=3)
        if common_heads is None:
            common_heads = {}
        if in_channels is None:
            in_channels = [128]
        assert init_cfg is None, (
            "To prevent abnormal initialization behavior, init_cfg is not allowed to be set"
        )
        super().__init__(init_cfg=init_cfg)

        num_classes = [len(t["class_names"]) for t in tasks]
        # a shared convolution
        self.shared_conv = ConvModule(
            in_channels,
            share_conv_channel,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=bias,
        )

        self.task_heads = nn.ModuleList()

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(num_cls, num_heatmap_convs)))
            separate_head.update(
                in_channels=share_conv_channel, heads=heads, num_cls=num_cls
            )
            separate_head.pop("type")
            self.task_heads.append(SeparateHead(**separate_head))

    def forward(self, x: list[torch.Tensor]) -> list[dict]:
        """Forward pass.

        Parameters
        ----------
            x (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns
        -------
            list[dict]: Output results for tasks contains the following keys:
                -reg (torch.Tensor): 2D regression value with the
                    shape of [B, 2, H, W].
                -height (torch.Tensor): Height value with the
                    shape of [B, 1, H, W].
                -dim (torch.Tensor): Size value with the shape
                    of [B, 3, H, W].
                -rot (torch.Tensor): Rotation value with the
                    shape of [B, 2, H, W].
                -vel (torch.Tensor): Velocity value with the
                    shape of [B, 2, H, W].
                -heatmap (torch.Tensor): Heatmap with the shape of
                    [B, N, H, W].
        """
        x = self.shared_conv(x)
        return [task(x) for task in self.task_heads]
