# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from mmpose v0.29 AEHigherResolutionHead for mmpose 1.x inference.
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmengine.model.weight_init import constant_init, normal_init
from mmengine.structures import InstanceData, PixelData
from torch import Tensor

from mmpose.models.backbones.resnet import BasicBlock
from mmpose.models.utils.tta import aggregate_heatmaps, flip_heatmaps
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (
    ConfigType,
    Features,
    InstanceList,
    OptConfigType,
    OptSampleList,
    Predictions,
)


@MODELS.register_module()
class AEHigherResolutionHead(nn.Module):
    """HigherHRNet head (associative embedding + deconv stage).

    Weight layout matches the official MMPose HigherHRNet-W32 COCO checkpoint
    (``keypoint_head.*`` remapped to ``head.*``).
    """

    def __init__(
        self,
        in_channels: int,
        num_keypoints: int = 17,
        num_joints: Optional[int] = None,
        tag_per_joint: bool = True,
        tag_per_keypoint: Optional[bool] = None,
        tag_dim: int = 1,
        extra: Optional[dict] = None,
        num_deconv_layers: int = 1,
        num_deconv_filters: Sequence[int] = (32,),
        num_deconv_kernels: Sequence[int] = (4,),
        num_basic_blocks: int = 4,
        cat_output: Optional[Sequence[bool]] = None,
        with_ae_loss: Optional[Sequence[bool]] = None,
        decoder: OptConfigType = None,
        init_cfg: OptConfigType = None,
        **kwargs,
    ):
        super().__init__()
        # Accept both mmpose0 (num_joints) and mmpose1 (num_keypoints) names.
        if num_joints is not None:
            num_keypoints = num_joints
        if tag_per_keypoint is not None:
            tag_per_joint = tag_per_keypoint

        self.num_keypoints = int(num_keypoints)
        self.tag_dim = int(tag_dim)
        self.tag_per_keypoint = bool(tag_per_joint)
        self.num_deconvs = int(num_deconv_layers)
        self.cat_output = list(cat_output or [True] * self.num_deconvs)
        with_ae_loss = list(with_ae_loss or ([True] + [False] * self.num_deconvs))

        dim_tag = self.num_keypoints if self.tag_per_keypoint else 1

        final_layer_output_channels = []
        if with_ae_loss[0]:
            final_layer_output_channels.append(self.num_keypoints + dim_tag)
        else:
            final_layer_output_channels.append(self.num_keypoints)
        for i in range(self.num_deconvs):
            if with_ae_loss[i + 1]:
                final_layer_output_channels.append(self.num_keypoints + dim_tag)
            else:
                final_layer_output_channels.append(self.num_keypoints)

        deconv_layer_output_channels = []
        for i in range(self.num_deconvs):
            if with_ae_loss[i]:
                deconv_layer_output_channels.append(self.num_keypoints + dim_tag)
            else:
                deconv_layer_output_channels.append(self.num_keypoints)

        self.final_layers = self._make_final_layers(
            in_channels,
            final_layer_output_channels,
            extra,
            self.num_deconvs,
            num_deconv_filters,
        )
        self.deconv_layers = self._make_deconv_layers(
            in_channels,
            deconv_layer_output_channels,
            self.num_deconvs,
            num_deconv_filters,
            num_deconv_kernels,
            num_basic_blocks,
            self.cat_output,
        )

        self.decoder = None
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)

    @staticmethod
    def _make_final_layers(
        in_channels, final_layer_output_channels, extra, num_deconv_layers, num_deconv_filters
    ):
        if extra is not None and "final_conv_kernel" in extra:
            assert extra["final_conv_kernel"] in [1, 3]
            kernel_size = extra["final_conv_kernel"]
            padding = 1 if kernel_size == 3 else 0
        else:
            kernel_size = 1
            padding = 0

        final_layers = [
            build_conv_layer(
                cfg=dict(type="Conv2d"),
                in_channels=in_channels,
                out_channels=final_layer_output_channels[0],
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            )
        ]
        for i in range(num_deconv_layers):
            final_layers.append(
                build_conv_layer(
                    cfg=dict(type="Conv2d"),
                    in_channels=num_deconv_filters[i],
                    out_channels=final_layer_output_channels[i + 1],
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                )
            )
        return nn.ModuleList(final_layers)

    def _make_deconv_layers(
        self,
        in_channels,
        deconv_layer_output_channels,
        num_deconv_layers,
        num_deconv_filters,
        num_deconv_kernels,
        num_basic_blocks,
        cat_output,
    ):
        deconv_layers = []
        for i in range(num_deconv_layers):
            if cat_output[i]:
                in_channels = in_channels + deconv_layer_output_channels[i]
            planes = num_deconv_filters[i]
            deconv_kernel, padding, output_padding = self._get_deconv_cfg(num_deconv_kernels[i])
            layers = [
                nn.Sequential(
                    build_upsample_layer(
                        dict(type="deconv"),
                        in_channels=in_channels,
                        out_channels=planes,
                        kernel_size=deconv_kernel,
                        stride=2,
                        padding=padding,
                        output_padding=output_padding,
                        bias=False,
                    ),
                    nn.BatchNorm2d(planes, momentum=0.1),
                    nn.ReLU(inplace=True),
                )
            ]
            for _ in range(num_basic_blocks):
                layers.append(nn.Sequential(BasicBlock(planes, planes)))
            deconv_layers.append(nn.Sequential(*layers))
            in_channels = planes
        return nn.ModuleList(deconv_layers)

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        if deconv_kernel == 4:
            return 4, 1, 0
        if deconv_kernel == 3:
            return 3, 1, 1
        if deconv_kernel == 2:
            return 2, 0, 0
        raise ValueError(f"Not supported num_kernels ({deconv_kernel}).")

    @staticmethod
    def _select_hrnet_feat(feats: Union[Tensor, Sequence[Tensor]]) -> Tensor:
        """HigherHRNet consumes the highest-resolution HRNet stream (32-ch)."""
        if isinstance(feats, (list, tuple)):
            # Prefer 32-ch high-res branch; fall back to first tensor.
            for f in feats:
                if isinstance(f, Tensor) and f.dim() == 4 and f.shape[1] == 32:
                    return f
            return feats[0]
        return feats

    def forward_stages(self, x: Union[Tensor, Sequence[Tensor]]) -> List[Tensor]:
        """Return raw multi-stage outputs (same layout as mmpose0)."""
        x = self._select_hrnet_feat(x)
        final_outputs = []
        y = self.final_layers[0](x)
        final_outputs.append(y)
        for i in range(self.num_deconvs):
            if self.cat_output[i]:
                x = torch.cat((x, y), 1)
            x = self.deconv_layers[i](x)
            y = self.final_layers[i + 1](x)
            final_outputs.append(y)
        return final_outputs

    def _split_hm_tag(self, out: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        heatmaps = out[:, : self.num_keypoints]
        if out.shape[1] > self.num_keypoints:
            tags = out[:, self.num_keypoints :]
        else:
            tags = None
        return heatmaps, tags

    def forward(self, feats: Union[Tensor, Tuple[Tensor]]) -> Tuple[Tensor, Tensor]:
        """Return aggregated (heatmaps, tags) for decode / from-h predict."""
        stages = self.forward_stages(feats)
        heatmaps_list = []
        tags0 = None
        for i, stage in enumerate(stages):
            hm, tags = self._split_hm_tag(stage)
            heatmaps_list.append(hm)
            if i == 0:
                tags0 = tags
        # Average heatmaps after aligning to the highest-resolution stage.
        target = heatmaps_list[-1]
        aligned = []
        for hm in heatmaps_list:
            if hm.shape[-2:] != target.shape[-2:]:
                hm = F.interpolate(hm, size=target.shape[-2:], mode="bilinear", align_corners=False)
            aligned.append(hm)
        heatmaps = torch.stack(aligned, dim=0).mean(dim=0)
        if tags0 is None:
            tags0 = torch.zeros(
                heatmaps.shape[0],
                self.num_keypoints * self.tag_dim if self.tag_per_keypoint else self.tag_dim,
                *heatmaps.shape[-2:],
                device=heatmaps.device,
                dtype=heatmaps.dtype,
            )
        elif tags0.shape[-2:] != heatmaps.shape[-2:]:
            tags0 = F.interpolate(tags0, size=heatmaps.shape[-2:], mode="bilinear", align_corners=False)
        return heatmaps, tags0

    def _flip_tags(self, tags: Tensor, flip_indices: List[int], shift_heatmap: bool = True):
        B, C, H, W = tags.shape
        K = self.num_keypoints
        L = self.tag_dim
        tags = tags.flip(-1)
        if self.tag_per_keypoint:
            assert C == K * L
            tags = tags.view(B, L, K, H, W)
            tags = tags[:, :, flip_indices]
            tags = tags.view(B, C, H, W)
        if shift_heatmap:
            tags[..., 1:] = tags[..., :-1].clone()
        return tags

    def predict(
        self,
        feats: Features,
        batch_data_samples: OptSampleList,
        test_cfg: ConfigType = {},
    ) -> Predictions:
        from mmengine.utils import is_list_of

        multiscale_test = test_cfg.get("multiscale_test", False)
        flip_test = test_cfg.get("flip_test", False)
        shift_heatmap = test_cfg.get("shift_heatmap", False)
        align_corners = test_cfg.get("align_corners", False)
        restore_heatmap_size = test_cfg.get("restore_heatmap_size", False)
        output_heatmaps = test_cfg.get("output_heatmaps", False)

        if multiscale_test:
            assert is_list_of(feats, list if flip_test else tuple)
        else:
            assert is_list_of(feats, tuple if flip_test else Tensor)
            feats = [feats]

        if restore_heatmap_size:
            img_shape = batch_data_samples[0].metainfo["img_shape"]
            heatmap_size = (img_shape[1], img_shape[0])
        else:
            heatmap_size = None

        multiscale_heatmaps = []
        multiscale_tags = []
        for scale_idx, _feats in enumerate(feats):
            if not flip_test:
                _heatmaps, _tags = self.forward(_feats)
                # Match AE-HRNet: when restore_heatmap_size, decode in input/image
                # pixel space with decoder scale_factor≈1.
                if heatmap_size is not None:
                    _heatmaps = aggregate_heatmaps(
                        [_heatmaps],
                        size=heatmap_size,
                        align_corners=align_corners,
                        mode="average",
                    )
                    if _tags is not None:
                        _tags = aggregate_heatmaps(
                            [_tags],
                            size=heatmap_size,
                            align_corners=align_corners,
                            mode="average",
                        )
            else:
                assert isinstance(_feats, list) and len(_feats) == 2
                flip_indices = batch_data_samples[0].metainfo["flip_indices"]
                _feats_orig, _feats_flip = _feats
                _heatmaps_orig, _tags_orig = self.forward(_feats_orig)
                _heatmaps_flip, _tags_flip = self.forward(_feats_flip)
                _heatmaps_flip = flip_heatmaps(
                    _heatmaps_flip,
                    flip_mode="heatmap",
                    flip_indices=flip_indices,
                    shift_heatmap=shift_heatmap,
                )
                _tags_flip = self._flip_tags(
                    _tags_flip, flip_indices=flip_indices, shift_heatmap=shift_heatmap
                )
                _heatmaps = aggregate_heatmaps(
                    [_heatmaps_orig, _heatmaps_flip],
                    size=heatmap_size,
                    align_corners=align_corners,
                    mode="average",
                )
                if scale_idx == 0:
                    _tags = aggregate_heatmaps(
                        [_tags_orig, _tags_flip],
                        size=heatmap_size,
                        align_corners=align_corners,
                        mode="concat",
                    )
                else:
                    _tags = None
            multiscale_heatmaps.append(_heatmaps)
            multiscale_tags.append(_tags)

        if len(feats) > 1:
            batch_heatmaps = aggregate_heatmaps(
                multiscale_heatmaps, align_corners=align_corners, mode="average"
            )
        else:
            batch_heatmaps = multiscale_heatmaps[0]
        batch_tags = multiscale_tags[0]
        preds = self.decode((batch_heatmaps, batch_tags))
        if output_heatmaps:
            pred_fields = [
                PixelData(heatmaps=_hm, tags=_tg)
                for _hm, _tg in zip(batch_heatmaps.detach(), batch_tags.detach())
            ]
            return preds, pred_fields
        return preds

    def decode(self, batch_outputs: Union[Tensor, Tuple[Tensor]]) -> InstanceList:
        if self.decoder is None:
            raise RuntimeError(
                f"The decoder has not been set in {self.__class__.__name__}."
            )

        def _pack_and_call(args, func):
            if not isinstance(args, tuple):
                args = (args,)
            return func(*args)

        if self.decoder.support_batch_decoding:
            batch_keypoints, batch_scores, batch_instance_scores = _pack_and_call(
                batch_outputs, self.decoder.batch_decode
            )
        else:
            batch_output_np = to_numpy(batch_outputs, unzip=True)
            batch_keypoints, batch_scores, batch_instance_scores = [], [], []
            for outputs in batch_output_np:
                keypoints, scores, instance_scores = _pack_and_call(outputs, self.decoder.decode)
                batch_keypoints.append(keypoints)
                batch_scores.append(scores)
                batch_instance_scores.append(instance_scores)

        return [
            InstanceData(
                bbox_scores=instance_scores,
                keypoints=keypoints,
                keypoint_scores=scores,
            )
            for keypoints, scores, instance_scores in zip(
                batch_keypoints, batch_scores, batch_instance_scores
            )
        ]

    def init_weights(self):
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for _, m in self.final_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
