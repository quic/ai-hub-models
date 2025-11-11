# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TRF
from easyocr.craft_utils import getDetBoxes
from easyocr.easyocr import Reader
from easyocr.recognition import custom_mean
from easyocr.utils import (
    diff,
    four_point_transform,
    group_text_box,
)
from PIL import Image

from qai_hub_models.utils.draw import draw_box_from_corners, draw_box_from_xyxy
from qai_hub_models.utils.image_processing import (
    app_to_net_image_inputs,
    denormalize_coordinates,
    numpy_image_to_torch,
    resize_pad,
)

DETECTOR_ARGS = {
    "text_threshold": 0.7,
    "link_threshold": 0.4,
    "low_text": 0.4,
    "poly": False,
    "slope_ths": 0.1,
    "ycenter_ths": 0.5,
    "height_ths": 0.5,
    "width_ths": 0.5,
    "add_margin": 0.1,
    "min_size": 20,
}

RECOGNIZER_ARGS = {
    "allowlist": None,
    "blocklist": None,
    "beamWidth": 5,
    "contrast_ths": 0.1,
    "adjust_contrast": 0.5,
}


# xmin, xmax, ymin, ymax
box_xx_yy = tuple[int, int, int, int]

# ((x1,y1), (x2,y2), (x3,y3), (x4,y4))
box_4corners = tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]


class EasyOCRApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with EasyOCR.

    The app uses 2 models:
        * detector
        * recognizer

    For a given image input, the app will:
        * use the reformat_input() to pre-process the image (img, img_cv_grey)
        * run detector_preprocess() change input to tensor
        * run detector and post-process the result to bbox list
        * run recognizer twice for low confident score
        * map the score according to decoder
    """

    def __init__(
        self,
        detector: Callable[[torch.Tensor], torch.Tensor],
        recognizer: Callable[[torch.Tensor], torch.Tensor],
        detector_img_shape: tuple[int, int],
        recognizer_img_shape: tuple[int, int],
        lang_list: list[str],
        decoder_mode: str = "greedy",
    ):
        self.detector = detector
        self.recognizer = recognizer
        self.decoder = decoder_mode
        ocr_reader = Reader(
            lang_list,
            gpu=False,
            quantize=False,
        )
        self.character = ocr_reader.character
        self.lang_char = ocr_reader.lang_char
        self.model_lang = ocr_reader.model_lang
        self.converter = ocr_reader.converter
        self.detector_img_shape = detector_img_shape
        self.recognizer_img_shape = recognizer_img_shape

        if RECOGNIZER_ARGS["allowlist"]:
            self.ignore_char = "".join(
                set(self.character) - set(RECOGNIZER_ARGS["allowlist"])  # type: ignore[call-overload]
            )
        elif RECOGNIZER_ARGS["blocklist"]:
            self.ignore_char = "".join(set(RECOGNIZER_ARGS["blocklist"]))  # type: ignore[call-overload]
        else:
            self.ignore_char = "".join(set(self.character) - set(self.lang_char))
        self.ignore_char_idx = []
        for char in self.ignore_char:
            self.ignore_char_idx.append(self.character.index(char) + 1)

    def detector_preprocess(
        self, NHWC_int_numpy_frames: list[np.ndarray]
    ) -> tuple[torch.Tensor, list[float], list[tuple[int, int]]]:
        """
        Resize and prepare input images (must be RGB) for detection.

        Inputs:
            NHWC_int_numpy_frames: list[np.ndarray]
                [H', W', 3], uint8
                Input images

        Outputs:
            detector_input_frames: torch.Tensor
                [B, 3, H', W'], fp32, range 0-1, RGB
                Input tensor for the detector network.

            scales: list[float]
                List of scaling factors used to resize each input image for network inference.

            paddings: list[tuple[int, int]]
                List of padding (width, height) used to resize each input image for network inference.
        """
        detector_input_frames_list, scales, paddings = [], [], []
        for frame in NHWC_int_numpy_frames:
            frame_resized, scale, pad = resize_pad(
                numpy_image_to_torch(frame, to_float=False).to(torch.float32) / 255,
                self.detector_img_shape,
            )
            detector_input_frames_list.append(frame_resized)
            scales.append(scale)
            paddings.append(pad)
        detector_input_frames = torch.cat(detector_input_frames_list)

        return detector_input_frames, scales, paddings

    def detector_postprocess(
        self,
        results: torch.Tensor,
        img_scale_ratios: list[float],
        img_paddings: list[tuple[int, int]],
    ) -> tuple[list[list[box_xx_yy]], list[list[box_4corners]]]:
        """
        Process results of detector network.

        Inputs:
            results: torch.Tensor
                Output of the detector network.

            img_scale_ratios: list[tuple[float, float]]
                List of scaling factors used to resize the input image for network inference.

            img_paddings:
                List of padding (width, height) used to resize each input image for network inference.

        Outputs:
            horizontal_boxes_per_img: list[list[box_xx_yy]]
                List of bounding boxes (absolute pixel values) in each image.
                These boxes are always a rectangle that is parallel to the image's coordinate space.
                Order: (xmin, xmax, ymin, ymax)

            free_boxes_per_img: list[list[box_4corners]]
                List of bounding boxes (absolute pixel values) in each image.
                These boxes may be any parallelogram.
                Order: ((x1,y1), (x2,y2), (x3,y3), (x4,y4))

        """
        boxes_per_img: list[list[np.ndarray]] = []
        horizontal_boxes_per_img: list[list[box_xx_yy]] = []
        free_boxes_per_img: list[list[box_4corners]] = []

        for out, img_scale_ratio, img_padding in zip(
            results, img_scale_ratios, img_paddings, strict=False
        ):
            # make score and link map
            score_text = out[:, :, 0].cpu().data.numpy()
            score_link = out[:, :, 1].cpu().data.numpy()

            # Post-processing
            required_keys = [
                "textmap",
                "linkmap",
                "text_threshold",
                "link_threshold",
                "low_text",
                "poly",
            ]
            filtered_args = {
                k: DETECTOR_ARGS[k] for k in required_keys if k in DETECTOR_ARGS
            }

            # horizontal_boxes & free_boxes layout: top-left, top-right, low-right, low-left
            horizontal_boxes: list[np.ndarray]
            free_boxes: list[np.ndarray]
            horizontal_boxes, free_boxes, _ = getDetBoxes(
                score_text, score_link, **filtered_args
            )
            assert len(horizontal_boxes) == len(free_boxes)

            # Select either horizontal_boxes or free_boxes for each returned box index. Prefer free_boxes if they exist.
            #
            # We combine both types of boxes into 1 list because some free boxes are
            # reinterpreted as horizontal boxes later -- see group_text_box()
            detections: list[np.ndarray] = []
            for box_idx in range(len(horizontal_boxes)):
                if free_boxes[box_idx] is not None:
                    box = torch.Tensor(free_boxes[box_idx])
                else:
                    box = torch.Tensor(horizontal_boxes[box_idx])

                # coordinate adjustment for image scaling and padding
                # we divide scale & padding by 2 to reflect the network's scaling on output indices
                denormalize_coordinates(
                    box,
                    (1, 1),
                    img_scale_ratio / 2,
                    (img_padding[0] // 2, img_padding[1] // 2),
                )
                detections.append(box.numpy().astype(np.int32).reshape(-1))

            boxes_per_img.append(detections)

        required_keys = [
            "slope_ths",
            "ycenter_ths",
            "height_ths",
            "width_ths",
            "add_margin",
        ]
        filtered_args = {
            k: DETECTOR_ARGS[k] for k in required_keys if k in DETECTOR_ARGS
        }
        for text_box in boxes_per_img:
            # This will reinterpret free boxes as horizontal boxes if the free box
            # "slant" is less than 'slope_ths'. Read the function for more details.
            horizontal_list_raw, free_list_raw = group_text_box(
                text_box, **filtered_args
            )
            horizontal_list: list[box_xx_yy] = [tuple(x) for x in horizontal_list_raw]
            free_list: list[box_4corners] = [
                cast(box_4corners, tuple(tuple(y) for y in x)) for x in free_list_raw
            ]
            if DETECTOR_ARGS["min_size"]:
                horizontal_list = [
                    i
                    for i in horizontal_list
                    if max(i[1] - i[0], i[3] - i[2]) > DETECTOR_ARGS["min_size"]
                ]
                free_list = [
                    i
                    for i in free_list
                    if max(diff([c[0] for c in i]), diff([c[1] for c in i]))
                    > DETECTOR_ARGS["min_size"]
                ]
            horizontal_boxes_per_img.append(horizontal_list)
            free_boxes_per_img.append(free_list)

        return horizontal_boxes_per_img, free_boxes_per_img

    def recognizer_get_text(
        self,
        img_cv_grey: np.ndarray,
        horizontal_boxes: list[box_xx_yy],
        free_boxes: list[box_4corners],
    ) -> tuple[
        list[tuple[box_xx_yy, str, np.float64]],
        list[tuple[box_4corners, str, np.float64]],
    ]:
        """
        Predict the text in the given image using the recognizer network.

        Inputs:
            img_cv_grey: np.ndarray
                [H', W', 3], uint8
                Input image

            horizontal_boxes: list[box_xx_yy]
                List of bounding box corners (absolute pixel values) in each image.
                These boxes are always a rectangle that is parallel to the image's coordinate space.
                Order: (xmin, xmax, ymin, ymax)

            free_boxes_per_img: list[box_4corners]
                List of bounding box corners (absolute pixel values) in each image.
                These boxes may be any parallelogram.
                Order: ((x1,y1), (x2,y2), (x3,y3), (x4,y4))

        Outputs:
            result_box_xxyy: list[tuple[box_xx_yy, str, np.float64]]
                This is a list of outputs, one for each detection related to a horizontal box.
                In this tuple:
                    box: box_xx_yy
                        The bounding box coordinates corresponding to this prediction.

                    text: str
                        The predicted text.

                    confidence: np.float64
                        Prediction confidence.


            result_box_4corners: list[tuple[box_4corners, str, np.float64]]]
                This is a list of outputs, one for each detection related to a free box.
                In this tuple:
                    box: box_4corners
                        The bounding box coordinates corresponding to this prediction.

                    text: str
                        The predicted text.

                    confidence: np.float64
                        Prediction confidence.
        """
        # If horizontal boxes and free boxes are not set, use the entire image instead
        if not horizontal_boxes and not free_boxes:
            y_max, x_max = img_cv_grey.shape
            horizontal_boxes = [(0, x_max, 0, y_max)]
            free_boxes = []

        ## Cut out boxes
        # List[cutout image, box coords, minimum y coordinate]
        img_cutouts: list[tuple[np.ndarray, box_xx_yy | box_4corners, int]] = []

        # Free boxes must be warped to a square before cropping
        for free_box in free_boxes:
            rect = np.array(free_box, dtype="float32")
            cutout = four_point_transform(img_cv_grey, rect)
            # Sometimes a predicted parallelogram is not valid, and therefore has
            # a cutout with a dimension of 0. We ignore these.
            if 0 in cutout.shape:
                continue

            img_cutouts.append(
                (cutout, free_box, min(rect[0][1], rect[1][1], rect[2][1], rect[3][1]))
            )

        # Horizontal boxes can be directly cropped out of the image
        for box in horizontal_boxes:
            x_min = max(0, box[0])
            x_max = int(min(box[1], img_cv_grey.shape[1]))
            y_min = max(0, box[2])
            y_max = int(min(box[3], img_cv_grey.shape[0]))

            # Sometimes a predicted parallelogram is not valid, and therefore has
            # a cutout with a dimension of 0. We ignore these.
            if y_max - y_min <= 0 or x_max - x_min <= 0:
                continue

            cutout = img_cv_grey[y_min:y_max, x_min:x_max]
            img_cutouts.append((cutout, box, y_min))

        img_cutouts = sorted(
            img_cutouts, key=lambda item: item[2]
        )  # sort by vertical position

        # Preprocess cutouts to fit input shape of the recognizer
        cutout_frames_list = []
        for frame, _, _ in img_cutouts:
            img = numpy_image_to_torch(frame[..., None])
            # Use top left pixel as a heuristic of background color to use for padding
            # Since zero padding creates a black rectangle that is sometimes parsed as
            # another character.
            frame_resized, _, _ = resize_pad(
                img,
                self.recognizer_img_shape,
                horizontal_float="left",
                pad_value=img[0][0][0][0].item(),
            )
            cutout_frames_list.append(frame_resized)

        ## Predict text from all cutouts
        pred_text_confidence = self.recognizer_inference(cutout_frames_list)

        # For predictions with low confidence, predict again with higher contrast.
        low_confidence_indices = [
            i
            for i, (_, confidence) in enumerate(pred_text_confidence)
            if (
                RECOGNIZER_ARGS["contrast_ths"] is not None
                and confidence < RECOGNIZER_ARGS["contrast_ths"]
            )
        ]
        low_confience_cutouts = [cutout_frames_list[i] for i in low_confidence_indices]
        if low_confience_cutouts:
            contrast = (
                1 / RECOGNIZER_ARGS["contrast_ths"]
                if RECOGNIZER_ARGS["contrast_ths"]
                else 1
            )
            low_confience_high_contrast_cutout_frames = [
                TRF.adjust_contrast(img, contrast).unsqueeze(0)
                for img in low_confience_cutouts
            ]
            high_contrast_pred_text_confidence = self.recognizer_inference(
                low_confience_high_contrast_cutout_frames
            )
        else:
            high_contrast_pred_text_confidence = []

        ## Collect Results
        result_box_xxyy: list[tuple[box_xx_yy, str, np.float64]] = []
        result_box_4corners: list[tuple[box_4corners, str, np.float64]] = []
        for i, ((_, cutout_box_corners, _), (pred_text, pred_confidence)) in enumerate(
            zip(img_cutouts, pred_text_confidence, strict=False)
        ):
            res_text: str
            res_confidence: np.float64

            if i in low_confidence_indices:
                # If this was predicted with enhanced contrast, then take the prediction with higher confidence.
                high_contrast_pred_text, high_contrast_pred_confidence = (
                    high_contrast_pred_text_confidence[low_confidence_indices.index(i)]
                )
                if pred_confidence > high_contrast_pred_confidence:
                    res_text, res_confidence = pred_text, pred_confidence
                else:
                    res_text, res_confidence = (
                        high_contrast_pred_text,
                        high_contrast_pred_confidence,
                    )
            else:
                res_text, res_confidence = pred_text, pred_confidence

            if not res_text:
                # ignore empty predictions
                continue

            # The recognizer can hallucinate these tokens when there is
            # substantial empty space at the end of the text.
            #
            # When the input image is padded to match the input shape of the recognizer,
            # empty space (padding) is always added to the right of the image.
            res_text = res_text.strip()
            if res_text[-1] in ["]", "|"]:
                res_text = res_text[:-1].strip()

            # Append to the correct list depending on the bbox type.
            if isinstance(cutout_box_corners[0], tuple):
                cutout_box_4corners = cast(box_4corners, cutout_box_corners)
                result_box_4corners.append(
                    (cutout_box_4corners, res_text, res_confidence)
                )
            else:
                cutout_box_tl = cast(box_xx_yy, cutout_box_corners)
                result_box_xxyy.append((cutout_box_tl, res_text, res_confidence))

        return result_box_xxyy, result_box_4corners

    def recognizer_inference(
        self, cutout_frames: list[torch.Tensor]
    ) -> list[tuple[str, np.float64]]:
        """
        Predict the text in the given cut out frames.

        Inputs:
            cutout_frames: list[torch.Tensor]
                list of [1, 1, H', W'], fp32
                Input cutouts. These should be image cutouts
                containing only the text for the recognizer to read.

        Outputs:
            results: list[tuple[str, np.float64]]
                Predictions, one per input frame.
                In this tuple:
                    text: str
                        The predicted text.

                    confidence: np.float64
                        Prediction confidence.
        """
        result: list[tuple[str, np.float64]] = []
        preds_list = []
        with torch.no_grad():
            preds_list = [
                self.recognizer(cutout_frame) for cutout_frame in cutout_frames
            ]
        preds = torch.cat(preds_list)

        # Select max probabilty (greedy decoding) then decode index to character
        preds_size = torch.IntTensor([preds.size(1)] * len(cutout_frames))

        ######## filter ignore_char, rebalance
        preds_prob = F.softmax(preds, dim=2)
        preds_prob[:, :, self.ignore_char_idx] = 0.0
        pred_norm = preds_prob.sum(dim=2)
        preds_prob = preds_prob / pred_norm.unsqueeze(-1)

        preds_str: list[str]
        if self.decoder == "greedy":
            # Select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds_prob.max(2)
            preds_index = preds_index.view(-1)
            preds_str = self.converter.decode_greedy(
                preds_index.data.cpu().detach().numpy(), preds_size.data
            )
        elif self.decoder == "beamsearch":
            k = preds_prob.cpu().detach().numpy()
            preds_str = self.converter.decode_beamsearch(
                k, beamWidth=RECOGNIZER_ARGS["beamWidth"]
            )
        elif self.decoder == "wordbeamsearch":
            k = preds_prob.cpu().detach().numpy()
            preds_str = self.converter.decode_wordbeamsearch(
                k, beamWidth=RECOGNIZER_ARGS["beamWidth"]
            )
        else:
            raise NotImplementedError(f"Unknown decoder {self.decoder}")

        preds_prob_np = preds_prob.cpu().detach().numpy()
        values = preds_prob_np.max(axis=2)
        indices = preds_prob_np.argmax(axis=2)
        preds_max_prob = []
        for v, i in zip(values, indices, strict=False):
            max_probs = v[i != 0]
            if len(max_probs) > 0:
                preds_max_prob.append(max_probs)
            else:
                preds_max_prob.append(np.array([0]))

        for pred, pred_max_prob in zip(preds_str, preds_max_prob, strict=False):
            confidence_score = custom_mean(pred_max_prob)
            result.append((pred, confidence_score))

        return result

    def predict(self, *args, **kwargs):
        return self.predict_text_from_image(*args, **kwargs)

    def predict_text_from_image(
        self, pixel_values_or_image: np.ndarray | Image.Image
    ) -> list[tuple[Image.Image, list[str], list[np.float64]]]:
        """
        From provided array or image, predict (bounding box of texts, text, scores)

        Parameters
        ----------
            pixel_values_or_image
                PIL image(s)
                or
                numpy array (N H W C x uint8) or (H W C x uint8) -- both RGB channel layout

        Returns
        -------
            results: list[tuple[Image.Image, list[str], list[np.float64]]]
                Predictions for each image.
                In this tuple:
                    image: Image.Image
                        Predicted image with bounding boxes drawn.

                    text: list[str]
                        The predicted texts.

                    confidence: list[np.float64]
                        Prediction confidence for each predicted text / bounding box combo.

        """
        # Get frames
        NHWC_int_numpy_frames, _ = app_to_net_image_inputs(pixel_values_or_image)
        NHWC_int_numpy_GRAY_frames = [
            cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in NHWC_int_numpy_frames
        ]

        # detector
        detector_input_frames, scales, paddings = self.detector_preprocess(
            NHWC_int_numpy_frames
        )
        results = self.detector(detector_input_frames)
        horizontal_boxes_per_img, free_boxes_per_img = self.detector_postprocess(
            results, scales, paddings
        )

        # recognizer
        output: list[tuple[Image.Image, list[str], list[np.float64]]] = []
        for img, img_gray, horizontal_boxes, free_boxes in zip(
            NHWC_int_numpy_frames,
            NHWC_int_numpy_GRAY_frames,
            horizontal_boxes_per_img,
            free_boxes_per_img,
            strict=False,
        ):
            img_results_horizonal, img_results_free = self.recognizer_get_text(
                img_gray, horizontal_boxes, free_boxes
            )

            # Collect output & draw boxes
            img = img.copy()
            texts = []
            confidences = []
            for box_coords, text, confidence in img_results_horizonal:
                draw_box_from_xyxy(
                    img,
                    (box_coords[0], box_coords[2]),
                    (box_coords[1], box_coords[3]),
                    color=(0, 255, 0),
                    size=2,
                )
                texts.append(text)
                confidences.append(confidence)

            for free_box_coords, text, confidence in img_results_free:
                draw_box_from_corners(
                    img,
                    np.array(free_box_coords),
                    color=(0, 255, 0),
                    size=2,
                )
                texts.append(text)
                confidences.append(confidence)

            output.append((Image.fromarray(img), texts, confidences))

        return output
