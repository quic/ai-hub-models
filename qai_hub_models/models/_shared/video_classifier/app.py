# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
import torchvision.io

from qai_hub_models.models._shared.video_classifier.model import KineticsClassifier


def normalize(video: torch.Tensor):
    """Normalize the video frames.
    Parameters:
        video: Video tensor (Number of frames x HWC) with values between 0-255
               Channel Layout: RGB

    Returns:
        video: Video is normalized to have values between 0-1
               and transposed so the shape is Channel x Number of frames x HW.
    """
    return video.permute(3, 0, 1, 2).to(torch.float32) / 255


def resize(video: torch.Tensor, size: Tuple[int, int]):
    """
    Interpolate the frames of the image to match model's input resolution.

    Parameters:
        video: torch.Tensor

    Returns:
        video: Resized video is returned.
               Selected settings for resize were recommended.

    """
    return torch.nn.functional.interpolate(
        video, size=size, scale_factor=None, mode="bilinear", align_corners=False
    )


def crop(video: torch.Tensor, output_size: Tuple[int, int]):
    """
    Parameters:
        video: torch.Tensor
            Input video torch.Tensor.
        output_size: desired output shape for each frame.

    Returns:
        video: torch.Tensor
            Center cropped based on the output size

    """
    h, w = video.shape[-2:]
    th, tw = output_size
    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return video[..., i : (i + th), j : (j + tw)]


def normalize_base(
    video: torch.Tensor, mean: List[float], std: List[float]
) -> torch.Tensor:
    """

    Parameters:
        video: Input video torch.Tensor
        mean: Mean to be subtracted per channel of the input.
        std: Standard deviation to be divided per each channel.

    Returns:
        video: Normalized based on provided mean and scale.
               The operaion is done per channle.

    """
    shape = (-1,) + (1,) * (video.dim() - 1)
    mean_tensor = torch.as_tensor(mean).reshape(shape)
    std_tensor = torch.as_tensor(std).reshape(shape)
    return (video - mean_tensor) / std_tensor


def read_video_per_second(path: str) -> torch.Tensor:
    """

    Parameters:
        path: Path of the input video.

    Returns:
        input_video: Reads video from path and converts to torch tensor.

    """
    input_video, _, _ = torchvision.io.read_video(path, pts_unit="sec")
    return input_video


def preprocess_video_kinetics_400(input_video: torch.Tensor):
    """
    Preprocess the input video correctly for video classification inference.

    Parameters:
        input_video: Raw input tensor

    Returns:
        video: Normalized, resized, cropped and normalized by channel for input model.
               This preprocessing is dd

    """
    mean = [0.43216, 0.394666, 0.37645]
    std = [0.22803, 0.22145, 0.216989]
    input_video = normalize(input_video)
    input_video = resize(input_video, (128, 171))
    input_video = crop(input_video, (112, 112))
    input_video = normalize_base(input_video, mean=mean, std=std)
    return input_video


def get_class_name_kinetics_400() -> List[str]:
    """Return the class name."""
    actions = "abseiling,air drumming,answering questions,applauding,applying cream,archery,arm wrestling,arranging flowers,assembling computer,auctioning,baby waking up,baking cookies,balloon blowing,bandaging,barbequing,bartending,beatboxing,bee keeping,belly dancing,bench pressing,bending back,bending metal,biking through snow,blasting sand,blowing glass,blowing leaves,blowing nose,blowing out candles,bobsledding,bookbinding,bouncing on trampoline,bowling,braiding hair,breading or breadcrumbing,breakdancing,brush painting,brushing hair,brushing teeth,building cabinet,building shed,bungee jumping,busking,canoeing or kayaking,capoeira,carrying baby,cartwheeling,carving pumpkin,catching fish,catching or throwing baseball,catching or throwing frisbee,catching or throwing softball,celebrating,changing oil,changing wheel,checking tires,cheerleading,chopping wood,clapping,clay pottery making,clean and jerk,cleaning floor,cleaning gutters,cleaning pool,cleaning shoes,cleaning toilet,cleaning windows,climbing a rope,climbing ladder,climbing tree,contact juggling,cooking chicken,cooking egg,cooking on campfire,cooking sausages,counting money,country line dancing,cracking neck,crawling baby,crossing river,crying,curling hair,cutting nails,cutting pineapple,cutting watermelon,dancing ballet,dancing charleston,dancing gangnam style,dancing macarena,deadlifting,decorating the christmas tree,digging,dining,disc golfing,diving cliff,dodgeball,doing aerobics,doing laundry,doing nails,drawing,dribbling basketball,drinking,drinking beer,drinking shots,driving car,driving tractor,drop kicking,drumming fingers,dunking basketball,dying hair,eating burger,eating cake,eating carrots,eating chips,eating doughnuts,eating hotdog,eating ice cream,eating spaghetti,eating watermelon,egg hunting,exercising arm,exercising with an exercise ball,extinguishing fire,faceplanting,feeding birds,feeding fish,feeding goats,filling eyebrows,finger snapping,fixing hair,flipping pancake,flying kite,folding clothes,folding napkins,folding paper,front raises,frying vegetables,garbage collecting,gargling,getting a haircut,getting a tattoo,giving or receiving award,golf chipping,golf driving,golf putting,grinding meat,grooming dog,grooming horse,gymnastics tumbling,hammer throw,headbanging,headbutting,high jump,high kick,hitting baseball,hockey stop,holding snake,hopscotch,hoverboarding,hugging,hula hooping,hurdling,hurling (sport),ice climbing,ice fishing,ice skating,ironing,javelin throw,jetskiing,jogging,juggling balls,juggling fire,juggling soccer ball,jumping into pool,jumpstyle dancing,kicking field goal,kicking soccer ball,kissing,kitesurfing,knitting,krumping,laughing,laying bricks,long jump,lunge,making a cake,making a sandwich,making bed,making jewelry,making pizza,making snowman,making sushi,making tea,marching,massaging back,massaging feet,massaging legs,massaging person's head,milking cow,mopping floor,motorcycling,moving furniture,mowing lawn,news anchoring,opening bottle,opening present,paragliding,parasailing,parkour,passing American football (in game),passing American football (not in game),peeling apples,peeling potatoes,petting animal (not cat),petting cat,picking fruit,planting trees,plastering,playing accordion,playing badminton,playing bagpipes,playing basketball,playing bass guitar,playing cards,playing cello,playing chess,playing clarinet,playing controller,playing cricket,playing cymbals,playing didgeridoo,playing drums,playing flute,playing guitar,playing harmonica,playing harp,playing ice hockey,playing keyboard,playing kickball,playing monopoly,playing organ,playing paintball,playing piano,playing poker,playing recorder,playing saxophone,playing squash or racquetball,playing tennis,playing trombone,playing trumpet,playing ukulele,playing violin,playing volleyball,playing xylophone,pole vault,presenting weather forecast,pull ups,pumping fist,pumping gas,punching bag,punching person (boxing),push up,pushing car,pushing cart,pushing wheelchair,reading book,reading newspaper,recording music,riding a bike,riding camel,riding elephant,riding mechanical bull,riding mountain bike,riding mule,riding or walking with horse,riding scooter,riding unicycle,ripping paper,robot dancing,rock climbing,rock scissors paper,roller skating,running on treadmill,sailing,salsa dancing,sanding floor,scrambling eggs,scuba diving,setting table,shaking hands,shaking head,sharpening knives,sharpening pencil,shaving head,shaving legs,shearing sheep,shining shoes,shooting basketball,shooting goal (soccer),shot put,shoveling snow,shredding paper,shuffling cards,side kick,sign language interpreting,singing,situp,skateboarding,ski jumping,skiing (not slalom or crosscountry),skiing crosscountry,skiing slalom,skipping rope,skydiving,slacklining,slapping,sled dog racing,smoking,smoking hookah,snatch weight lifting,sneezing,sniffing,snorkeling,snowboarding,snowkiting,snowmobiling,somersaulting,spinning poi,spray painting,spraying,springboard diving,squat,sticking tongue out,stomping grapes,stretching arm,stretching leg,strumming guitar,surfing crowd,surfing water,sweeping floor,swimming backstroke,swimming breast stroke,swimming butterfly stroke,swing dancing,swinging legs,swinging on something,sword fighting,tai chi,taking a shower,tango dancing,tap dancing,tapping guitar,tapping pen,tasting beer,tasting food,testifying,texting,throwing axe,throwing ball,throwing discus,tickling,tobogganing,tossing coin,tossing salad,training dog,trapezing,trimming or shaving beard,trimming trees,triple jump,tying bow tie,tying knot (not on a tie),tying tie,unboxing,unloading truck,using computer,using remote controller (not gaming),using segway,vault,waiting in line,walking the dog,washing dishes,washing feet,washing hair,washing hands,water skiing,water sliding,watering plants,waxing back,waxing chest,waxing eyebrows,waxing legs,weaving basket,welding,whistling,windsurfing,wrapping present,wrestling,writing,yawning,yoga,zumba"
    return actions.split(",")


def recognize_action_kinetics_400(prediction: torch.Tensor) -> List[str]:
    """
    Return the top 5 class names.
    Parameters:
        prediction: Get the probability for all classes.

    Returns:
        classnames: List of class ids from Kinetics-400 dataset is returned.

    """
    # Get top 5 class probabilities
    prediction = torch.topk(prediction.flatten(), 5).indices

    actions = get_class_name_kinetics_400()
    return [actions[pred] for pred in prediction]


class KineticsClassifierApp:
    """
    This class consists of light-weight "app code" that is required to
    perform end to end inference with an KineticsClassifier.

    For a given image input, the app will:
        * Pre-process the video (resize and normalize)
        * Run Video Classification
        * Return the probability of each class.
    """

    def __init__(self, model: KineticsClassifier):
        self.model = model

    def predict(self, path: str | Path) -> List[str]:
        """
        From the provided path of the video, predict probability distribution
        over the 400 Kinetics classes and return the class name.

        Parameters:
            path: Path to the raw video

        Returns:
            prediction: List[str] with top 5 most probable classes for a given video.
        """

        # Reads the video via provided path
        input_video = read_video_per_second(str(path))

        # Preprocess the video
        input_video = preprocess_video_kinetics_400(input_video)

        # Inference using mdoel
        raw_prediction = self.model(input_video.unsqueeze(0))

        return recognize_action_kinetics_400(raw_prediction)
