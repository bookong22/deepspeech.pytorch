# import argparse
import json
# import os
# import random
# import time

# import numpy as np
# import torch.distributed as dist
# import torch.utils.data.distributed
import torch    # 20200113    torch.nn.CTCLoss

from data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler, DistributedBucketingSampler
from decoder import GreedyDecoder
# from logger import VisdomLogger, TensorBoardLogger
# from model import DeepSpeech, supported_rnns
# from test import evaluate
# from utils import reduce_tensor, check_loss

if __name__ == '__main__':
    labels_path = "labels.json"
    with open(labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))
    print("labels : ", labels)
    decoder = GreedyDecoder(labels)
    print("decoder : ", decoder)
    audio_conf = dict(sample_rate=args.sample_rate,
                      window_size=args.window_size,
                      window_stride=args.window_stride,
                      window=args.window,
                      noise_dir=args.noise_dir,
                      noise_prob=args.noise_prob,
                      noise_levels=(args.noise_min, args.noise_max))
    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, augment=args.augment)
    train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)

