import argparse
import os
from datetime import datetime

import wandb
from torch import Tensor
import numpy as np


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--arch", type=str)
    parser.add_argument("--experiment_out_name", type=str)
    parser.add_argument("--save_file_suffix", type=str)
    parser.add_argument("--start_checkpoint_path", type=str)
    parser.add_argument("--task", type=str, default="combined")
    parser.add_argument("--wandblog", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    return args


def create_logging_folders(args):
    def create_folder(fol):
        if not os.path.isdir(fol):
            os.mkdir(fol)

    create_folder("outputs")
    create_folder("outputs/%s" % args.experiment_out_name)
    create_folder("outputs/%s/saved_models" % args.experiment_out_name)
    cur_dt = datetime.now()
    cur_dt = cur_dt.strftime("%y%m%d_%H%M%S")
    run_path = "outputs/{}/saved_models/benchmark-{}".format(args.experiment_out_name, cur_dt)
    create_folder(run_path)
    return run_path


def get_log_filehandle(args, train_file_name):
    DATASET_NAME = args.dataset_name
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    ARCH = args.arch
    SAVE_FILE_SUFFIX = args.save_file_suffix
    TASK = args.task

    if TASK == "combined":
        SAVE_FILE = "outputs/%s/logs/%s_%s_%s_%s_%s_%s.out" % (
            args.experiment_out_name,
            train_file_name,
            ARCH,
            DATASET_NAME,
            NUM_EPOCHS,
            BATCH_SIZE,
            SAVE_FILE_SUFFIX,
        )
    else:
        SAVE_FILE = "outputs/%s/logs/%s_%s_%s_%s_%s_%s_%s.out" % (
            args.experiment_out_name,
            train_file_name,
            TASK,
            ARCH,
            DATASET_NAME,
            NUM_EPOCHS,
            BATCH_SIZE,
            SAVE_FILE_SUFFIX,
        )
    SAVE_HANDLER = open(SAVE_FILE, "w")
    return SAVE_HANDLER


def log_wandb4(phase, epoch, epoch_loss, epoch_accs):
    logdict = {}
    logdict[phase + '_loss'] = epoch_loss.item()
    for id, acc in enumerate(epoch_accs):
        logdict[phase + '_acc{}'.format(id)] = acc

    logdict[phase+'_acc1_3'] = np.sqrt(epoch_accs[1]*epoch_accs[3])
    wandb.log(logdict, step=epoch)


def log_wandb(phase, epoch, d_metric):
    oldkeys = list(d_metric.keys())
    for old_key in oldkeys:
        d_metric['{}_{}'.format(phase, old_key)] = d_metric.pop(old_key)
    wandb.log(d_metric, step=epoch)


# util function for generating interactive image mask from components
def wb_mask(bg_img, pred_mask, true_mask, labels):
    res = []
    for i in range(len(bg_img)):
           res.append(wandb.Image(bg_img[i], masks={
            "prediction": dict(mask_data=pred_mask[i].squeeze(), class_labels=labels),
            "ground truth": dict(mask_data=true_mask[i].squeeze(), class_labels=labels)}))
    return res


def image2np(image: Tensor) -> np.ndarray:
    "Convert from torch style `image` to numpy/matplotlib style - borrowed from fast.ai"
    res = image.cpu().permute(0, 2, 3, 1).numpy()
    return res
