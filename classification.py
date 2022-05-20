from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import os
from PIL import ImageFile

import sys
import utils.logger_utils as utils_log
from segmentation import get_datasets_loaders, save_models
from utils.data_utils import get_dataset_info

from models.models import get_model
import wandb

from utils.random_seed import fix_random_seed

ImageFile.LOAD_TRUNCATED_IMAGES = True


SAVE_HANDLER = sys.stdout


def cost_weight_task(TASK):
    if TASK == "rotation":
        return [0.0, 1.0, 0.0, 0.0]
    if TASK == "car_model":
        return [0.0, 0.0, 0.0, 1.0]
    elif TASK == "combined":
        return [0.0, 1.0, 0.0, 1.0]



def train(args):
    utils_log.create_logging_folders(args)
    if args.wandblog:
        wandb.init(project='ood-generalisation', name='cls_seen-{}_task-{}'.format(args.dataset_name, args.task))

    dsets, dset_loaders, dset_sizes, NUM_CLASSES = get_datasets_loaders(args, debugImages=False)

    SAVE_FILE_SUFFIX = args.save_file_suffix
    if args.start_checkpoint_path:
        print("Loading from %s" % args.start_checkpoint_path)
        model = torch.load(args.start_checkpoint_path)
        checkpoint_model_name = args.start_checkpoint_path.split("/")[-1].split(".pt")[0]
        SAVE_FILE_SUFFIX = "%s_%s" % (args.save_file_suffix, checkpoint_model_name)
    else:
        model = get_model(args.arch, NUM_CLASSES)

    multi_losses = [nn.CrossEntropyLoss() for _ in range(4)]
    GPU = torch.cuda.is_available()
    if GPU:
        model.cuda()
        if args.arch == "MULTITASKRESNET":
            model.fcs = [i.cuda() for i in model.fcs]

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_model = model
    best_model_gm = model
    best_acc = 0.0
    best_val_loss = 100

    for epoch in range(args.num_epochs):
        print("Epoch %s" % epoch, file=SAVE_HANDLER, flush=True)
        cost_weights = cost_weight_task(args.task)
        for phase in ("train", "val", "test"):
            print("%s phase" % phase, file=SAVE_HANDLER, flush=True)
            iters = 0
            phase_epoch_corrects = [0, 0, 0, 0]
            phase_epoch_loss = 0
            if phase == "train":
                model.train()
                torch.set_grad_enabled(True)
            else:
                print("model eval", file=SAVE_HANDLER)
                model.eval()
                torch.set_grad_enabled(False)

            for data in dset_loaders[phase]:
                inputs, labels_all, paths = data
                if GPU:
                    inputs = Variable(inputs.float().cuda())

                optimizer.zero_grad()
                model_outs = model(inputs)

                calculated_loss = 0

                batch_corrects = [0, 0, 0, 0]
                for i in range(4):
                    labels = labels_all[:, i]
                    if GPU:
                        labels = Variable(labels.long().cuda())

                    loss = multi_losses[i]
                    outputs = model_outs[i]
                    calculated_loss += cost_weights[i] * loss(outputs, labels)
                    _, preds = torch.max(outputs.data, 1)
                    batch_corrects[i] = torch.sum(preds == labels.data)
                    phase_epoch_corrects[i] += batch_corrects[i]

                phase_epoch_loss += calculated_loss

                if phase == "train":
                    calculated_loss.backward()
                    optimizer.step()

                if iters % 50 == 0:
                    print("Epoch %s, Iters %s" % (epoch, iters), file=SAVE_HANDLER)
                iters += 1

            epoch_loss = phase_epoch_loss / dset_sizes[phase]
            epoch_accs = [float(i) / dset_sizes[phase] for i in phase_epoch_corrects]
            gm_epoch_accs = 1
            for i in range(4):
                gm_epoch_accs = gm_epoch_accs * epoch_accs[i]
            if gm_epoch_accs > best_acc:
                best_acc = gm_epoch_accs
                best_model_gm = model

            print("Epoch loss: %s" % epoch_loss.item(), file=SAVE_HANDLER)
            print("Epoch accs: ", epoch_accs, file=SAVE_HANDLER)

            if args.wandblog:
                utils_log.log_wandb4(phase, epoch, epoch_loss, epoch_accs)

            if phase == "val":
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model = model

                gm_epoch_accs = 1
                for i in range(2):
                    gm_epoch_accs = gm_epoch_accs * epoch_accs[i]

                if gm_epoch_accs > best_acc:
                    best_acc = gm_epoch_accs
                    best_model_gm = model

    save_models(best_model, args, SAVE_FILE_SUFFIX)
    print('Job completed, best model saved')


if __name__ == "__main__":
    fix_random_seed(42, True)
    args = utils_log.parse_config()
    train(args)
