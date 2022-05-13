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
from utils.data_utils import get_dataset_info

from models.models import get_model
import wandb

ImageFile.LOAD_TRUNCATED_IMAGES = True

# SAVE_HANDLER = utils_log.get_log_filehandle(args, train_file_name)
SAVE_HANDLER = sys.stdout


def get_transforms():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def get_datasets_loaders(args, debugImages = False):
    print(args, file=SAVE_HANDLER, flush=True)

    image_transform = get_transforms()

    data_dir = "data/"
    dataset_cls, NUM_CLASSES, file_list_root, att_path = get_dataset_info(args.dataset_name)
    shuffles = {"train": True, "val": True, "test": False}

    ################ GET FROM USER CONFIG - TODO #####################
    file_lists = {}
    dsets = {}
    dset_loaders = {}
    dset_sizes = {}
    for phase in ["train", "val", "test"]:
        file_lists[phase] = "%s/%s_list_%s.txt" % (file_list_root, phase, args.dataset_name)
        dsets[phase] = dataset_cls(file_lists[phase], att_path, image_transform, data_dir)
        if debugImages:
            dsets[phase].show_images_on_subplot(labels=True, images=True)

        dset_loaders[phase] = torch.utils.data.DataLoader(
            dsets[phase],
            batch_size=args.batch_size,
            shuffle=shuffles[phase],
            num_workers=2,
            drop_last=True,
        )
        dset_sizes[phase] = len(dsets[phase])

    print("Dataset sizes:", file=SAVE_HANDLER)
    print(len(dsets["train"]), file=SAVE_HANDLER)
    print(len(dsets["val"]), file=SAVE_HANDLER)
    print(len(dsets["test"]), file=SAVE_HANDLER)

    return dsets, dset_loaders, dset_sizes, NUM_CLASSES


def train_one_epoch(model, criterion, epoch, optimizer, dset_loaders, dset_sizes, GPU, best_model_metrics, phases=('train', 'val')):
    best_model, best_acc, best_val_loss = best_model_metrics[0], best_model_metrics[1], best_model_metrics[2]
    print("Epoch %s" % epoch, file=SAVE_HANDLER, flush=True)
    for phase in phases:
        print("%s phase" % phase, file=SAVE_HANDLER, flush=True)
        iters = 0
        phase_epoch_corrects = [0, 0, 0, 0]
        phase_epoch_loss = 0
        if phase == "train":
            model.train()
            torch.set_grad_enabled(True)
        else:
            print("model eval", file=SAVE_HANDLER)
            model.eval_one_epoch()
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
                labels = labels_all
                if GPU:
                    labels = Variable(labels.long().cuda())

                outputs = model_outs
                calculated_loss += criterion(outputs, labels)

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

        print("Epoch loss: %s" % epoch_loss.item(), file=SAVE_HANDLER)
        print("Epoch accs: ", epoch_accs, file=SAVE_HANDLER)

        if args.wandblog:
            utils_log.log_wandb(phase, epoch, epoch_loss, epoch_accs)

        if phase == "val":
            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model = model

        best_model_metrics[0], best_model_metrics[1], best_model_metrics[2] = best_model, best_acc, best_val_loss


def eval_one_epoch(model, criterion, epoch, optimizer, dset_loaders, dset_sizes, GPU, best_model_metrics):
    train_one_epoch(model, criterion, epoch, optimizer, dset_loaders, dset_sizes, GPU, best_model_metrics, phases=('test'))


def save_models(args, SAVE_FILE_SUFFIX):
    file = os.path.basename(__file__)
    train_file_name = os.path.splitext(file)[0]
    if args.task == "combined":
        modelpath = "outputs/%s/saved_models/%s_model_%s_%s_%s.pt" % (
            args.experiment_out_name, train_file_name, args.arch, args.dataset_name, SAVE_FILE_SUFFIX)
    else:
        modelpath = "outputs/%s/saved_models/%s_%s_model_%s_%s_%s.pt" % (
            args.experiment_out_name, train_file_name, args.task, args.arch, args.dataset_name, SAVE_FILE_SUFFIX)
        # with open(modelpath,"wb") as F:
        #         torch.save(best_model, F)

    if args.wandblog:
        wandb.config.update({"model_path": modelpath})
        wandb.save(os.path.join(os.getcwd(), modelpath))


def train(args):
    utils_log.create_logging_folders(args)
    if args.wandblog:
        wandb.init(project='ood-generalisation')
        # ,               name='runnametest2')

    dsets, dset_loaders, dset_sizes, NUM_CLASSES = get_datasets_loaders(args, debugImages=True)

    SAVE_FILE_SUFFIX = args.save_file_suffix
    if args.start_checkpoint_path:
        print("Loading from %s" % args.start_checkpoint_path)
        model = torch.load(args.start_checkpoint_path)
        checkpoint_model_name = args.start_checkpoint_path.split("/")[-1].split(".pt")[0]
        SAVE_FILE_SUFFIX = "%s_%s" % (args.save_file_suffix, checkpoint_model_name)
    else:
        model = get_model(args.arch, NUM_CLASSES)

    criterion = nn.CrossEntropyLoss()

    GPU = torch.cuda.is_available()
    if GPU:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # best_model = model
    # best_acc = 0.0
    # best_val_loss = 100
    # best_model_metric = [best_model, best_acc, best_val_loss]
    #
    # for epoch in range(args.num_epochs):
    #     train_one_epoch(model, criterion, epoch, optimizer, dset_loaders, dset_sizes, GPU, best_model_metric)
    #     eval_one_epoch(model, criterion, epoch, optimizer, dset_loaders, dset_sizes, GPU, best_model_metric)
    #
    # save_models(args, SAVE_FILE_SUFFIX)
    print('Job completed')


if __name__ == "__main__":
    args = utils_log.parse_config()
    train(args)
