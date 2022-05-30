from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os
from PIL import ImageFile
import numpy as np
import sys
import utils.logger_utils as utils_log
from utils.data_utils import get_dataset_info

from models.models import get_model
import wandb
from metrics.segmentation_metrics import IoU_Metric
from utils.random_seed import fix_random_seed

ImageFile.LOAD_TRUNCATED_IMAGES = True

# SAVE_HANDLER = utils_log.get_log_filehandle(args, train_file_name)
SAVE_HANDLER = sys.stdout
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


def get_transforms(inverse=False):
    # inverse transform is needed to plot images in original colors (debug reasons)
    # run with flag False to get proper CV image transform
    if not inverse:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                       transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])
        return invTrans


def get_datasets_loaders(args, debugImages=False):
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
            dsets[phase].show_images_on_subplot(labels=True, images=False)

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


def train_one_epoch(model, criterion, epoch, optimizer, dset_loaders, dset_sizes, GPU, best_model_metrics,
                    metric_tracker,
                    phases=('train', 'val')):
    best_model, best_acc, best_val_loss = best_model_metrics[0], best_model_metrics[1], best_model_metrics[2]
    print("Epoch %s" % epoch, file=SAVE_HANDLER, flush=True)
    for phase in phases:
        print("%s phase" % phase, file=SAVE_HANDLER, flush=True)
        iters = 0
        phase_epoch_loss = 0

        if phase == "train":
            model.train()
            torch.set_grad_enabled(True)
        else:
            print("model eval", file=SAVE_HANDLER)
            model.eval()
            torch.set_grad_enabled(False)

        metric_tracker.reset()
        for data in dset_loaders[phase]:
            inputs, labels, paths = data
            labels = labels.long().squeeze()

            if GPU:
                inputs = inputs.float().cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            pred = model(inputs)

            calculated_loss = criterion(pred, labels)
            phase_epoch_loss += calculated_loss
            metric_tracker.add(pred.detach(), labels.detach())

            if phase == "train":
                calculated_loss.backward()
                optimizer.step()

            if iters % 50 == 0:
                print("Epoch %s, Iters %s" % (epoch, iters), file=SAVE_HANDLER)
            iters += 1

        epoch_loss = phase_epoch_loss / dset_sizes[phase]
        d_metric = metric_tracker.finalise()
        d_metric['loss'] = epoch_loss

        print("Epoch loss: %s" % epoch_loss.item(), file=SAVE_HANDLER)
        print("Epoch mIOU: ", d_metric['miou'], file=SAVE_HANDLER)

        if args.wandblog:
            utils_log.log_wandb(phase, epoch, d_metric)

        if phase == "val":
            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model = model

        best_model_metrics[0], best_model_metrics[1], best_model_metrics[2] = best_model, best_acc, best_val_loss
        model.__setattr__('last_epoch', epoch)


def eval_one_epoch(model, criterion, epoch, optimizer, dset_loaders, dset_sizes, GPU, best_model_metrics,
                   metric_tracker):
    train_one_epoch(model, criterion, epoch, optimizer, dset_loaders, dset_sizes, GPU, best_model_metrics,
                    metric_tracker,
                    phases=('test',))


def on_epoch_end(args, model, dsets, GPU, num_to_log=16):
    model.eval()
    torch.set_grad_enabled(False)
    for dset in dsets:
        dload = torch.utils.data.DataLoader(dset, batch_size=num_to_log, shuffle=False, num_workers=0)
        input_batch = next(iter(dload))

        mask_list = []
        inputs, labels, paths = input_batch
        # the raw background image as a numpy array
        # show images
        # for bg_im in inputs:
        #     plt.imshow(bg_im.cpu().permute(1, 2, 0).numpy())
        #     plt.show()
        # for path in paths:
        #     img = mpimg.imread(path)
        #     imgplot = plt.imshow(img)
        #     plt.show()
        #
        # for label in labels:
        #     plt.imshow(label.cpu().permute(1, 2, 0).numpy())
        #     plt.show()

        bg_image = utils_log.image2np(get_transforms(True)(inputs) * 255).astype(np.uint8)
        # run the model on that image

        labels = labels.long().squeeze()
        if GPU:
            inputs = inputs.float().cuda()
            labels = labels.cuda()

        prediction_mask = model(inputs)
        prediction_mask = torch.softmax(prediction_mask, dim=1)
        # for label in prediction_mask:
        #     plt.imshow(np.expand_dims(label.cpu().permute(1, 2, 0).numpy()[:, :, 0] > 0.5, 2).astype(np.uint8))
        #     # plt.imshow((label.cpu().permute(1, 2, 0).numpy()[:, :, 0] > 0.5).astype(np.uint8))
        #
        #     plt.show()
        prediction_mask = torch.argmax(prediction_mask, dim=1)
        prediction_mask = np.round(utils_log.image2np(torch.unsqueeze(prediction_mask, 1))).astype(np.uint8)

        # ground truth mask
        true_mask = np.round(utils_log.image2np(torch.unsqueeze(labels, 1))).astype(np.uint8)
        # for label in true_mask:
        #     plt.imshow(label)
        #     plt.show()

        mask_list.extend(utils_log.wb_mask(bg_image, prediction_mask, true_mask, labels=dset.labels()))

        # log all composite images to W&B
        if args.wandblog:
            wandb.log({"predictions": mask_list})


def save_models(best_model, args, SAVE_FILE_SUFFIX):
    file = os.path.basename(__file__)
    train_file_name = os.path.splitext(file)[0]
    if args.task == "combined":
        modelpath = "outputs/%s/saved_models/%s_model_%s_%s_%s.pt" % (
            args.experiment_out_name, train_file_name, args.arch, args.dataset_name, SAVE_FILE_SUFFIX)
    else:
        modelpath = "outputs/%s/saved_models/%s_%s_model_%s_%s_%s.pt" % (
            args.experiment_out_name, train_file_name, args.task, args.arch, args.dataset_name, SAVE_FILE_SUFFIX)
        with open(modelpath, "wb") as F:
            torch.save(best_model, F)

    if args.wandblog:
        wandb.config.update({"model_path": modelpath})
        wandb.save(os.path.join(os.getcwd(), modelpath))


def train(args):
    utils_log.create_logging_folders(args)
    if args.wandblog:
        wandb.init(project='ood-generalisation', name='segm_seen-{}_task-{}'.format(args.dataset_name, args.task))

    dsets, dset_loaders, dset_sizes, NUM_CLASSES = get_datasets_loaders(args, debugImages=False)

    SAVE_FILE_SUFFIX = args.save_file_suffix
    if args.start_checkpoint_path:
        print("Loading from %s" % args.start_checkpoint_path)
        model = torch.load(args.start_checkpoint_path)
        checkpoint_model_name = args.start_checkpoint_path.split("/")[-1].split(".pt")[0]
        # SAVE_FILE_SUFFIX = "%s_%s" % (args.save_file_suffix, checkpoint_model_name)
    else:
        model = get_model(args.arch, NUM_CLASSES)

    criterion = nn.CrossEntropyLoss()

    # device = torch.device("cpu" if not torch.cuda.is_available() else args.device)
    # torch.cuda.set_device(args.device)
    GPU = torch.cuda.is_available()
    if GPU:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_model = model
    best_acc = 0.0
    best_val_loss = 100
    best_model_metric = [best_model, best_acc, best_val_loss]

    metric_tracker = IoU_Metric(num_classes=NUM_CLASSES)

    for epoch in range(args.num_epochs):
        # train_one_epoch(model, criterion, epoch, optimizer, dset_loaders, dset_sizes, GPU, best_model_metric,
        #                 metric_tracker)
        # eval_one_epoch(model, criterion, epoch, optimizer, dset_loaders, dset_sizes, GPU, best_model_metric,
        #                metric_tracker)
        on_epoch_end(args, model, [dsets['test']], GPU)
    #
    # save_models(best_model_metric[0], args, SAVE_FILE_SUFFIX)
    print('Job completed')


if __name__ == "__main__":
    fix_random_seed(42, True)
    args = utils_log.parse_config()
    train(args)


def dummy_CE():
    loss = nn.CrossEntropyLoss()
    input = torch.randn(10, 1, 3, 5, requires_grad=True)
    target = torch.empty(10, 3, dtype=torch.long).random_(5)
    output = loss(input, target)
    output.backward()

    # input = torch.randn(3, 5, requires_grad=True)
    #
    # target = torch.randn(3, 5).softmax(dim=1)
    # output = loss(input, target)
    output.backward()
