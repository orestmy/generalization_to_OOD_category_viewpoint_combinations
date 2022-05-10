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


args = utils_log.parse_config()
utils_log.create_logging_folders(args)

if args.wandblog:
    wandb.init(project='ood-generalisation')
               # ,               name='runnametest2')

DATASET_NAME = args.dataset_name
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
ARCH = args.arch
SAVE_FILE_SUFFIX = args.save_file_suffix
TASK = args.task

if args.start_checkpoint_path:
    checkpoint_model_name = args.start_checkpoint_path.split("/")[-1].split(".pt")[0]
    SAVE_FILE_SUFFIX = "%s_%s" % (SAVE_FILE_SUFFIX, checkpoint_model_name)

file = os.path.basename(__file__)
train_file_name = os.path.splitext(file)[0]

SAVE_HANDLER = utils_log.get_log_filehandle(args, train_file_name)
SAVE_HANDLER = sys.stdout


print(args, file=SAVE_HANDLER, flush=True)


image_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

data_dir = "data/"
dataset_cls, NUM_CLASSES, file_list_root, att_path = get_dataset_info(DATASET_NAME)
shuffles = {"train": True, "val": True, "test": False}

################ GET FROM USER CONFIG - TODO #####################
file_lists = {}
dsets = {}
dset_loaders = {}
dset_sizes = {}
for phase in ["train", "val", "test"]:
    file_lists[phase] = "%s/%s_list_%s.txt" % (file_list_root, phase, DATASET_NAME)
    dsets[phase] = dataset_cls(file_lists[phase], att_path, image_transform, data_dir)
    # dsets[phase].show_images_on_subplot_categories(4,4)
    dset_loaders[phase] = torch.utils.data.DataLoader(
        dsets[phase],
        batch_size=BATCH_SIZE,
        shuffle=shuffles[phase],
        num_workers=2,
        drop_last=True,
    )
    dset_sizes[phase] = len(dsets[phase])


print("Dataset sizes:", file=SAVE_HANDLER)
print(len(dsets["train"]), file=SAVE_HANDLER)
print(len(dsets["val"]), file=SAVE_HANDLER)
print(len(dsets["test"]), file=SAVE_HANDLER)


if args.start_checkpoint_path:
    print("Loading from %s" % args.start_checkpoint_path)
    model = torch.load(args.start_checkpoint_path)
else:
    model = get_model(ARCH, NUM_CLASSES)

multi_losses = [nn.CrossEntropyLoss() for _ in range(4)]
GPU = torch.cuda.is_available()
if GPU:
    model.cuda()
    if ARCH == "MULTITASKRESNET":
        model.fcs = [i.cuda() for i in model.fcs]

optimizer = optim.Adam(model.parameters(), lr=0.001)


def cost_weight_task():
    if TASK == "rotation":
        return [0.0, 1.0, 0.0, 0.0]
    if TASK == "car_model":
        return [0.0, 0.0, 0.0, 1.0]
    elif TASK == "combined":
        return [0.0, 1.0, 0.0, 1.0]


best_model = model
best_model_gm = model
best_acc = 0.0
best_val_loss = 100

for epoch in range(NUM_EPOCHS):
    print("Epoch %s" % epoch, file=SAVE_HANDLER, flush=True)
    cost_weights = cost_weight_task()
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
            utils_log.log_wandb(phase, epoch, epoch_loss, epoch_accs)

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

if TASK == "combined":
    modelpath = "outputs/%s/saved_models/%s_model_%s_%s_%s.pt"%(args.experiment_out_name, train_file_name, ARCH, DATASET_NAME, SAVE_FILE_SUFFIX)
else:
    modelpath = "outputs/%s/saved_models/%s_%s_model_%s_%s_%s.pt"%(args.experiment_out_name, train_file_name, TASK, ARCH, DATASET_NAME, SAVE_FILE_SUFFIX)
with open(modelpath,"wb") as F:
        torch.save(best_model, F)

if args.wandblog:
    wandb.config.update({"model_path": modelpath})
    wandb.save(os.path.join(os.getcwd(),modelpath))
print('Job completed, best model saved')