from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
import os
from PIL import ImageFile

import sys
import utils.logger_utils as utils_log
from segmentation import get_datasets_loaders, save_models
from utils.data_utils import get_dataset_info

from models.models import get_model
import wandb

from utils.helper_utils import get_digits
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
    run_path = utils_log.create_logging_folders(args)
    if args.wandblog:
        wandb.init(project='ood-generalisation', name='cls_seen{}_task-{}'.format(get_digits(args.dataset_name), args.task))

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
        torch.cuda.set_device(args.device)
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

    on_epoch_end(args, model, dset_loaders, ['val', 'test'], GPU)

    save_models(run_path, best_model, args, SAVE_FILE_SUFFIX)
    print('Job completed, best model saved')

def on_epoch_end(args, model, dloader, phases, GPU):
    model.eval()
    torch.set_grad_enabled(False)
    embeddings_l = []
    labels_l = []

    for phase in phases:
        for data in dloader[phase]:
            inputs, labels, paths = data

            labels = labels.long().squeeze()
            if GPU:
                model.cuda()
                inputs = inputs.float().cuda()

            model(inputs)

            batch_embedding = model.embedding.cpu().numpy()
            embeddings_l.extend(batch_embedding.tolist())
            batch_labels1 = labels[:, 1].numpy()
            batch_labels12 = labels[:, 3].numpy()
            string_lab = ['{}_{}_{}'.format(batch_labels1[i], batch_labels12[i], phase) for i in range(len(batch_labels12))]
            labels_l.extend(string_lab)

    cols = ['f{}'.format(i) for i in range(len(embeddings_l[0]))]
    for i in range(len(embeddings_l)):
        embeddings_l[i].append(labels_l[i])
        embeddings_l[i].append(labels_l[i].split('_')[-1])

    cols.append('label1')
    cols.append('phase')

    if args.wandblog:
        wandb.log({
            "embeddings": wandb.Table(
                columns=cols,
                data=embeddings_l
            )
        })

def get_embedding_from_wandbjson():
    # json file is stored at the end of segmentation run in Wandb
    file = 'embeddings/embeddings-combined.json'
    import json
    with open(file, 'r') as j:
        embeddings = json.load(j)
    feat = embeddings['data']
    return feat

def embeddings_to_tsv():
    # create embeddings for t-SNE tensorboard projector visualisation.
    # https://projector.tensorflow.org/ - append Q_id and Same as header for labels.tsv
    filename = 'embeddings/viz_combined/features_full_combined.tsv'
    filename_label = 'embeddings/viz_combined/labels_full_combined.tsv'
    feat = get_embedding_from_wandbjson()
    n = len(feat[0])
    with open(filename, 'a+') as embed_file, open(filename_label, 'a+') as label_file:
        for i in range(len(feat)):
            embedding = feat[i][:n-2]
            embedding_str = ''.join(["{:.1f}".format(num) + '\t' for num in embedding])
            embed_file.write(embedding_str + '\n')
            triple = feat[i][n-2].split('_')
            label_file.write(feat[i][n-2] + '\t' + feat[i][n-1] + '\t' + triple[0] + '\t' + triple[1] + '\n')


if __name__ == "__main__":
    fix_random_seed(42, True)
    args = utils_log.parse_config()
    train(args)
    # embeddings_to_tsv()
