import argparse
import os
import wandb


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
    args = parser.parse_args()
    return args

def create_logging_folders(args):
    def create_folder(fol):
        if not os.path.isdir(fol):
            os.mkdir(fol)

    create_folder("outputs")
    create_folder("outputs/%s" % args.experiment_out_name)
    create_folder("outputs/%s/saved_models" % args.experiment_out_name)
    create_folder("outputs/%s/accuracies" % args.experiment_out_name)
    create_folder("outputs/%s/logs" % args.experiment_out_name)
    create_folder("outputs/%s/losses" % args.experiment_out_name)

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

def log_wandb(phase, epoch, epoch_loss, epoch_accs):
    logdict = {}
    logdict[phase + '_loss'] = epoch_loss.item()
    for id, acc in enumerate(epoch_accs):
        logdict[phase + '_acc{}'.format(id)] = acc
    wandb.log(logdict, step = epoch)