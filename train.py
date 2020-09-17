"""Train the model"""

import argparse
import logging
import os
import torch
import torch.nn as nn
from tqdm import tqdm as tqdm
from transformers import AdamW
import utils
import shutil
import model.net as net
from transformers import get_linear_schedule_with_warmup
from model.data_loader import DoulingoDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from evaluation import evaluate_model
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/Original',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/marian',
                    help="Directory containing params.json")
parser.add_argument('--tensorboard_dir', default='experiments/marian/runs',
                    help="Directory  containing tensorboard summaries")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def train(epoch):
    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(train_data, total=len(train_data)) as t:
        for step, (src_lang_batch, trg_lang_batch) in enumerate(t):

            # compute model loss
            loss = model(src_lang_batch, trg_lang_batch)
            loss = loss.sum()

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # performs updates using calculated gradients
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

            # update the average loss
            loss_avg.update(loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

            # store summaries only once in a while
            if step % model_params.save_summary_steps == 0:
                # Adding summaries to tensorboard
                tb.add_scalar("Loss/train", loss_avg(), len(train_data) * epoch + step)

    logging.info(f"- Finished Full Epoch {epoch}, Average Loss:{loss_avg()}")


def train_and_evaluate():
    # reload weights from restore_file if specified
    if args.restore_file is not None:
        restore_path = os.path.join(
            args.tensorboard_dir, args.restore_file + '.pth.tar')
        if os.path.exists(restore_path):
            logging.info("Restoring parameters from {}".format(restore_path))
            utils.load_checkpoint(restore_path, model.module, optimizer)

    best_val_f1 = 0.0

    for epoch in range(1, model_params.num_epochs+1):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch, model_params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(epoch)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate_model(model.module, val_dataset, report_dir=args.tensorboard_dir)

        logging.info(val_metrics)

        val_f1 = val_metrics['Weighted Macro F1']
        is_best = val_f1 >= best_val_f1

        # Save weights
        utils.save_checkpoint({'epoch': epoch,
                               'state_dict': model.module.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=args.tensorboard_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best macro f1")
            best_val_f1 = val_f1

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                args.tensorboard_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)
            last_report_path = os.path.join(args.tensorboard_dir, 'report.txt')
            best_report_path = os.path.join(args.tensorboard_dir, 'best_report.txt')
            if os.path.exists(last_report_path):
                shutil.copy(last_report_path, best_report_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            args.tensorboard_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

        # last_report_path = os.path.join(model_dir, f"report_{epoch}.txt")
        # utils.save_report(report, last_report_path)


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    model_params_json_path = os.path.join(args.model_dir, 'params.json')
    data_params_json_path = os.path.join(args.data_dir, 'params.json')
    assert os.path.isfile(
        model_params_json_path), "No json configuration file found at {}".format(model_params_json_path)
    assert os.path.isfile(
        data_params_json_path), "No json configuration file found at {}".format(data_params_json_path)

    data_params = utils.DataParams.from_json(data_params_json_path)
    model_params = utils.Params(cuda=torch.cuda.is_available(), src='en', trg='hu')
    model_params.update(model_params_json_path)

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if model_params.cuda:
        torch.cuda.manual_seed(230)

    # Create tensorboard summary writer
    tb = SummaryWriter(args.tensorboard_dir)

    # Set the logger
    utils.set_logger(os.path.join(args.tensorboard_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # load data train and validation data
    train_dataset = DoulingoDataset(data_params)
    weights = train_dataset.samples_weights(scale_factor=50)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights))
    train_data = DataLoader(train_dataset,
                            sampler=sampler,
                            batch_size=model_params.batch_size)

    val_dataset = DoulingoDataset(data_params, split='val')
    val_data = DataLoader(val_dataset, shuffle=False)

    logging.info("- done.")

    # Define the model and optimizer
    model = net.Net(model_params)
    model = nn.DataParallel(model)
    if model_params.cuda:
        model = model.cuda()

    # model._reset_parameters()
    optimizer = AdamW(model.parameters(),
                      lr=model_params.learning_rate,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=model_params.adam_eps  # args.adam_epsilon  - default is 1e-8.
                      )
    # Create the learning rate scheduler.
    total_steps = len(train_data) * model_params.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=1,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(model_params.num_epochs))
    train_and_evaluate()
