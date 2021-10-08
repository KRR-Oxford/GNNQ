import torch
import argparse
import torchmetrics
import optuna
from optuna.trial import TrialState
import os
from datetime import datetime
import json
import pickle
from torch.utils.tensorboard import SummaryWriter
from model import HGNN
from data_utils import create_data_object
from test import test

# Todo:
#  - Start encoding unary predicates in the initial feature vectors - different dim for initial feature vector and hidden states?
#  - Think about how and where to construct path for models
#  - Double check that root of query answer is in first position
#  - Double check that ids for sub-queries are correct
#  - Clean up and comment functions in the data_util.py
#  - Double check behavior if subquery does not have answers on training data
#  - Think about rules with different body structures
#  - Change code such that val and test use the same code
#  - Change code such that for param optimisation the query answers do not have to be computed every time
#  - Use a file to specify the head relations in the data generation procedure
#  - Evaluate best trial on test data

# Hyperparameters used in this function can not be tuned with optuna
def prep_data(data_directories, relation2id=None):
    data = []
    for directory in data_directories:
        data_object, relation2id = create_data_object(os.path.join(directory, 'graph.nt'), os.path.join(directory, 'corrupted_graph.nt'),
                                                      args.query_string, args.aug, args.max_num_subquery_vars, relation2id)
        data.append(data_object)
    return data, relation2id


def train(device, train_data, val_data, log_directory, model_directory, args, trial=None):

    writer = SummaryWriter(log_directory)

    base_dim = args.base_dim
    num_layers = args.num_layers
    epochs = args.epochs
    learning_rate = args.lr
    lr_scheduler_step_size = args.lr_scheduler_step_size
    negative_slope = args.negative_slope
    positive_sample_weight = args.positive_sample_weight

    if trial:
        base_dim = trial.suggest_int('base_dim', 8, 32)
        num_layers = trial.suggest_int('num_layers', 1, 4)
        learning_rate = trial.suggest_float("lr", 0.001, 0.1, step=0.001)
        lr_scheduler_step_size = trial.suggest_int('lr_scheduler_step_size', 1, 10)
        positive_sample_weight = trial.suggest_int('positive_sample_weight', 1, 20)
        negative_slope = trial.suggest_float('negative_slope', 0.01, 0.2, step=0.01)
    else:
        with open(os.path.join(log_directory, 'config.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    model = HGNN(len(train_data[0]['x'][0]), base_dim, train_data[0]['num_edge_types_by_shape'], num_layers)
    model.to(device)
    for param in model.parameters():
        print(type(param.data), param.size())

    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model))

    # Adam optimizer already updates the learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=0.5)

    # Needs to be defined as module in the HGNN class to automatically move to GPU
    train_accuracy = torchmetrics.Accuracy(threshold=0.5)
    train_precision = torchmetrics.Precision(threshold=0.5)
    train_recall = torchmetrics.Recall(threshold=0.5)
    val_accuracy = torchmetrics.Accuracy(threshold=0.5)
    val_precision = torchmetrics.Precision(threshold=0.5)
    val_recall = torchmetrics.Recall(threshold=0.5)

    for epoch in range(epochs):
        print('Epoch-{0} lr: {1}'.format(epoch, lr_scheduler.get_last_lr()))
        model.train()
        optimizer.zero_grad()
        total_train_loss = 0
        for data_object in train_data:
            pred = model(data_object['x'], data_object['hyperedge_indices'], data_object['hyperedge_types'],
                         logits=True, negative_slope=negative_slope).flatten()
            # Weigh false positive samples from the previous epoch higher to address bad recall
            # We could also just use a small fraction of negative samples
            sample_weights_train = positive_sample_weight * data_object['y'] + torch.ones(len(data_object['y']))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, data_object['y'],
                                                                        weight=sample_weights_train)
            loss.backward()
            total_train_loss = total_train_loss + loss
            print('Train loss')
            print(loss)
            pred = torch.sigmoid(pred)
            train_accuracy(pred, data_object['y'].int())
            train_precision(pred, data_object['y'].int())
            train_recall(pred, data_object['y'].int())
        optimizer.step()

        model.eval()

        # ToDo: Add auc score as metric
        acc = train_accuracy.compute().item()
        pre = train_precision.compute().item()
        re = train_recall.compute().item()
        print('Train')
        print('Accuracy ' + str(acc))
        print('Precision ' + str(pre))
        print('Recall ' + str(re))
        writer.add_scalar('Loss training', total_train_loss, epoch)
        writer.add_scalar('Precision training', pre, epoch)
        writer.add_scalar('Recall training', re, epoch)
        train_accuracy.reset()
        train_precision.reset()
        train_recall.reset()

        if (epoch != 0) and (epoch % args.val_epochs == 0):
            total_loss = 0
            for data_object in val_data:
                pred = model(data_object['x'], data_object['hyperedge_indices'], data_object['hyperedge_types'],
                             negative_slope=negative_slope).flatten()
                # Weigh false positive samples from the previous epoch higher to address bad recall
                sample_weights_val = args.positive_sample_weight * data_object['y'] + torch.ones(len(data_object['y']))
                total_loss = total_loss + torch.nn.functional.binary_cross_entropy(pred, data_object['y'],
                                                                                   weight=sample_weights_val)
                val_accuracy(pred, data_object['y'].int())
                val_precision(pred, data_object['y'].int())
                val_recall(pred, data_object['y'].int())

            if trial:
                trial.report(total_loss, epoch)
            lr_scheduler.step()

            print('Val')
            print(total_loss)
            val_acc = val_accuracy.compute().item()
            val_pre = val_precision.compute().item()
            val_re = val_recall.compute().item()
            print('Accuracy ' + str(val_acc))
            print('Precision ' + str(val_pre))
            print('Recall ' + str(val_re))
            writer.add_scalar('Loss val', total_loss, epoch)
            writer.add_scalar('Precision val', pre, epoch)
            writer.add_scalar('Recall val', re, epoch)
            val_accuracy.reset()
            val_precision.reset()
            val_recall.reset()

            if trial and trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    if trial:
        torch.save(model.state_dict(), os.path.join(model_directory, 'trial{}.pt'.format(trial.number)))
    else:
        torch.save(model.state_dict(), os.path.join(model_directory, 'model.pt'))
    # Report best metric -- can this be different from the metric used for trial report


def objective(trial, device, train_data, val_data, log_directory, model_directory, args):
    loss = train(device, train_data, val_data, log_directory, model_directory, args, trial)
    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bla bla')
    parser.add_argument('--query_string', type=str,
                        default='SELECT distinct ?v0 WHERE { ?v0  <http://schema.org/caption> ?v1 . ?v0   <http://schema.org/text> ?v2 . ?v0 <http://schema.org/contentRating> ?v3 . ?v0   <http://purl.org/stuff/rev#hasReview> ?v4 .  ?v4 <http://purl.org/stuff/rev#title> ?v5 . ?v4  <http://purl.org/stuff/rev#reviewer> ?v6 . ?v7 <http://schema.org/actor> ?v6 . ?v7 <http://schema.org/language> ?v8  }')
    parser.add_argument('--train_data', type=str, nargs='+', default=['datasets/dataset1/'])
    parser.add_argument('--val_data', type=str, nargs='+', default=['datasets/dataset2/'])
    parser.add_argument('--test_data', type=str, nargs='+', default=['datasets/dataset3/'])
    parser.add_argument('--aug', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--max_num_subquery_vars', type=int, default=5)
    parser.add_argument('--pretrained_model', type=str, default='')
    parser.add_argument('--relation2id', type=str, default='')
    parser.add_argument('--base_dim', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--val_epochs', type=int, default=10)
    parser.add_argument('--lr', type=int, default=0.00625)
    parser.add_argument('--lr_scheduler_step_size', type=int, default=10)
    parser.add_argument('--negative_slope', type=int, default=0.1)
    parser.add_argument('--positive_sample_weight', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default='runs/')
    parser.add_argument('--hyperparam_tune', action='store_true', default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    now = datetime.now()
    date_time = now.strftime("%d_%m_%Y_%H:%M:%S")
    current_directory = os.getcwd()
    log_directory = os.path.join(current_directory, args.log_dir + date_time)
    model_directory = os.path.join(log_directory, 'models')

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    if args.relation2id:
        with open(args.relations2id, 'rb') as f:
            relation2id = pickle.load(f)
        train_data, _  = prep_data(args.train_data, relation2id)
        val_data, _ = prep_data(args.train_data, relation2id)
    else:
        train_data, relation2id = prep_data(args.train_data)
        val_data, _ = prep_data(args.train_data, relation2id)
        with open(os.path.join(model_directory, 'relation2id.pickle'), 'wb') as f:
            pickle.dump(relation2id, f)

    if not args.hyperparam_tune:
        train(device, train_data, val_data, log_directory, model_directory, args)
    else:
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=30, interval_steps=1))
        study.optimize(lambda trial: objective(trial, device, train_data, val_data, log_directory, model_directory, args), n_trials=100)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        trial = study.best_trial
        print("Best trial is trial number {}".format(trial.number))

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    if args.test:
        print('Start testing')
        test(args.test_data, args.query_string, model_directory, args.base_dim, args.num_layers, args.negative_slope, args.aug, args.max_num_subquery_vars, device)


