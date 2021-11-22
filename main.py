import pickle

import torch
import argparse
import torchmetrics
import optuna
from optuna.trial import TrialState
import os
from datetime import datetime
import json
from torch.utils.tensorboard import SummaryWriter
from model import HGNN
from data_utils import generate_subqueries, prep_data
from eval import eval, compute_metrics


# Todo:
#  - Encode unary predicates in the initial feature vectors - different dim for initial feature vector and hidden states
#  - Double check that root of query answer is in first position
#  - Double check behavior if a subquery does not have answers on training data
#  - Use a file to specify the head relations in the data generation procedure
#  - Add or max as aggregation function?


def train(device, feat_dim, shapes_dict, train_data, val_data, log_directory, model_directory, args, subqueries=None,
          summary_writer=None, trial=None):
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

    model = HGNN(args.query_string, feat_dim, base_dim, shapes_dict, num_layers,
                 negative_slope, args.max_aggr, args.monotonic, subqueries)
    model.to(device)
    for name, param in model.named_parameters():
        print(name, type(param.data), param.size())

    # Adam optimizer already updates the learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=0.5)

    # Needs to be defined as module in the HGNN class to automatically move to GPU
    threshold = 0.5
    train_accuracy = torchmetrics.Accuracy(threshold=threshold)
    train_precision = torchmetrics.Precision(threshold=threshold)
    train_recall = torchmetrics.Recall(threshold=threshold)

    for epoch in range(epochs):
        print('Epoch-{0} with lr {1}'.format(epoch, lr_scheduler.get_last_lr()))
        model.train()
        optimizer.zero_grad()
        total_train_loss = 0
        batch = [train_data[i] for i in torch.randperm(len(train_data))[:args.batch_size]]
        print('Training!')
        for data_object in batch:
            pred = model(data_object['x'], data_object['indices_dict'], logits=True).flatten()
            # positive_pred = torch.zeros_like(pred)
            # positive_pred[pred >= 0.5] = 1
            # false_positive_pred = torch.clamp(positive_pred - data_object['y'], min=0)
            # We could also just use a small fraction of negative samples
            sample_weights_train = positive_sample_weight * data_object['y'] + (
                    torch.ones(len(data_object['y'])) - data_object['y'])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, data_object['y'],
                                                                        weight=sample_weights_train)
            loss.backward()
            total_train_loss = total_train_loss + loss
            print('Loss: ' + str(loss.item()))
            pred = torch.sigmoid(pred)
            train_accuracy(pred, data_object['y'].int())
            train_precision(pred, data_object['y'].int())
            train_recall(pred, data_object['y'].int())
        optimizer.step()

        acc = train_accuracy.compute().item()
        pre = train_precision.compute().item()
        re = train_recall.compute().item()
        print('Accuracy for all answers: ' + str(acc))
        print('Precision for all answers: ' + str(pre))
        print('Recall for all answers: ' + str(re))
        if summary_writer:
            summary_writer.add_scalar('Loss for all answers on the training datasets.', total_train_loss, epoch)
            summary_writer.add_scalar('Precision for all answers on the training datasets.', pre, epoch)
            summary_writer.add_scalar('Recall for all answers on the training datasets.', re, epoch)
        train_accuracy.reset()
        train_precision.reset()
        train_recall.reset()
        # dummy loss
        loss = 10000
        if (epoch != 0) and (epoch % args.val_epochs == 0):
            loss, val_acc, val_pre, val_re, val_auc, val_unobserved_pre, val_unobserved_re, val_unobserved_auc = compute_metrics(
                val_data, model, threshold)
            if trial:
                trial.report(loss, epoch)
            lr_scheduler.step()

            print('Validating!')
            print('Validation loss: ' + str(loss.item()))
            print('Accuracy for all answers: ' + str(val_acc))
            print('Precision for all answers:  ' + str(val_pre))
            print('Recall for all answers: ' + str(val_re))
            print('AUC for all answers: ' + str(val_auc))
            print('Precision for unobserved answers:  ' + str(val_unobserved_pre))
            print('Recall for unobserved answers: ' + str(val_unobserved_re))
            print('AUC for unobserved answers: ' + str(val_unobserved_auc))
            if summary_writer:
                summary_writer.add_scalar('Loss for all answers on the validation datasets.', loss, epoch)
                summary_writer.add_scalar('Precision for all answers on the validation datasets.', val_pre, epoch)
                summary_writer.add_scalar('Recall for all answers on the validation datasets.', val_re, epoch)
                summary_writer.add_scalar('AUC for all answers on the validation datasets.', val_auc, epoch)
                summary_writer.add_scalar('Precision for unobserved answers on the validation datasets.',
                                          val_unobserved_pre, epoch)
                summary_writer.add_scalar('Recall for unobserved answers on the validation datasets.',
                                          val_unobserved_re, epoch)
                summary_writer.add_scalar('AUC for unobserved answers on the validation datasets.', val_unobserved_auc,
                                          epoch)

            if trial and trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    if trial:
        torch.save(model, os.path.join(model_directory, 'trial{}.pt'.format(trial.number)))
    else:
        torch.save(model, os.path.join(model_directory, 'model.pt'))
    # Why does optuna require to a return value?
    return loss


def objective(trial, device, train_data, val_data, log_directory, model_directory, args):
    loss = train(device=device, train_data=train_data, val_data=val_data, log_directory=log_directory,
                 model_directory=model_directory, args=args, summary_writer=None, trial=trial)
    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--query_string', type=str)
    parser.add_argument('--train_data', type=str, nargs='+')
    parser.add_argument('--val_data', type=str, nargs='+')
    parser.add_argument('--test_data', type=str, nargs='+')
    parser.add_argument('--log_dir', type=str, default='runs/')
    parser.add_argument('--aug', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--max_aggr', action='store_true', default=False)
    parser.add_argument('--hyperparam_tune', action='store_true', default=False)
    parser.add_argument('--monotonic', action='store_true', default=False)
    parser.add_argument('--subquery_gen_strategy', type=str, default='not greedy')
    parser.add_argument('--max_num_subquery_vars', type=int, default=6)
    parser.add_argument('--subquery_depth', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--val_epochs', type=int, default=10)
    # Optimize
    parser.add_argument('--base_dim', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--lr', type=float, default=0.00625)
    parser.add_argument('--lr_scheduler_step_size', type=int, default=10)
    parser.add_argument('--negative_slope', type=float, default=0.1)
    parser.add_argument('--positive_sample_weight', type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    now = datetime.now()
    date_time = now.strftime("%d_%m_%Y_%H:%M:%S")
    current_directory = os.getcwd()
    log_directory = os.path.join(current_directory, args.log_dir + date_time)
    model_directory = os.path.join(log_directory, 'models')
    writer = SummaryWriter(log_directory)

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    if args.aug:
        subqueries, subquery_shape = generate_subqueries(args.train_data[0], query_string=args.query_string,
                                                         subquery_gen_strategy=args.subquery_gen_strategy,
                                                         subquery_depth=args.subquery_depth,
                                                         max_num_subquery_vars=args.max_num_subquery_vars)

        train_data = prep_data(data_directories=args.train_data, query_string=args.query_string, aug=args.aug,
                               subqueries=subqueries)
        val_data = prep_data(data_directories=args.val_data, query_string=args.query_string, aug=args.aug,
                             subqueries=subqueries)

        shapes_dict = {k: 1 for k, v in train_data[0]['indices_dict'].items()}
        shapes_dict = {**shapes_dict, **subquery_shape}
    else:
        subqueries = None
        train_data = prep_data(data_directories=args.train_data, query_string=args.query_string, aug=args.aug)
        val_data = prep_data(data_directories=args.val_data, query_string=args.query_string, aug=args.aug)

        shapes_dict = {k: 1 for k, v in train_data[0]['indices_dict'].items()}

    if not args.hyperparam_tune:
        train(device=device, feat_dim=1, shapes_dict=shapes_dict, train_data=train_data, val_data=val_data,
              log_directory=log_directory, model_directory=model_directory, subqueries=subqueries, args=args,
              summary_writer=writer)

        if args.test:
            print('Start testing')
            eval(test_data_directories=args.test_data, model_directory=model_directory, aug=args.aug, device=device,
                 summary_writer=writer)

    else:
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=30, interval_steps=10))
        study.optimize(
            lambda trial: objective(trial=trial, device=device, train_data=train_data, val_data=val_data,
                                    log_directory=log_directory, model_directory=model_directory, args=args),
            n_trials=100)

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
