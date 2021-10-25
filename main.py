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
from data_utils import prep_data
from eval import eval, compute_metrics


# Todo:
#  - Encode unary predicates in the initial feature vectors - different dim for initial feature vector and hidden states
#  - Double check that root of query answer is in first position
#  - Double check that ids for sub-queries are correct
#  - Clean up and comment functions in the data_util.py
#  - Double check behavior if a subquery does not have answers on training data
#  - Use a file to specify the head relations in the data generation procedure
#  - Evaluation of unobserved answers
#  - Add or max as aggregation function?


def train(device, train_data, val_data, log_directory, model_directory, args, summary_writer=None, trial=None):
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

    model = HGNN(len(train_data[0]['x'][0]), base_dim, train_data[0]['num_edge_types_by_shape'], num_layers,
                 negative_slope)
    model.to(device)
    for param in model.parameters():
        print(type(param.data), param.size())

    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model))

    # Adam optimizer already updates the learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=0.5)

    # Needs to be defined as module in the HGNN class to automatically move to GPU
    threshold = 0.5
    train_accuracy = torchmetrics.Accuracy(threshold=threshold)
    train_precision = torchmetrics.Precision(threshold=threshold)
    train_recall = torchmetrics.Recall(threshold=threshold)

    for epoch in range(epochs):
        print('Epoch-{0} lr: {1}'.format(epoch, lr_scheduler.get_last_lr()))
        model.train()
        optimizer.zero_grad()
        total_train_loss = 0
        batch = [train_data[i] for i in torch.randperm(len(train_data))[:args.batch_size]]
        print('Training!')
        for data_object in batch:
            pred = model(data_object['x'], data_object['hyperedge_indices'], data_object['hyperedge_types'],
                         logits=True).flatten()
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
            print('Loss: ' + str(loss))
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
            print('Validation loss :' + str(loss))
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
                summary_writer.add_scalar('Precision for unobserved answers on the validation datasets.', val_unobserved_pre, epoch)
                summary_writer.add_scalar('Recall for unobserved answers on the validation datasets.', val_unobserved_re, epoch)
                summary_writer.add_scalar('AUC for unobserved answers on the validation datasets.', val_unobserved_auc, epoch)

            if trial and trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    if trial:
        torch.save(model.state_dict(), os.path.join(model_directory, 'trial{}.pt'.format(trial.number)))
    else:
        torch.save(model.state_dict(), os.path.join(model_directory, 'model.pt'))
    # Why does optuna require to a return value?
    return loss


def objective(trial, device, train_data, val_data, log_directory, model_directory, args):
    loss = train(device=device, train_data=train_data, val_data=val_data, log_directory=log_directory,
                 model_directory=model_directory, args=args, summary_writer=None, trial=trial)
    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bla bla')
    parser.add_argument('--query_string', type=str,
                        default='SELECT distinct ?v0 WHERE { ?v0  <http://schema.org/caption> ?v1 . ?v0   <http://schema.org/text> ?v2 . ?v0 <http://schema.org/contentRating> ?v3 . ?v0   <http://purl.org/stuff/rev#hasReview> ?v4 .  ?v4 <http://purl.org/stuff/rev#title> ?v5 . ?v4  <http://purl.org/stuff/rev#reviewer> ?v6 . ?v7 <http://schema.org/actor> ?v6 . ?v7 <http://schema.org/language> ?v8  }')
    parser.add_argument('--train_data', type=str, nargs='+',
                        default=['datasets/wsdbm-data-model-v1/dataset1/', 'datasets/wsdbm-data-model-v1/dataset2/',
                                 'datasets/wsdbm-data-model-v1/dataset3/'])
    parser.add_argument('--val_data', type=str, nargs='+', default=['datasets/wsdbm-data-model-v1/dataset4/'])
    parser.add_argument('--test_data', type=str, nargs='+', default=['datasets/wsdbm-data-model-v1/dataset5/','datasets/wsdbm-data-model-v1/dataset6/','datasets/wsdbm-data-model-v1/dataset7/'])
    parser.add_argument('--aug', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--subquery_gen_strategy', type=str, default='not greedy')
    parser.add_argument('--max_num_subquery_vars', type=int, default=6)
    parser.add_argument('--subquery_depth', type=int, default=2)
    parser.add_argument('--pretrained_model', type=str, default='')
    parser.add_argument('--relation2id', type=str, default='')
    parser.add_argument('--base_dim', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--val_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=int, default=0.00625)
    parser.add_argument('--lr_scheduler_step_size', type=int, default=10)
    parser.add_argument('--negative_slope', type=int, default=0.1)
    parser.add_argument('--positive_sample_weight', type=int, default=2)
    parser.add_argument('--log_dir', type=str, default='runs/')
    parser.add_argument('--hyperparam_tune', action='store_true', default=False)
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

    if args.relation2id:
        with open(args.relations2id, 'rb') as f:
            relation2id = pickle.load(f)
        train_data, _ = prep_data(data_directories=args.train_data, query_string=args.query_string, aug=args.aug,
                                  subquery_gen_strategy=args.subquery_gen_strategy, subquery_depth=args.subquery_depth,
                                  max_num_subquery_vars=args.max_num_subquery_vars, relation2id=relation2id)
        val_data, _ = prep_data(data_directories=args.val_data, query_string=args.query_string, aug=args.aug,
                                subquery_gen_strategy=args.subquery_gen_strategy, subquery_depth=args.subquery_depth,
                                max_num_subquery_vars=args.max_num_subquery_vars, relation2id=relation2id)
    else:
        train_data, relation2id = prep_data(data_directories=args.train_data, query_string=args.query_string,
                                            aug=args.aug,
                                            subquery_gen_strategy=args.subquery_gen_strategy,
                                            subquery_depth=args.subquery_depth,
                                            max_num_subquery_vars=args.max_num_subquery_vars)
        val_data, _ = prep_data(data_directories=args.val_data, query_string=args.query_string, aug=args.aug,
                                subquery_gen_strategy=args.subquery_gen_strategy, subquery_depth=args.subquery_depth,
                                max_num_subquery_vars=args.max_num_subquery_vars, relation2id=relation2id)
        with open(os.path.join(model_directory, 'relation2id.pickle'), 'wb') as f:
            pickle.dump(relation2id, f)

    if not args.hyperparam_tune:
        train(device=device, train_data=train_data, val_data=val_data, log_directory=log_directory,
              model_directory=model_directory, args=args, summary_writer=writer)

        if args.test:
            print('Start testing')
            eval(test_data_directories=args.test_data, query_string=args.query_string, model_directory=model_directory,
                 base_dim=args.base_dim, num_layers=args.num_layers, negative_slope=args.negative_slope,
                 aug=args.aug, subquery_gen_strategy=args.subquery_gen_strategy, subquery_depth=args.subquery_depth,
                 max_num_subquery_vars=args.max_num_subquery_vars, device=device, summary_writer=writer)

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
