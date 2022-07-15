import torch
import argparse
import torchmetrics
import os
import optuna
from optuna.trial import TrialState
from optuna.samplers import RandomSampler
from datetime import datetime
from model import HGNN
from data_utils import generate_subqueries, prep_data, create_batch_data_object
from eval import eval, compute_metrics
import json
from load_watdiv import load_watdiv_benchmark
from load_fb15k237 import load_fb15k237_benchmark


def train(device, feat_dim, shapes_dict, train_data, val_data, log_directory, model_directory, args, subqueries=None, trial=None):

    # Samples hyperparameters if a trial object is passed to the train function
    if trial:
        print('Starting trial-{}!'.format(trial.number))
        args.base_dim = trial.suggest_int('base_dim', 8, 64)
        args.learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.1001, step=0.0005) #default 0.001
        args.negative_slope = trial.suggest_float('negative_slope', 0.001, 0.101, step=0.005) #default 0.01
        args.positive_sample_weight = trial.suggest_int('positive_sample_weight', 1, 100)
        with open(os.path.join(log_directory, 'trial-{}-config.txt'.format(trial.number)), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    else:
        with open(os.path.join(log_directory, 'config.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    # Creates a model instance
    model = HGNN(query_string=args.query_string, feat_dim=feat_dim, base_dim=args.base_dim, shapes_dict=shapes_dict,
                 num_layers=args.num_layers, negative_slope=args.negative_slope, subqueries=subqueries)

    model.to(device)
    for name, param in model.named_parameters():
        print(name, type(param.data), param.size())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    threshold = 0.5
    train_precision = torchmetrics.Precision(threshold=threshold).to(device)
    train_recall = torchmetrics.Recall(threshold=threshold).to(device)

    # Main training loop
    val_ap = 0
    for epoch in range(1, args.epochs + 1):
        print('Epoch-{0}!'.format(epoch))
        model.train()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        total_train_loss = 0
        # Creates batch with specified batch size
        batch = [train_data[i] for i in torch.randperm(len(train_data))[:args.batch_size]]
        if args.batch_size > 10:
            batch = [create_batch_data_object(batch)]
        print('Training!')
        # Loops through data objects in a batch
        for data_object in batch:
            pred = model(data_object['feat'], data_object['indices_dict'], logits=True, device=device).flatten()
            pred = pred[data_object['nodes']]
            y = data_object['labels'].to(device)
            sample_weights_train = args.positive_sample_weight * y + (torch.ones(len(y), device=device) - y)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, y, weight=sample_weights_train)
            loss.backward()
            total_train_loss = total_train_loss + loss
            pred = torch.sigmoid(pred)
            train_precision(pred, y.int())
            train_recall(pred, y.int())
        # Updates parameters for every batch
        optimizer.step()
        print('Loss: ' + str(total_train_loss.item()))

        pre = train_precision.compute().item()
        re = train_recall.compute().item()
        print('Precision for all samples: ' + str(pre))
        print('Recall for all samples: ' + str(re))
        train_precision.reset()
        train_recall.reset()

        if (epoch != 0) and (epoch % args.val_epochs == 0):
            with torch.no_grad():
                model.eval()
                loss, val_pre, val_re, val_ap, val_unobserved_pre, val_unobserved_re, val_unobserved_ap = compute_metrics(
                    val_data, model, device, threshold)

                print('Validating!')
                print('Validation loss: ' + str(loss.item()))
                print('Precision for all samples:  ' + str(val_pre))
                print('Recall for all samples: ' + str(val_re))
                print('AP for all samples: ' + str(val_ap))
                print('Precision for unmasked samples:  ' + str(val_unobserved_pre))
                print('Recall for unmasked samples: ' + str(val_unobserved_re))
                print('AP for unmasked samples: ' + str(val_unobserved_ap))

                if trial:
                    trial.report(val_ap, epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

    if trial:
        torch.save(model, os.path.join(model_directory, 'trial-{}-model.pt'.format(trial.number)))
    else:
        torch.save(model, os.path.join(model_directory, 'model.pt'))
    return val_ap


if __name__ == '__main__':
    torch.cuda.empty_cache()
    torch.manual_seed(0)

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--query_string', type=str)
    parser.add_argument('--train_data', type=str, nargs='+')
    parser.add_argument('--val_data', type=str, nargs='+')
    parser.add_argument('--test_data', type=str, nargs='+')
    parser.add_argument('--log_dir', type=str, default='runs/')
    parser.add_argument('--aug', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--max_num_subquery_vars', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--val_epochs', type=int, default=25)
    parser.add_argument('--base_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--negative_slope', type=float, default=0.1)
    parser.add_argument('--positive_sample_weight', type=int, default=1)
    parser.add_argument('--tune_param', action='store_true', default=False)
    parser.add_argument('--gpu', action='store_true', default=False)
    args = parser.parse_args()

    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    now = datetime.now()
    date_time = now.strftime("%d_%m_%Y_%H:%M:%S")
    current_directory = os.getcwd()
    log_directory = os.path.join(current_directory, args.log_dir + date_time)
    model_directory = os.path.join(log_directory, 'models')

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    # Loads data
    if 'fb15k237' in args.train_data[0]:
        train_samples, train_nodes, types, train_labels, train_masks, graphs = load_fb15k237_benchmark(args.train_data[0])
        val_samples, val_nodes, types, val_labels, val_masks, graphs = load_fb15k237_benchmark(args.val_data[0])
    else:
        train_samples, train_nodes, types, train_labels, train_masks, graphs = load_watdiv_benchmark(args.train_data,
                                                                                              args.query_string)
        val_samples, val_nodes, types, val_labels, val_masks, graphs = load_watdiv_benchmark(args.val_data, args.query_string)

    # Preprocesses the data
    if args.aug:
        subqueries, subquery_shape = generate_subqueries(query_string=args.query_string, max_num_subquery_vars=args.max_num_subquery_vars)

        print('Training samples!')
        train_data_objects = prep_data(labels=train_labels, sample_graphs=train_samples, nodes=train_nodes, masks=train_masks, device=device, aug=args.aug,
                                       subqueries=subqueries, types=types)
        print('Validation samples!')
        val_data_objects = prep_data(labels=val_labels, sample_graphs=val_samples, nodes=val_nodes, masks=val_masks, device=device, aug=args.aug,
                                     subqueries=subqueries, types=types)

        rels = set()
        for d in train_data_objects + val_data_objects:
            for k in d['indices_dict'].keys():
                rels.add(k)
        shapes_dict = {k: 1 for k in rels}
        shapes_dict = {**shapes_dict, **subquery_shape}
    else:
        subqueries = None
        print('Training samples!')
        train_data_objects = prep_data(labels=train_labels, sample_graphs=train_samples, nodes=train_nodes, masks=train_masks, device=device, aug=args.aug,
                                       subqueries=subqueries, types=types)
        print('Validation samples!')
        val_data_objects = prep_data(labels=val_labels, sample_graphs=val_samples, nodes=val_nodes, masks=val_masks, device=device, aug=args.aug,
                                     subqueries=subqueries, types=types)

        rels = set()
        for d in train_data_objects + val_data_objects:
            for k in d['indices_dict'].keys():
                rels.add(k)
        shapes_dict = {k: 1 for k in rels}

    feat_dim = len(train_data_objects[0]['feat'][0])

    # Starts the training loop
    if not args.tune_param:
        train(device=device, feat_dim=feat_dim, shapes_dict=shapes_dict, train_data=train_data_objects,
            val_data=val_data_objects, log_directory=log_directory, model_directory=model_directory, subqueries=subqueries, args=args)

        if args.test:
            print('Start Testing!')
            eval(test_data=args.test_data, model_directory=os.path.join(model_directory, 'model.pt'), aug=args.aug, device=device)

    else:
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=30, interval_steps=10), sampler=RandomSampler(0))
        study.optimize(lambda trial: train(trial=trial, device=device, feat_dim=feat_dim, shapes_dict=shapes_dict,
                                    train_data=train_data_objects, val_data=val_data_objects,
                                    log_directory=log_directory, model_directory=model_directory, subqueries=subqueries, args=args), n_trials=100, gc_after_trial=True)

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
            with torch.no_grad():
                print('Start Testing!')
                eval(test_data=args.test_data, model_directory=os.path.join(model_directory, 'trial-{}-model.pt'.format(trial.number)), aug=args.aug, device=device)


