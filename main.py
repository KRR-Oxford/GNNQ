import torch
import argparse
import torchmetrics
import os
import optuna
from optuna.trial import TrialState
from datetime import datetime
from model import HGNN
from data_utils import generate_subqueries, prep_data
from eval import eval, compute_metrics
import json
from load_watdiv import load_watdiv_benchmark
from load_fb15k237 import load_fb15k237_benchmark


# Function containing the main training loop
def train(device, feat_dim, shapes_dict, train_data, val_data, log_directory, model_directory, args, subqueries=None,
          summary_writer=None, trial=None):

    if trial:
        args.base_dim = trial.suggest_int('base_dim', 8, 64)
        args.learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1, step=0.005)
        args.positive_sample_weight = trial.suggest_int('positive_sample_weight', 1, 100)
        args.negative_slope = trial.suggest_float('negative_slope', 0.0, 0.5, step=0.1)
        with open(os.path.join(log_directory, 'trial-{}-config.txt'.format(trial.number)), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    else:
        with open(os.path.join(log_directory, 'config.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    # This is the definition of the model
    model = HGNN(query_string=args.query_string, feat_dim=feat_dim, base_dim=args.base_dim, shapes_dict=shapes_dict,
                 num_layers=args.num_layers, negative_slope=args.negative_slope, subqueries=subqueries)

    model.to(device)
    for name, param in model.named_parameters():
        print(name, type(param.data), param.size())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Needs to be defined as module in the HGNN class to automatically move to GPU
    threshold = 0.5
    train_precision = torchmetrics.Precision(threshold=threshold)
    train_recall = torchmetrics.Recall(threshold=threshold)

    # This is the main training loop
    val_ap = 0
    for epoch in range(args.epochs):
        print('Epoch-{0}!'.format(epoch))
        model.train()
        optimizer.zero_grad()
        total_train_loss = 0
        # Creates batch with specified batch size
        batch = [train_data[i] for i in torch.randperm(len(train_data))[:args.batch_size]]
        print('Training!')

        # Loop through data objects in a batch
        for data_object in batch:
            pred = model(data_object['feat'], data_object['indices_dict'], logits=True).flatten()
            pred = pred[data_object['nodes']]
            y = data_object['labels']
            sample_weights_train = args.positive_sample_weight * y + (torch.ones(len(y)) - y)
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
        if summary_writer:
            summary_writer.add_scalar('Loss for all training samples.', total_train_loss, epoch)
            summary_writer.add_scalar('Precision for all training samples.', pre, epoch)
            summary_writer.add_scalar('Recall for all training samples.', re, epoch)
        train_precision.reset()
        train_recall.reset()
        if (epoch != 0) and (epoch % args.val_epochs == 0):
            # This is called twice
            model.eval()
            loss, val_pre, val_re, val_ap, val_unobserved_pre, val_unobserved_re, val_unobserved_ap = compute_metrics(
                val_data, model, threshold)

            if trial:
                trial.report(val_ap, epoch)

            print('Validating!')
            print('Validation loss: ' + str(loss.item()))
            print('Precision for all samples:  ' + str(val_pre))
            print('Recall for all samples: ' + str(val_re))
            print('AP for all samples: ' + str(val_ap))
            print('Precision for unmasked samples:  ' + str(val_unobserved_pre))
            print('Recall for unmasked samples: ' + str(val_unobserved_re))
            print('AP for unmasked samples: ' + str(val_unobserved_ap))
            if summary_writer:
                summary_writer.add_scalar('Loss for all validations samples.', loss, epoch)
                summary_writer.add_scalar('Precision for all validations samples.', val_pre, epoch)
                summary_writer.add_scalar('Recall for all validations samples.', val_re, epoch)
                summary_writer.add_scalar('AP for all all validations samples.', val_ap, epoch)
                summary_writer.add_scalar('Precision for unmasked validation samples.',
                                          val_unobserved_pre, epoch)
                summary_writer.add_scalar('Recall for unmasked validation samples.',
                                          val_unobserved_re, epoch)
                summary_writer.add_scalar('AP for unmasked validation samples.', val_unobserved_ap,
                                          epoch)
            if trial:
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

    if trial:
        torch.save(model, os.path.join(model_directory, 'trial-{}-model.pt'.format(trial.number)))
    else:
        torch.save(model, os.path.join(model_directory, 'model.pt'))
    return val_ap


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--query_string', type=str)
    parser.add_argument('--train_data', type=str, nargs='+')
    parser.add_argument('--val_data', type=str, nargs='+')
    parser.add_argument('--test_data', type=str, nargs='+')
    parser.add_argument('--log_dir', type=str, default='runs/')
    parser.add_argument('--aug', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--max_num_subquery_vars', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--val_epochs', type=int, default=10)
    parser.add_argument('--base_dim', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--negative_slope', type=float, default=0.1)
    parser.add_argument('--positive_sample_weight', type=int, default=1)
    parser.add_argument('--tune_param', action='store_true', default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    now = datetime.now()
    date_time = now.strftime("%d_%m_%Y_%H:%M:%S")
    current_directory = os.getcwd()
    log_directory = os.path.join(current_directory, args.log_dir + date_time)
    model_directory = os.path.join(log_directory, 'models')

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    # Load data
    if 'fb15k237' in args.train_data[0]:
        train_samples, train_nodes, types, train_labels, train_masks, graphs = load_fb15k237_benchmark(args.train_data[0])
        val_samples, val_nodes, types, val_labels, val_masks, graphs = load_fb15k237_benchmark(args.val_data[0])
    else:
        train_samples, train_nodes, types, train_labels, train_masks, graphs = load_watdiv_benchmark(args.train_data,
                                                                                              args.query_string)
        val_samples, val_nodes, types, val_labels, val_masks, graphs = load_watdiv_benchmark(args.val_data, args.query_string)

    # Create data objects
    if args.aug:
        subqueries, subquery_shape = generate_subqueries(query_string=args.query_string,
                                                         subquery_gen_strategy=args.subquery_gen_strategy,
                                                         subquery_depth=args.subquery_depth,
                                                         max_num_subquery_vars=args.max_num_subquery_vars)

        print('Training samples!')
        train_data_objects = prep_data(train_labels, train_samples, train_nodes, train_masks, aug=args.aug,
                                       subqueries=subqueries, types=types)
        print('Validation samples!')
        val_data_objects = prep_data(val_labels, val_samples, val_nodes, val_masks, aug=args.aug,
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
        train_data_objects = prep_data(train_labels, train_samples, train_nodes, train_masks, aug=args.aug,
                                       subqueries=subqueries, types=types)
        print('Validation samples!')
        val_data_objects = prep_data(val_labels, val_samples, val_nodes, val_masks, aug=args.aug,
                                     subqueries=subqueries, types=types)

        rels = set()
        for d in train_data_objects + val_data_objects:
            for k in d['indices_dict'].keys():
                rels.add(k)
        shapes_dict = {k: 1 for k in rels}

    if types:
        feat_dim = len(types) + 1
    else:
        feat_dim = 1

    # Function the contains the main train loop
    if not args.tune_param:
        train(device=device, feat_dim=feat_dim, shapes_dict=shapes_dict, train_data=train_data_objects,
            val_data=val_data_objects, log_directory=log_directory, model_directory=model_directory, subqueries=subqueries, args=args)

        if args.test:
            print('Start Testing!')
            eval(test_data=args.test_data, model_directory=os.path.join(model_directory, 'model.pt'), aug=args.aug, device=device)

    else:
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=30, interval_steps=10))
        study.optimize(lambda trial: train(trial=trial, device=device, feat_dim=feat_dim, shapes_dict=shapes_dict,
                                    train_data=train_data_objects, val_data=val_data_objects,
                                    log_directory=log_directory, model_directory=model_directory, args=args), n_trials=100)

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
            print('Start Testing!')
            eval(test_data=args.test_data, model_directory=os.path.join(model_directory, 'trial-{}-model.pt'.format(trial.number)), aug=args.aug, device=device)


