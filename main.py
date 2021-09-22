import torch
from model import HGNN
from data_utils import create_data_object
import numpy as np
import argparse
import torchmetrics
import optuna
from optuna.trial import TrialState
import pickle
from torch.utils.tensorboard import SummaryWriter

# ToDo: Trained model and logs should be saved in a directory with data and most important hyperparameters

parser = argparse.ArgumentParser(description='Bla bla')
parser.add_argument('--train_data', type=str, nargs='+', default=['wsdbm-data-model-2/dataset1/'])
parser.add_argument('--val_data', type=str, nargs='+', default=['wsdbm-data-model-2/dataset2/'])
parser.add_argument('--query_string', type=str, default='SELECT distinct ?v0 WHERE { ?v0  <http://schema.org/caption> ?v1 . ?v0   <http://schema.org/text> ?v2 . ?v0 <http://schema.org/contentRating> ?v3 . ?v0   <http://purl.org/stuff/rev#hasReview> ?v4 .  ?v4 <http://purl.org/stuff/rev#title> ?v5 . ?v4  <http://purl.org/stuff/rev#reviewer> ?v6 . ?v7 <http://schema.org/actor> ?v6 . ?v7 <http://schema.org/language> ?v8  }')
parser.add_argument('--aug', type=bool, default=True)
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--encoding', type=str, default='')
parser.add_argument('--base_dim', type=int, default=16)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument('--val_epochs', type=int, default=10)
parser.add_argument('--lr', type=int, default=0.00625)
parser.add_argument('--lr_scheduler_step_size', type=int, default=10)
parser.add_argument('--negative_slope', type=int, default=0.1)
parser.add_argument('--positive_sample_weight', type=int, default=1)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):
    writer = SummaryWriter()

    train_data_directories = args.train_data
    val_data_directories = args.val_data
    query_string = args.query_string
    base_dim = args.base_dim
    num_layers = args.num_layers
    epochs = args.epochs
    val_epochs = args.val_epochs
    learning_rate = args.lr
    lr_scheduler_step_size = args.lr_scheduler_step_size
    negative_slope = args.negative_slope
    positive_sample_weight = args.positive_sample_weight
    aug = args.aug

    # base_dim = trial.suggest_int('base_dim', 8, 32)
    # num_layers = trial.suggest_int('num_layers', 1, 4)
    # learning_rate = trial.suggest_float("lr", 0.001, 0.1, step=0.001)
    # lr_scheduler_step_size = trial.suggest_int('lr_scheduler_step_size', 1, 10)
    # positive_sample_weight = trial.suggest_int('positive_sample_weight', 1, 20)
    # negative_slope = trial.suggest_float('negative_slope',0.01, 0.2, step=0.01)

    train_data = []
    val_data = []

    if args.encoding:
        with open(args.encoding, 'rb') as f:
            relation2id = pickle.load(f)
    else:
        relation2id = None
    for directory in train_data_directories:
        data_object, relation2id = create_data_object(directory + 'graph.nt', directory + 'corrupted_graph.nt' , query_string, base_dim, aug, 2, relation2id)
        train_data.append(data_object)

    for directory in val_data_directories:
        data_object, relation2id = create_data_object(directory + 'graph.nt', directory + 'corrupted_graph.nt', query_string, base_dim, aug, 2, relation2id)
        val_data.append(data_object)


    model = HGNN(base_dim, train_data[0]['num_edge_types_by_shape'], num_layers)
    model.to(device)
    for param in model.parameters():
        print(type(param.data), param.size())

    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model))

    # Adam optimizer already updates the learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=0.5)

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
            pred = model(data_object['x'], data_object['hyperedge_indices'], data_object['hyperedge_types'], logits=True, negative_slope=negative_slope).flatten()
            # Weigh false positive samples from the previous epoch higher to address bad recall
            # We could also just use a small fraction of negative samples
            sample_weights_train = positive_sample_weight * data_object['y'] + torch.ones(len(data_object['y']))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, data_object['y'],weight=sample_weights_train)
            loss.backward()
            total_train_loss = total_train_loss + loss
            print('Train loss')
            print(loss)
            pred = torch.sigmoid(pred)
            train_accuracy(pred,data_object['y'].int())
            train_precision(pred, data_object['y'].int())
            train_recall(pred, data_object['y'].int())
        optimizer.step()

        model.eval()

        #ToDo: Add auc score as metric
        acc = train_accuracy.compute().item()
        pre = train_precision.compute().item()
        re = train_recall.compute().item()
        print('Train')
        print('Accuracy ' + str(acc))
        print('Precision ' + str(pre))
        print('Recall ' + str(re))
        writer.add_scalar('Loss training', total_train_loss, epoch)
        writer.add_scalar('Precision training',pre, epoch)
        writer.add_scalar('Recall training', re, epoch)
        train_accuracy.reset()
        train_precision.reset()
        train_recall.reset()

        if (epoch % val_epochs == 0) & (epoch != 0):
            total_loss = 0
            for data_object in val_data:
                pred = model(data_object['x'], data_object['hyperedge_indices'], data_object['hyperedge_types'], negative_slope=negative_slope).flatten()
                # Weigh false positive samples from the previous epoch higher to address bad recall
                sample_weights_val = positive_sample_weight * data_object['y'] + torch.ones(len(data_object['y']))
                total_loss = total_loss + torch.nn.functional.binary_cross_entropy(pred, data_object['y'],weight=sample_weights_val)
                val_accuracy(pred, data_object['y'].int())
                val_precision(pred, data_object['y'].int())
                val_recall(pred, data_object['y'].int())

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

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    torch.save(model.state_dict(), './models/' +'trial{}.pt'.format(trial.number))
    with open('./models/' + 'relation2id.pickle', 'wb') as f:
        pickle.dump(relation2id, f)
    # Report best metric -- can this be different from the metric used for trial report
    return loss



study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=30, interval_steps=1))
study.optimize(objective, n_trials=1)

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