import torch
from model import HGNN
from utils import load_triples, load_answers, create_triples_with_ids, create_y_vector, create_index_matrices, add_tuples_to_index_matrices
import numpy as np
import argparse
from torchmetrics import Accuracy, Precision, Recall
import optuna
from optuna.trial import TrialState

parser = argparse.ArgumentParser(description='Bla bla')
parser.add_argument('--train_data', type=str, default='train_large_2')
parser.add_argument('--val_data', type=str, default='val')
parser.add_argument('--base_dim', type=int, default=16)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--val_epochs', type=int, default=10)
parser.add_argument('--lr', type=int, default=0.1)
parser.add_argument('--lr_scheduler_step_size', type=int, default=2)
parser.add_argument('--negative_slope', type=int, default=0.1)
parser.add_argument('--positive_sample_weight', type=int, default=1)
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

subquery_answers_files = ['/subquery_answers.pickle','/subquery_answers2.pickle']
val_subquery_answers_files = ['/subquery_answers.pickle','/subquery_answers2.pickle']

triples = load_triples(args.train_data + '/graph.ttl')
triples, entity2id, relation2id, _, _ = create_triples_with_ids(triples)
num_nodes = len(entity2id)
answers = load_answers(args.train_data + '/answers.pickle')
answers = [entity2id[entity[0]] for entity in answers]
y_train = create_y_vector(answers, num_nodes)
y_train_int = y_train.int()
hyperedge_index_train, hyperedge_type_train, num_edge_types_by_shape_train = create_index_matrices(triples)
for file in subquery_answers_files:
    subquery_answers = load_answers(args.train_data + file)
    subquery_answers = [[entity2id[entity] for entity in answer] for answer in subquery_answers]
    hyperedge_index_train, hyperedge_type_train, num_edge_types_by_shape_train = add_tuples_to_index_matrices(
        subquery_answers, hyperedge_index_train, hyperedge_type_train, num_edge_types_by_shape_train)


triples_val = load_triples(args.val_data + '/graph.ttl')
triples_val, entity2id_val, _, _, _ = create_triples_with_ids(triples_val, relation2id)
num_nodes_val = len(entity2id_val)
val_answers = load_answers(args.val_data + '/answers.pickle')
val_answers = [entity2id_val[entity[0]] for entity in val_answers]
y_val = create_y_vector(val_answers, num_nodes_val)
y_val_int = y_val.int()
hyperedge_index_val, hyperedge_type_val, num_edge_types_by_shape_val = create_index_matrices(triples_val)
for file in val_subquery_answers_files:
    subquery_answers_val = load_answers(args.val_data + file)
    subquery_answers_val = [[entity2id_val[entity] for entity in answer] for answer in subquery_answers_val]
    hyperedge_index_val, hyperedge_type_val, num_edge_types_by_shape_val = add_tuples_to_index_matrices(subquery_answers_val, hyperedge_index_val, hyperedge_type_val, num_edge_types_by_shape_val)


def objective(trial):
    base_dim = args.base_dim
    num_layers = args.num_layers
    epochs = args.epochs
    val_epochs = args.val_epochs
    learning_rate = args.lr
    lr_scheduler_step_size = args.lr_scheduler_step_size
    negative_slope = args.negative_slope
    positive_sample_weight = args.positive_sample_weight

    base_dim = trial.suggest_int('base_dim', 8, 32)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    learning_rate = trial.suggest_float("lr", 0.001, 0.1, step=0.001)
    lr_scheduler_step_size = trial.suggest_int('lr_scheduler_step_size', 1, 10)
    positive_sample_weight = trial.suggest_int('positive_sample_weight', 1, 20)
    negative_slope = trial.suggest_float('negative_slope',0.01, 0.2, step=0.01)

    sample_weights_train = positive_sample_weight * y_train + torch.ones(len(y_train))
    sample_weights_val = positive_sample_weight * y_val + torch.ones(len(y_val))


    model = HGNN(base_dim, num_edge_types_by_shape_train, num_layers)
    model.to(device)
    for param in model.parameters():
        print(type(param.data), param.size())

    # Adam optimizer already updates the learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=0.5)
    loss_fn = torch.nn.BCEWithLogitsLoss(weight=sample_weights_train)
    loss_fn_val = torch.nn.BCELoss(weight=sample_weights_val)

    accuracy = Accuracy(threshold=0.5)
    precision = Precision(threshold=0.5)
    recall = Recall(threshold=0.5)

    acc = 0
    pre = 0
    rec = 0

    for epoch in range(epochs):
        model.train()


        x_train = torch.cat((torch.ones(num_nodes, 1), torch.zeros(num_nodes, base_dim - 1)), dim=1)
        x_val = torch.cat((torch.ones(num_nodes_val, 1), torch.zeros(num_nodes_val, base_dim - 1)), dim=1)

        pred = model(x_train, hyperedge_index_train, hyperedge_type_train,logits=True, negative_slope=negative_slope).flatten()
        # Weigh false positive samples from the previous epoch higher to address bad recall
        loss = loss_fn(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch-{0} lr: {1}'.format(epoch, lr_scheduler.get_last_lr()))

        model.eval()
        pred = torch.sigmoid(pred)
        acc = accuracy(pred, y_train_int)
        pre = precision(pred, y_train_int)
        rec = recall(pred, y_train_int)
        print('Train')
        print(loss)
        print(acc)
        print(pre)
        print(rec)

        if (epoch % val_epochs == 0) & (epoch != 0):
            pred = model(x_val, hyperedge_index_val, hyperedge_type_val).flatten()
            acc = accuracy(pred, y_val_int)
            pre = precision(pred, y_val_int)
            rec = recall(pred, y_val_int)
            loss = loss_fn_val(pred, y_val)
            # lr_scheduler.step(loss)
            lr_scheduler.step()

            # print(lr_scheduler._last_lr)
            print('Val')
            print(loss)
            print(acc)
            print(pre)
            print(rec)
            trial.report(loss, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    torch.save(model.state_dict(), './models/' +'trial{}.pt'.format(trial.number))
    return pre



study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=30, interval_steps=1))
study.optimize(objective, n_trials=100)

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