import torch
from model import HGNN
from utils import load_triples, load_answers, create_triples_with_ids, create_y_vector, create_index_matrices, add_tuples_to_index_matrices
import numpy as np
import argparse
from torchmetrics import Accuracy, Precision, Recall
import optuna
from optuna.trial import TrialState

parser = argparse.ArgumentParser(description='Bla bla')
parser.add_argument('--train_data', type=str, default='train')
parser.add_argument('--val_data', type=str, default='val')
parser.add_argument('--base_dim', type=int, default=16)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--val_epochs', type=int, default=10)
parser.add_argument('--lr', type=int, default=0.1)
args = parser.parse_args()

# base_dim = args.base_dim
# num_layers = args.num_layers
# epochs = args.epochs
# learning_rate = args.lr
# val_epochs = args.val_epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

subquery_answers_files = []
val_subquery_answers_files = []

triples = load_triples(args.train_data + '/graph.ttl')
triples, entity2id, relation2id, _, _ = create_triples_with_ids(triples)
num_nodes = len(entity2id)
answers = load_answers(args.train_data + '/simple_answers.pickle')
answers = [entity2id[entity[0]] for entity in answers]
y_train = create_y_vector(answers, num_nodes)
hyperedge_index_train, hyperedge_type_train, num_edge_types_by_shape_train = create_index_matrices(triples)
for file in subquery_answers_files:
    subquery_answers = load_answers(args.train_data + file)
    subquery_answers = [[entity2id[entity] for entity in answer] for answer in subquery_answers]
    hyperedge_index_train, hyperedge_type_train, num_edge_types_by_shape_train = add_tuples_to_index_matrices(
        subquery_answers, hyperedge_index_train, hyperedge_type_train, num_edge_types_by_shape_train)


triples_val = load_triples(args.val_data + '/graph.ttl')
triples_val, entity2id_val, _, _, _ = create_triples_with_ids(triples_val, relation2id)
num_nodes_val = len(entity2id_val)
val_answers = load_answers(args.val_data + '/val_simple_answers.pickle')
val_answers = [entity2id_val[entity[0]] for entity in val_answers]
y_val = create_y_vector(val_answers, num_nodes_val)
hyperedge_index_val, hyperedge_type_val, num_edge_types_by_shape_val = create_index_matrices(triples_val)
for file in val_subquery_answers_files:
    subquery_answers_val = load_answers(args.val_data + file)
    subquery_answers_val = [[entity2id_val[entity] for entity in answer] for answer in subquery_answers_val]
    hyperedge_index_val, hyperedge_type_val, num_edge_types_by_shape_val = add_tuples_to_index_matrices(subquery_answers_val, hyperedge_index_val, hyperedge_type_val, num_edge_types_by_shape_val)


def objective(trial):
    # base_dim = trial.suggest_int('base_dim', 8, 32)
    base_dim = 16
    # learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    learning_rate = 0.1
    # ratio_negative_samples = trial.suggest_int('ratio_negative_samples', 1, 5)
    ratio_negative_samples = 1
    num_layers = args.num_layers
    epochs = args.epochs
    val_epochs = args.val_epochs


    model = HGNN(base_dim, num_edge_types_by_shape_train, num_layers)
    model.to(device)
    for param in model.parameters():
        print(type(param.data), param.size())

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCELoss()

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

        pred = model(x_train, hyperedge_index_train, hyperedge_type_train).flatten()
        pred_sampled = torch.cat((pred[answers], pred[torch.randint(x_train.size()[0], size=(ratio_negative_samples * len(answers),))]), dim=0)
        loss = loss_fn(pred_sampled, torch.cat((torch.ones(len(answers)), torch.zeros(ratio_negative_samples * len(answers))), dim = 0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(epoch)

        model.eval()
        acc = accuracy(pred, y_train)
        pre = precision(pred, y_train)
        rec = recall(pred, y_train)
        print('Train')
        print(acc)
        print(pre)
        print(rec)

        if epoch % val_epochs == 0:
            model.eval()
            pred = model(x_val, hyperedge_index_val, hyperedge_type_val).flatten()
            acc = accuracy(pred, y_val)
            pre = precision(pred, y_val)
            rec = recall(pred, y_val)
            print('Val')
            print(acc)
            print(pre)
            print(rec)
            trial.report(pre, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    torch.save(model.state_dict(), './models/' +'trial{}.pt'.format(trial.number))
    return pre



study = optuna.create_study(direction='maximize')
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