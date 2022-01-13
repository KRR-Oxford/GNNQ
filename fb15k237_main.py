import torch
import argparse
import torchmetrics
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from model import HGNN
from fb15k237_data_utils import generate_subqueries, prep_data
from fb15k237_eval import compute_metrics
import json
import pickle

# Todo:
#  - Encode unary predicates in the initial feature vectors - different dim for initial feature vector and hidden states
#  - Use a file to specify the head relations in the data generation procedure

def train(device, feat_dim, shapes_dict, train_data, val_data, log_directory, model_directory, args, subqueries=None,
          summary_writer=None, trial=None):
    base_dim = args.base_dim
    num_layers = args.num_layers
    epochs = args.epochs
    learning_rate = args.lr
    lr_scheduler_step_size = args.lr_scheduler_step_size
    negative_slope = args.negative_slope
    hyperedge_dropout_prob = args.hyperedge_dropout_prob

    with open(os.path.join(log_directory, 'config.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    model = HGNN(query_string=args.query_string, feat_dim=feat_dim, base_dim=base_dim, shapes_dict=shapes_dict,
                 num_layers=num_layers,
                 negative_slope=negative_slope, hyperedge_dropout_prob=hyperedge_dropout_prob, max_aggr=args.max_aggr,
                 monotonic=args.monotonic, subqueries=subqueries)

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
            pred = pred[data_object['answers']]
            y = data_object['labels']
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred,y)
            loss.backward()
            total_train_loss = total_train_loss + loss
            # print('Loss: ' + str(loss.item()))
            pred = torch.sigmoid(pred)
            train_accuracy(pred, data_object['labels'].int())
            train_precision(pred, data_object['labels'].int())
            train_recall(pred, data_object['labels'].int())
        optimizer.step()
        print('Loss: ' + str(total_train_loss.item()))

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
        if (epoch != 0) and (epoch % args.val_epochs == 0):
            loss, val_acc, val_pre, val_re, val_auc = compute_metrics(
                val_data, model, threshold)
            lr_scheduler.step()

            print('Validating!')
            print('Validation loss: ' + str(loss.item()))
            print('Accuracy for all answers: ' + str(val_acc))
            print('Precision for all answers:  ' + str(val_pre))
            print('Recall for all answers: ' + str(val_re))
            print('AUC for all answers: ' + str(val_auc))

            if summary_writer:
                summary_writer.add_scalar('Loss for all answers on the validation datasets.', loss, epoch)
                summary_writer.add_scalar('Precision for all answers on the validation datasets.', val_pre, epoch)
                summary_writer.add_scalar('Recall for all answers on the validation datasets.', val_re, epoch)
                summary_writer.add_scalar('AUC for all answers on the validation datasets.', val_auc, epoch)

    torch.save(model, os.path.join(model_directory, 'model.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--query_string', type=str)
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--val_data', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--log_dir', type=str, default='runs/')
    parser.add_argument('--aug', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--max_aggr', action='store_true', default=False)
    parser.add_argument('--hyperparam_tune', action='store_true', default=False)
    parser.add_argument('--monotonic', action='store_true', default=False)
    parser.add_argument('--subquery_gen_strategy', type=str, default='not greedy')
    parser.add_argument('--max_num_subquery_vars', type=int, default=6)
    parser.add_argument('--subquery_depth', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--val_epochs', type=int, default=10)
    # Optimize
    parser.add_argument('--base_dim', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--lr', type=float, default=0.00625)
    parser.add_argument('--lr_scheduler_step_size', type=int, default=10)
    parser.add_argument('--negative_slope', type=float, default=0.1)
    parser.add_argument('--hyperedge_dropout_prob', type=float, default=0.1)
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

    infile = open(args.train_data, 'rb')
    train_pos_samples, train_pos_answers, train_neg_samples, train_neg_answers = pickle.load(infile)
    infile.close()

    infile = open(args.val_data, 'rb')
    val_pos_samples, val_pos_answers, val_neg_samples, val_neg_answers = pickle.load(infile)
    infile.close()

    infile = open(args.test_data, 'rb')
    test_pos_samples, test_pos_answers, test_neg_samples, test_neg_answers = pickle.load(infile)
    infile.close()

    for g in train_pos_samples:
        print(len(g))

    if args.aug:
        subqueries, subquery_shape = generate_subqueries(query_string=args.query_string,
                                                         subquery_gen_strategy=args.subquery_gen_strategy,
                                                         subquery_depth=args.subquery_depth,
                                                         max_num_subquery_vars=args.max_num_subquery_vars)

        print('Positive training samples!')
        train_pos_data_objects = prep_data(1,train_pos_samples, train_pos_answers, aug=args.aug,
                               subqueries=subqueries)
        print('Negative training samples!')
        train_neg_data_objects = prep_data(0,train_neg_samples, train_neg_answers, aug=args.aug,
                             subqueries=subqueries)
        print('Positive validation samples!')
        val_pos_data_objects = prep_data(1, val_pos_samples, val_pos_answers, aug=args.aug,
                                          subqueries=subqueries)
        print('Negative validation samples!')
        val_neg_data_objects = prep_data(0, val_neg_samples, val_neg_answers, aug=args.aug,
                                          subqueries=subqueries)
        print('Positive testing samples!')
        test_pos_data_objects = prep_data(1,test_pos_samples, test_pos_answers, aug=args.aug,
                                           subqueries=subqueries)
        print('Negative testing samples!')
        test_neg_data_objects = prep_data(0,test_neg_samples, test_neg_answers, aug=args.aug,
                                           subqueries=subqueries)

        rels = set()
        for d in train_pos_data_objects + train_neg_data_objects + val_pos_data_objects + val_neg_data_objects + test_pos_data_objects + test_neg_data_objects:
            for k in d['indices_dict'].keys():
                rels.add(k)
        shapes_dict = {k: 1 for k in rels}
        shapes_dict = {**shapes_dict, **subquery_shape}
    else:
        subqueries = None
        print('Positive training samples!')
        train_pos_data_objects = prep_data(1, train_pos_samples, train_pos_answers, aug=args.aug,
                                           subqueries=subqueries)
        print('Negative training samples!')
        train_neg_data_objects = prep_data(0, train_neg_samples, train_neg_answers, aug=args.aug,
                                           subqueries=subqueries)
        print('Positive validation samples!')
        val_pos_data_objects = prep_data(1, val_pos_samples, val_pos_answers, aug=args.aug,
                                         subqueries=subqueries)
        print('Negative validation samples!')
        val_neg_data_objects = prep_data(0, val_neg_samples, val_neg_answers, aug=args.aug,
                                         subqueries=subqueries)
        print('Positive testing samples!')
        test_pos_data_objects = prep_data(1, test_pos_samples, test_pos_answers, aug=args.aug,
                                          subqueries=subqueries)
        print('Negative testing samples!')
        test_neg_data_objects = prep_data(0, test_neg_samples, test_neg_answers, aug=args.aug,
                                          subqueries=subqueries)

        rels = set()
        for d in train_pos_data_objects + train_neg_data_objects  + val_pos_data_objects + val_neg_data_objects +  test_pos_data_objects + test_neg_data_objects:
            for k in d['indices_dict'].keys():
                rels.add(k)
        shapes_dict = {k: 1 for k in rels}

    # The benchmark datasets do not contain unary predicates -- therefore the initial feature vector dimension can be set to one
    feat_dim = 1

    train(device=device, feat_dim=feat_dim, shapes_dict=shapes_dict, train_data=train_pos_data_objects+train_neg_data_objects, val_data=val_pos_data_objects+val_neg_data_objects,
          log_directory=log_directory, model_directory=model_directory, subqueries=subqueries, args=args,
          summary_writer=writer)

    if args.test:
        print('Start Testing!')
        model = torch.load(os.path.join(model_directory, 'model.pt'))
        model.to(device)
        for param in model.parameters():
            print(type(param.data), param.size())

        _, test_acc, test_pre, test_re, test_auc = compute_metrics(test_pos_data_objects + test_neg_data_objects, model)

        print('Testing!')
        print('Accuracy for all answers: ' + str(test_acc))
        print('Precision for all answers: ' + str(test_pre))
        print('Recall for all answers: ' + str(test_re))
        print('AUC for all answers: ' + str(test_auc))

