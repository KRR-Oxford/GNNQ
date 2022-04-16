import pickle
import torch


def load_fb15k237_benchmark(data):
    infile = open(data, 'rb')
    pos_samples, pos_answers, neg_samples, neg_answers = pickle.load(infile)
    infile.close()
    return pos_samples + neg_samples, [[a] for a in pos_answers + neg_answers], torch.cat(
        (torch.ones(len(pos_answers)), torch.zeros(len(neg_answers))), dim=0).unsqueeze(dim=1), [[True]] * len(
        pos_answers + neg_answers), None
