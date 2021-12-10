from argparse import ArgumentParser
from collections import defaultdict
import re
import torch


def filter_rules(rules_dir, num_rules_per_predicate):
    input_file = open(rules_dir, "r")
    Lines = input_file.readlines()

    print("{} rules read.".format(len(Lines)))
    rules_by_head_predicate = defaultdict(list)
    for line in Lines:
        # Lines have four columns separated by tabs; the rule is in the fourth
        split = line.split('\t')
        num_occurances = split[0]
        confidence = split[2]
        rule = split[3]
        atoms = rule.split(' ')
        filter = False
        for atom in atoms:
            if atom != '<=':
                terms = atom.split('(')[1][:-1].split(',')
                if not re.match('[A-Z]', terms[0]) or not re.match('[A-Z]', terms[1]) or not float(
                        confidence) >= 0.25 or not int(num_occurances) <= 50:
                    filter = True
                    break
        if not filter:
            head_predicate = atoms[0].split('(')[0]
            rules_by_head_predicate[head_predicate].append(line)
    for head_predicate in rules_by_head_predicate:
        rules = rules_by_head_predicate[head_predicate]
        # randomly choose rules of every predicate
        rules_by_head_predicate[head_predicate] = [rules[i] for i in
                                                   torch.randperm(len(rules))[:num_rules_per_predicate]]
    return rules_by_head_predicate


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--rules_dir', help="File with all the rules",
                        default='./rules/delta-1000')
    parser.add_argument('--num_rules_per_predicate', type=int, default=10)
    args = parser.parse_args()

    filtered_rules_file = args.rules_dir + '_filtered.txt'
    rules_by_head_predicate = filter_rules(args.rules_dir, args.num_rules_per_predicate)
    with open(filtered_rules_file, 'w') as m:
        for head_predicate in rules_by_head_predicate:
            for rule in rules_by_head_predicate[head_predicate]:
                m.write(rule)
        m.close()
