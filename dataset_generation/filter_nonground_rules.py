from argparse import ArgumentParser
import re

parser = ArgumentParser()
parser.add_argument('--input', help="File with all the rules", default='../datasets/fb15k237rules/rules/gamma-1000')
args = parser.parse_args()


def parse_rule(rule):
    # Lines have four columns separated by tabs; the rule is in the fourth
    split = line.split('\t')
    confidence = split[2]
    rule = split[3]
    atoms = rule.split(' ')
    for atom in atoms:
        if atom != '<=':
            terms = atom.split('(')[1][:-1].split(',')
            if not re.match('[A-Z]', terms[0]) or not re.match('[A-Z]', terms[1]) or not float(confidence) >= 0.25:
                return False
    return True


rules = []
input_file = open(args.input, "r")
Lines = input_file.readlines()

print("{} rules read.".format(len(Lines)))

for line in Lines:

    rule_is_pure_nonground = parse_rule(line)
    if rule_is_pure_nonground:
        rules.append(line)

#  Write irreducible rules
filtered_rules_file = args.input + '_filtered.txt'
with open(filtered_rules_file, 'w') as m:
    for rule in rules:
        m.write(rule)
    m.close()
