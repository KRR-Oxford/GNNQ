from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--rules', help="File with all the rules learned by AnyBURL",
                    default='./rules/delta-1000_filtered.txt')

args = parser.parse_args()


def anyburl_atom_to_datalog_atom(atom):
    # Anyburl atoms are of the form string(X,Y)
    predicate = atom.split('(')[0]
    var1 = atom.split('(')[1][0]
    var2 = atom.split('(')[1][2]
    # Datalog atoms are of the form <string>[?X,?Y]
    return '<' + "http://dummyrel.com" + predicate + '>[?' + var1 + ',?' + var2 + ']'


def anyburl_rule_to_datalog_rule(rule):
    # Anyburl rules are of the form h(X,Y) <= p(X,Z), q(Y,Z)\n
    head = rule.split(' <= ')[0]
    body = rule.split(' <= ')[1].split(', ')
    body[-1] = body[-1][:-1]
    newhead = anyburl_atom_to_datalog_atom(head)
    newbody = []
    for atom in body:
        newbody.append(anyburl_atom_to_datalog_atom(atom))
    # Datalog rules are of the form <h>[?X,?Y] :- <p>[?X,?Z], <q>[?Y,?Z] .\n
    return newhead + ' :- ' + ', '.join(newbody) + ' .\n'


rules_file = open(args.rules, "r")
rules_lines = rules_file.readlines()
rules = []
for line in rules_lines:
    # For each line, the rule is in the fourth column
    rule = line.split('\t', 3)[3]
    # Transform AnyBURL rule to Datalog rule
    rule = anyburl_rule_to_datalog_rule(rule)
    rules.append(rule)
    # Each rule is mapped to the set of facts it entails on the dataset, currently empty

with open('rules.dlog', 'w') as f:
    for rule in rules:
        f.write(rule)
