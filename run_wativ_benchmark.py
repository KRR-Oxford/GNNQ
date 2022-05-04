import os
import sys
log_dir = 'watdiv_q1_v0_250/'
# log_dir = 'watdiv_q1_v4/'
# log_dir = 'watdiv_q1_v6/'
# log_dir = 'watdiv_q2_v2_250/'
# log_dir = 'watdiv_q2_v7/'
# log_dir = 'watdiv_q2_v8/'
query_string = '--query_string "SELECT distinct ?v0 WHERE { ?v0  <http://schema.org/caption> ?v1 . ?v0   <http://schema.org/text> ?v2 . ?v0 <http://schema.org/contentRating> ?v3 . ?v0   <http://purl.org/stuff/rev#hasReview> ?v4 .  ?v4 <http://purl.org/stuff/rev#title> ?v5 . ?v4  <http://purl.org/stuff/rev#reviewer> ?v6 . ?v7 <http://schema.org/actor> ?v6 . ?v7 <http://schema.org/language> ?v8  }" '
# query_string = '--query_string "SELECT distinct ?v4 WHERE { ?v0  <http://schema.org/caption> ?v1 . ?v0   <http://schema.org/text> ?v2 . ?v0 <http://schema.org/contentRating> ?v3 . ?v0   <http://purl.org/stuff/rev#hasReview> ?v4 .  ?v4 <http://purl.org/stuff/rev#title> ?v5 . ?v4  <http://purl.org/stuff/rev#reviewer> ?v6 . ?v7 <http://schema.org/actor> ?v6 . ?v7 <http://schema.org/language> ?v8  }" '
# query_string = '--query_string "SELECT distinct ?v6 WHERE { ?v0  <http://schema.org/caption> ?v1 . ?v0   <http://schema.org/text> ?v2 . ?v0 <http://schema.org/contentRating> ?v3 . ?v0   <http://purl.org/stuff/rev#hasReview> ?v4 .  ?v4 <http://purl.org/stuff/rev#title> ?v5 . ?v4  <http://purl.org/stuff/rev#reviewer> ?v6 . ?v7 <http://schema.org/actor> ?v6 . ?v7 <http://schema.org/language> ?v8  }" '
# query_string = '--query_string "SELECT distinct ?v2 WHERE { ?v0 <http://schema.org/legalName> ?v1 . ?v0 <http://purl.org/goodrelations/offers> ?v2 . ?v2  <http://schema.org/eligibleRegion> ?v10 . ?v2  <http://purl.org/goodrelations/includes> ?v3 . ?v4 <http://schema.org/jobTitle> ?v5 . ?v4 <http://xmlns.com/foaf/homepage> ?v6 . ?v4 <http://db.uwaterloo.ca/~galuc/wsdbm/makesPurchase> ?v7 . ?v7 <http://db.uwaterloo.ca/~galuc/wsdbm/purchaseFor> ?v3 . ?v3 <http://purl.org/stuff/rev#hasReview> ?v8 . ?v8 <http://purl.org/stuff/rev#totalVotes> ?v9 .}"'
# query_string = '--query_string "SELECT distinct ?v7 WHERE { ?v0 <http://schema.org/legalName> ?v1 . ?v0 <http://purl.org/goodrelations/offers> ?v2 . ?v2  <http://schema.org/eligibleRegion> ?v10 . ?v2  <http://purl.org/goodrelations/includes> ?v3 . ?v4 <http://schema.org/jobTitle> ?v5 . ?v4 <http://xmlns.com/foaf/homepage> ?v6 . ?v4 <http://db.uwaterloo.ca/~galuc/wsdbm/makesPurchase> ?v7 . ?v7 <http://db.uwaterloo.ca/~galuc/wsdbm/purchaseFor> ?v3 . ?v3 <http://purl.org/stuff/rev#hasReview> ?v8 . ?v8 <http://purl.org/stuff/rev#totalVotes> ?v9 .}"'
# query_string = '--query_string "SELECT distinct ?v8 WHERE { ?v0 <http://schema.org/legalName> ?v1 . ?v0 <http://purl.org/goodrelations/offers> ?v2 . ?v2  <http://schema.org/eligibleRegion> ?v10 . ?v2  <http://purl.org/goodrelations/includes> ?v3 . ?v4 <http://schema.org/jobTitle> ?v5 . ?v4 <http://xmlns.com/foaf/homepage> ?v6 . ?v4 <http://db.uwaterloo.ca/~galuc/wsdbm/makesPurchase> ?v7 . ?v7 <http://db.uwaterloo.ca/~galuc/wsdbm/purchaseFor> ?v3 . ?v3 <http://purl.org/stuff/rev#hasReview> ?v8 . ?v8 <http://purl.org/stuff/rev#totalVotes> ?v9 .}"'
datasets = ' --train_data datasets/watdiv/dataset1 datasets/watdiv/dataset2 datasets/watdiv/dataset3 --val_data datasets/watdiv/dataset4 --test_data datasets/watdiv/dataset5 datasets/watdiv/dataset6 datasets/watdiv/dataset7'
epochs = ' --epochs 250 '
base_dim = ' --base_dim 16 '
batch_size = ' --batch_size 1 '
query_depth = 4
sample_weight = ' --positive_sample_weight 2 '
aggr = ' '
command1 = "main.py  " + query_string + datasets + ' --test --log_dir ' + log_dir + ' --num_layers ' + str(query_depth - 1) + epochs + base_dim + sample_weight + batch_size + aggr
output_file1 = log_dir + "3l.txt"
command2 = "main.py  " + query_string + datasets + ' --test --log_dir ' + log_dir + ' --num_layers ' + str(query_depth) + epochs + base_dim + sample_weight + batch_size + aggr
output_file2 = log_dir + "4l.txt"
command3 = "main.py  " + query_string + datasets + ' --test --log_dir ' + log_dir + ' --num_layers ' + str(query_depth + 1) + epochs + base_dim + sample_weight + batch_size + aggr
output_file3 = log_dir + "5l.txt"
command4 = "main.py  " + query_string + datasets + ' --test --log_dir ' + log_dir + ' --aug --num_layers ' + str(query_depth - 1) + epochs + base_dim + sample_weight + batch_size + aggr
output_file4 = log_dir + "aug_3l.txt"
command5 = "main.py  " + query_string + datasets + ' --test --log_dir ' + log_dir + ' --aug --num_layers ' + str(query_depth) + epochs + base_dim + sample_weight + batch_size + aggr
output_file5 = log_dir + "aug_4l.txt"
os.system("nice -n 5 nohup sh -c '" +
        sys.executable + " " + command1 + " > " + output_file1 +  " && " +
        sys.executable + " " + command2 + " > " + output_file2 +  " && " +
        sys.executable + " " + command3 + " > " + output_file3 +  " && " +
        sys.executable + " " + command4 + " > " + output_file4 +  " && " +
        sys.executable + " " + command5 + " > " + output_file5 +  "' &" )