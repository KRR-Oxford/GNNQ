import os
import sys
log_dir = 'runs_org_01_250/'
# log_dir = 'runs_film/'
# log_dir = 'runs_person/'
query_string = ' --query_string "select distinct ?org where { ?org <http://dummyrel.com/organization/organization/headquarters./location/mailing_address/state_province_region> ?region . ?biggerregion <http://dummyrel.com/location/location/contains> ?region . ?biggerregion <http://dummyrel.com/location/location/adjoin_s./location/adjoining_relationship/adjoins> ?neighbourregion . ?biggerregion <http://dummyrel.com/location/country/capital> ?capital . ?neighbourregion <http://dummyrel.com/location/country/official_language> ?lang . ?capital <http://dummyrel.com/common/topic/webpage./common/webpage/category> ?category . ?capital <http://dummyrel.com/travel/travel_destination/climate./travel/travel_destination_monthly_climate/month> ?month }" '
# query_string = ' --query_string "select distinct ?film where { ?film <http://dummyrel.com/film/film/genre> ?genre . ?film <http://dummyrel.com/film/film/country> ?country . ?genre <http://dummyrel.com/media_common/netflix_genre/titles> ?titles . ?country <http://dummyrel.com/location/country/official_language> ?language . ?country2 <http://dummyrel.com/location/location/adjoin_s./location/adjoining_relationship/adjoins> ?country . ?country2 <http://dummyrel.com/organization/organization_member/member_of./organization/organization_membership/organization> ?org . ?event <http://dummyrel.com/base/culturalevent/event/entity_involved> ?org }" '
# query_string = ' --query_string "select distinct ?person where {?person  <http://dummyrel.com/people/person/nationality> ?place . ?person2 <http://dummyrel.com/people/person/places_lived./people/place_lived/location> ?place . ?person2 <http://dummyrel.com/award/award_winner/awards_won./award/award_honor/award_winner> ?winner . ?winner <http://dummyrel.com/film/actor/film./film/performance/film> ?film . ?genre <http://dummyrel.com/music/genre/artists> ?winner. }" '
datasets = ' --train_data  datasets/fb15k237/org_train_samples.pkl --val_data datasets/fb15k237/org_val_samples.pkl --test_data datasets/fb15k237/org_test_samples.pkl'
# datasets = ' --train_data  datasets/fb15k237/film_train_samples_02.pkl --val_data datasets/fb15k237/film_val_samples_02.pkl --test_data datasets/fb15k237/film_test_samples_02.pkl'
# datasets = ' --train_data  datasets/fb15k237/person_train_samples_03.pkl --val_data datasets/fb15k237/person_val_samples_03.pkl --test_data datasets/fb15k237/person_test_samples_03.pkl'
epochs = ' --epochs 250 '
base_dim = ' --base_dim 16 '
batch_size = ' --batch_size 40 '
query_depth = 4
val_epochs = ' --val_epochs 100 '
aggr = ' '
command1 = "main.py  " + query_string + datasets + ' --test --log_dir ' + log_dir + ' --num_layers ' + str(query_depth - 1) + epochs + base_dim + val_epochs + batch_size + aggr
output_file1 = log_dir + "3l.txt"
command2 = "main.py  " + query_string + datasets + ' --test --log_dir ' + log_dir + ' --num_layers ' + str(query_depth) + epochs + base_dim  + val_epochs + batch_size + aggr
output_file2 = log_dir + "4l.txt"
command3 = "main.py  " + query_string + datasets + ' --test --log_dir ' + log_dir + ' --num_layers ' + str(query_depth + 1) + epochs + base_dim  + val_epochs + batch_size + aggr
output_file3 = log_dir + "5l.txt"
command4 = "main.py  " + query_string + datasets + ' --test --log_dir ' + log_dir + ' --aug --num_layers ' + str(query_depth - 1) + epochs + base_dim  + val_epochs + batch_size + aggr
output_file4 = log_dir + "aug_3l.txt"
command5 = "main.py  " + query_string + datasets + ' --test --log_dir ' + log_dir + ' --aug --num_layers ' + str(query_depth) + epochs + base_dim + val_epochs + batch_size + aggr
output_file5 = log_dir + "aug_4l.txt"
os.system("nice -n 5 nohup sh -c '" +
        sys.executable + " " + command1 + " > " + output_file1 +  " && " +
        sys.executable + " " + command2 + " > " + output_file2 +  " && " +
        sys.executable + " " + command3 + " > " + output_file3 +  " && " +
        sys.executable + " " + command4 + " > " + output_file4 +  " && " +
        sys.executable + " " + command5 + " > " + output_file5 +  "' &" )