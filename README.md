# GNNQ - A neuro-symbolic approach to answer queries on incomplete knowledge graphs
This repository contains the implementation of GNNQ.
### Requirements:
- Python 3.*
- Download source code from Github. \
`git clone git@github.com:MaxPflueger/GNNQ.git`
- Navigate to the repository. \
`cd GNNQ`
- Create python virtual environment. \
`python -m venv env`
- Start virtual environment. \
`source env/bin/activate`
- Install PyTorch. \
`pip install torch==1.9.0`
- Install all other dependencies. \
`pip install -r requirements.txt`

### Datasets
You can download a folder called "datasets" containing all datasets used for the evaluation from the following link. Ensure that the downloaded folder is placed in the root folder of the GNNQ repository. The "datasets" folder has the subfolders "watdiv" and "fb15k237" containing the WatDiv and FB15k237 benchmarks respectively. Furthermore, the "datasets" folder contains the "benchmark_queries.txt" and "rules.txt" containing the benchmark queries and the completion rules for the FB15k237 benchmarks.

### Training
In order to reproduce our experiments, navigate to the root directory of the GNNQ repository and run the following commands. We first show and example for a WatDiv benchmark. 
```
python main.py  --log_dir watdiv_q1_v0/ --aug --num_layers 4 --base_dim 16 --test --epochs 250 --batch_size 1 --positive_sample_weight 2
--train_data datasets/watdiv/dataset1 datasets/watdiv/dataset2 datasets/watdiv/dataset3 --val_data datasets/watdiv/dataset4 --test_data datasets/watdiv/dataset5 datasets/watdiv/dataset6 datasets/watdiv/dataset7
--query_string "SELECT distinct ?v0 WHERE { ?v0  <http://schema.org/caption> ?v1 . ?v0   <http://schema.org/text> ?v2 . ?v0 <http://schema.org/contentRating> ?v3 . ?v0   <http://purl.org/stuff/rev#hasReview> ?v4 .  ?v4 <http://purl.org/stuff/rev#title> ?v5 . ?v4  <http://purl.org/stuff/rev#reviewer> ?v6 . ?v7 <http://schema.org/actor> ?v6 . ?v7 <http://schema.org/language> ?v8  } 
```

All WatDiv benchmarks use the same KGs. To reproduce the other WatDiv benchmarks simply exchange the query strings. All query strings for the WatDiv benchmarks can be found in "benchmark_queries.txt" file. For the evaluation of the baseline model remove the "--aug" parameter. 

In order to reproduce our experiments on FB15k237 benchmark run the following command.

```
python main.py  --log_dir fb15k237_org/ --aug --num_layers 4 --base_dim 16 --test --epochs 250 --batch_size 40 --positive_sample_weight 2
--train_data  datasets/fb15k237/org_train_samples.pkl --val_data datasets/fb15k237/org_val_samples.pkl --test_data datasets/fb15k237/org_test_samples.pkl
--query_string "select distinct ?org where { ?org <http://dummyrel.com/organization/organization/headquarters./location/mailing_address/state_province_region> ?region . ?biggerregion <http://dummyrel.com/location/location/contains> ?region . ?biggerregion <http://dummyrel.com/location/location/adjoin_s./location/adjoining_relationship/adjoins> ?neighbourregion . ?biggerregion <http://dummyrel.com/location/country/capital> ?capital . ?neighbourregion <http://dummyrel.com/location/country/official_language> ?lang . ?capital <http://dummyrel.com/common/topic/webpage./common/webpage/category> ?category . ?capital <http://dummyrel.com/travel/travel_destination/climate./travel/travel_destination_monthly_climate/month> ?month }"  
```
To reproduce the other FB15k237 benchmarks, exchange the training, validation and testing samples for the respective query. Furthermore, exchange the query string. All query strings for the FB15k237 benchmarks can be found in "benchmark_queries.txt" file. For the evaluation of the baseline model remove the "--aug" parameter. 