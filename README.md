#GNNQ: A Neuro-Symbolic Approach to Query Answering over Incomplete Knowledge Graphs
The GNNQ folder contains the source code and all datasets required to reproduce the experiments for the GNNQ paper.

### Requirements:
We assume that the following is pre-installed. We used the version specified in brackets.
- python (3.7 or higher)
- pip (19.2.3 or higher)
- venv

If this is not the case, instructions for the installation of the requirements can be found [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

Please follow the steps outlined below to reproduce the experiments.

###Dependencies:
Execute the following commands in the command line to install all dependencies required for our experiments.
- Navigate to the GNNQ folder. \
```cd path/to/download/GNNQ```
- Create python virtual environment. \
```python -m venv env```
- Start virtual environment. \
```source env/bin/activate```
- Install PyTorch. \
```pip install torch==1.9.0```
- Install all other dependencies. \
```pip install -r requirements.txt```

### Run experiments:
Note, that all the commands below require a '--val_data' parameter. The value of this parameter does not influence the outcome of our experiments. However, for technical reasons, our implementation expects that this parameter is passed a file containing a set of examples over an appropriate set of predicates. For this reason, we provide "dummy" validation samples for all benchmarks in the datasets folder. 

Now, to reproduce an experiment on the WatDiv benchmarks, run a version of the following command from the GNNQ folder. Please note that the virtual environment needs to be active. The following command is exemplary for the  WatDiv/Q1 benchmark. 
```
python main.py  --log_dir watdiv_q1/ --aug --num_layers 4 --test --batch_size 1 --positive_sample_weight 2 --train_data datasets/watdiv/dataset1 datasets/watdiv/dataset2 datasets/watdiv/dataset3 --val_data datasets/watdiv/dataset4 --test_data datasets/watdiv/dataset5 datasets/watdiv/dataset6 datasets/watdiv/dataset7 --query_string "SELECT distinct ?v0 WHERE { ?v0  <http://schema.org/caption> ?v1 . ?v0   <http://schema.org/text> ?v2 . ?v0 <http://schema.org/contentRating> ?v3 . ?v0   <http://purl.org/stuff/rev#hasReview> ?v4 .  ?v4 <http://purl.org/stuff/rev#title> ?v5 . ?v4  <http://purl.org/stuff/rev#reviewer> ?v6 . ?v7 <http://schema.org/actor> ?v6 . ?v7 <http://schema.org/language> ?v8  }" 
```

To reproduce experiments on other WatDiv benchmarks, specify a new logging directory using the `--log_dir` parameter, exchange the query string specified by the `--query_string` parameter. All query strings can be found in the `datasets/benchmark_queries.txt`-file. To reproduce experiments using the baseline model, remove the `--aug` parameter. The number of layers for all models can be specified using the `--num_layers`parameter.

To reproduce an experiment on the FB15k237 benchmark, run a version of the following command from the GNNQ folder. Please note again that the virtual environment needs to be active. The following command is exemplary for the FB15k237/Q1 benchmark.

```
python main.py  --log_dir fb15k237_q1/ --aug --num_layers 4 --test --batch_size 40 --train_data  datasets/fb15k237/org_train_samples.pkl --val_data datasets/fb15k237/org_val_samples.pkl --test_data datasets/fb15k237/org_test_samples.pkl --query_string "select distinct ?org where { ?org <http://dummyrel.com/organization/organization/headquarters./location/mailing_address/state_province_region> ?region . ?biggerregion <http://dummyrel.com/location/location/contains> ?region . ?biggerregion <http://dummyrel.com/location/location/adjoin_s./location/adjoining_relationship/adjoins> ?neighbourregion . ?biggerregion <http://dummyrel.com/location/country/capital> ?capital . ?neighbourregion <http://dummyrel.com/location/country/official_language> ?lang . ?capital <http://dummyrel.com/common/topic/webpage./common/webpage/category> ?category . ?capital <http://dummyrel.com/travel/travel_destination/climate./travel/travel_destination_monthly_climate/month> ?month }"  
```
To reproduce experiments on other FB15k237 benchmarks, specify a new logging directory using the `--log_dir` parameter, exchange the query string specified by the `--query_string` parameter. All query strings can be found in the `datasets/benchmark_queries.txt`-file. Furthermore, specify the training and testing samples for the respective query using the `--train_data` and `--test_data` parameters and specify the respective "dummy" validation samples using the `--val_data` parameter. All samples can be found in the `datasets/fb15k237` directory. To reproduce experiments using the baseline model, remove the `--aug` parameter. The number of layers for all models can be specified using the `--num_layers`parameter.

