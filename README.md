# GNNQ: A Neuro-Symbolic Approach to Query Answering over Incomplete Knowledge Graphs

### About
The GNNQ repository contains the source code for the GNNQ system presented in the paper "GNNQ: A Neuro-Symbolic Approach to Query Answering over Incomplete Knowledge Graphs" (insert link) published at ISWC22. 

GNNQ is a neuro-symbolic system to answer monadic tree-like conjunctive queries over incomplete KGs. GNNQ ﬁrst symbolically augments an input KG (formally a set of facts) with additional facts representing subsets matching connected query fragments, and then applies a generalisation of the Relational Graph Convolutional Networks (RGCNs) model to the augmented KG to produce the predicted query answers.

### Source Code
Clone the GNNQ repository.

` git clone https://github.com/KRR-Oxford/GNNQ.git ` or ` git clone git@github.com:KRR-Oxford/GNNQ.git `

### Requirements
We assume that the following is pre-installed. We used the respective versions specified in brackets.
- python (3.8.10 or higher)
- pip (19.2.3 or higher)
- venv

If this is not the case, instructions for the installation of the requirements can be found [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

Please follow the steps outlined below to reproduce the experiments.

### Dependencies
Follow the instructions below to install all dependencies required for our experiments.
- Navigate to the `GNNQ/` directory. \
```cd path/to/download/GNNQ```
- Create a virtual environment. \
```python -m venv env```
- Start virtual environment. \
```source env/bin/activate```
- Install PyTorch. Replace `${CUDA}` with `cpu` or `cu113`. \
```pip install torch==1.11.0+${CUDA}  --extra-index-url https://download.pytorch.org/whl/${CUDA}```
- Install PyTorch Scatter. Replace `${CUDA}` with `cpu` or `cu113`. \
```pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+${CUDA}.html```
- Install all other dependencies. \
```pip install -r requirements.txt```

### Datasets
The `datasets/` directory, containing both the WatDiv-Qi and the FB15k237-Qi benchmarks, can be downloaded from fighshare (https://figshare.com/s/c81e802987081569adab). Unzip the downloaded .zip-file and place the `datasets/` directory in the `GNNQ/` directory.

### Run Experiments
To run an experiment on the WatDiv benchmarks, run a variant of the following command from the GNNQ folder. Please remember that the virtual environment needs to be active. The following command is exemplary for the  WatDiv-Q1 benchmark. 
```
python main.py  --log_dir watdiv_q1_4l_aug/ --num_layers 4 --aug --test --train_data datasets/watdiv/train_samples --val_data datasets/watdiv/val_samples --test_data datasets/watdiv/test_samples --query_string "SELECT distinct ?v0 WHERE { ?v0  <http://schema.org/caption> ?v1 . ?v0   <http://schema.org/text> ?v2 . ?v0 <http://schema.org/contentRating> ?v3 . ?v0   <http://purl.org/stuff/rev#hasReview> ?v4 .  ?v4 <http://purl.org/stuff/rev#title> ?v5 . ?v4  <http://purl.org/stuff/rev#reviewer> ?v6 . ?v7 <http://schema.org/actor> ?v6 . ?v7 <http://schema.org/language> ?v8  }" 
```

To run experiments on other WatDiv benchmarks, specify a new logging directory using the `--log_dir` parameter and exchange the query specified by the `--query_string` parameter. All WatDiv benchmark queries can be found in the `datasets/benchmark_queries.txt`- file. To reproduce experiments using the baseline model, remove the `--aug` parameter. The number of layers for all models can be specified using the `--num_layers`parameter.

To reproduce an experiment on the FB15k237 benchmarks, run a variant of the following command from the GNNQ folder. Please remember again that the virtual environment needs to be active. The following command is exemplary for the FB15k237-Q1 benchmark.

```
python main.py  --log_dir fb15k237_q1_4l_aug/ --num_layers 4 --aug --test --batch_size 40 --train_data  datasets/fb15k237/org_train_samples --val_data datasets/fb15k237/org_val_samples --test_data datasets/fb15k237/org_test_samples --query_string "select distinct ?org where { ?org <http://dummyrel.com/organization/organization/headquarters./location/mailing_address/state_province_region> ?region . ?biggerregion <http://dummyrel.com/location/location/contains> ?region . ?biggerregion <http://dummyrel.com/location/location/adjoin_s./location/adjoining_relationship/adjoins> ?neighbourregion . ?biggerregion <http://dummyrel.com/location/country/capital> ?capital . ?neighbourregion <http://dummyrel.com/location/country/official_language> ?lang . ?capital <http://dummyrel.com/common/topic/webpage./common/webpage/category> ?category . ?capital <http://dummyrel.com/travel/travel_destination/climate./travel/travel_destination_monthly_climate/month> ?month }"  
```
To reproduce experiments on other FB15k237 benchmarks, specify a new logging directory using the `--log_dir` parameter and exchange the query specified by the `--query_string` parameter. All FB15k237 benchmark queries can be found in the `datasets/benchmark_queries.txt`-file. Furthermore, specify the training and testing samples for the respective query using the `--train_data` and `--test_data` parameters and specify the respective "dummy" validation samples using the `--val_data` parameter (the sample files for the FB15k237 benchmarks are named with the answer variable of the respective query). All samples files can be found in the `datasets/fb15k237/` directory. To reproduce experiments using the baseline model, remove the `--aug` parameter. The number of layers for all models can be specified using the `--num_layers`parameter.

### Tune Hyperparameters:
To tune hyperparameters for a benchmark use the `--tune` parameter. This will start an Optuna study with 100 trials.

