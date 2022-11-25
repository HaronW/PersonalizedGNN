# PersonalizedGNN

PersonalizedGNN is a graph neural network model that can predict the possible cancer driver genes.

- Please setup your own path.
- The following process are used to learn and rank one dataset (one patient).
- For multiple datasets, you should repeat Step.2 ~ Step.3 on each dataset and use the function multi_dataset() in process_result.py to integrate the results.
- PersonalizedGNN takes about 20 mins to learn and rank one dataset on a computer with NVIDIA RTX 3090.



## Run

### Step.1 install requirements

```shell
pip install -r requirements.txt
```



### Step.2 construct dataset

```shell
python ./constuct_single_dataset.py
```



### Step.3 train

```shell
python -u ./gat.py --gpu=0 --dataset='BRCA/' --filecode=1
```



### Step.4 process results

```shell
python ./process_result.py
```



## Files

| file name | note |
| --------------------------- | ------------------------------------------- |
| gat.py                      | train PersonalizedGNN model                 |
| models.py                   | PersonalizedGNN model                       |
| utils.py                    | tools for PersonalizedGNN                   |
| construct_single_network.py | generate Paired-SSN dataset                 |
| process_result.py           | process train result, rank and evaluate     |



## Performance

We use the average precision of top 30 genes as measurement and tested our model on BRCA, LUSC, and LUAD cancer datasets (all the patients).

| dataset name                | precision                                   |
| --------------------------- | ------------------------------------------- |
| BRCA                        | 0.661139                                    |
| LUSC                        | 0.720994                                    |
| LUAD                        | 0.897047                                    |



## Requirements

dgl-cuda11.1>=0.7.0
numpy>=1.21.2
pandas>=1.3.3
scikit_learn>=0.24.2
scipy>=1.7.1
torch>=1.10.0+cu102
tqdm>=4.62.3

see `requirments.txt`



## Copyright

Only used for scientific research and communication. **Unauthorized commercial use is prohibited.**
