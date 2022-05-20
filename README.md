# PersonalizedGNN

PersonalizedGNN is a graph neural network model that can predict the possible cancer driver genes.



## Run

install requirements

```shell
pip install -r requirements.txt
```



construct dataset

```shell
python ./constuct_single_dataset.py
```



train

```shell
python -u ./gat.py --gpu=0 --dataset='BRCA/' --filecode=1
```



process results

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
| process_result.py           | process raw train result, rank and evaluate |



## Performance

We use the average precision of top 30 genes as measurement and tested our model on BRCA, LUSC, and LUAD cancer datasets.

| dataset name                | precision                                   |
| --------------------------- | ------------------------------------------- |
| BRCA                        | 0.661139                                    |
| LUSC                        | 0.720994                                    |
| LUAD                        | 0.897047                                    |



## Requirements

dgl == 0.8.0.post1
h5py == 3.4.0
numpy == 1.21.2
pandas == 1.3.3
scikit_learn == 1.0.2
scipy == 1.7.1
torch == 1.10.0+cu102
tqdm == 4.62.3

see `requirments.txt`



## Copyright

Only used for scientific research and communication. **Unauthorized commercial use is prohibited.**