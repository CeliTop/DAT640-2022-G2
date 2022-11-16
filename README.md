# DAT640 Project
## Group 2 - TREC 2019 passage reranking
##### Geoffrey Brenne - Paul Duffaut - Célian Debéthune

Here is the group project carried out for the DAT640 subject taught at the University of Stavanger.
We chose the TREC-2019 passage reranking project and developed our project in python using Juptyter-Notebook.
We then wrote our own code to make it presentable and to show how to get our results.

The python modules used are:
- [pyterrier](https://github.com/terrier-org/pyterrier)
- pandas
- [trectools][trec_tools_link]
- [sentence_transformers][transformers_link]

## Features

- Obtain a top 1000 passages from the MS MARCO collection (8.8 Million documents) according to a query and the BM25 algorithm
- Evaluate the performance of our baseline retrieval

## How tu use it

##### Baseline Method
There are two files for performing BM-25 retrieval:
Both of them outputs `retrieved.txt` and `qrels.txt`: 

"retrieved.txt": The top 1000 passages ranked using BM-25 in TREC format: `BM_25_retrieval.py` and `BM_25_pyterrier.py`
| qid | Q0 | docno | rank | score | tag |
| ------ | ------ | ------ | ------ | ------ | ------ |
| 1 | Q0 | nhslo3844_12_012186 | 1 | 1.73315273652 | mySystem |
| 1 | Q0 | nhslo1393_12_003292 | 2 | 1.72581054377  | mySystem |
| 1 | Q0 | nhslo3844_12_002212 | 3 | 1.72522727817 | mySystem |
| 1 | Q0 | nhslo3844_12_012182 | 4 | 1.72522727817 | mySystem |
| 1 | Q0 | nhslo1393_12_003296 | 5 | 1.71374426875 | mySystem |

"qrels.txt": The test qrels for the same dataset, used to evaluate the model:
| qid | 0 | docno | relevance |
| ------ | ------ | ------ | ------ |
| 1 | 0 | aldf.1864_12_000027 | 1 |
| 1 | 0 | aller1867_12_000032 | 2 |
| 1 | 0 | aller1868_12_000012 | 0 |
| 1 | 0 | aller1871_12_000640 | 1 |
| 1 | 0 | arthr0949_12_000945 | 0 |

> Note: The tables are taken from the examples of the [TREC-TOOLS documentation][trec_tools_link]

Running one of the followings should generate the two files:
```sh
python3 BM_25_retrieval.py
```
```sh
python3 BM_25_pyterrier.py
```
Since the first BM-25 implementation is very slow, we use it to rank one query at a time, and the second one is ranking the whole test dataset.
The results are very similar, the same passages are retrieved, sometimes order changes a bit when BM-25 scores are very close.

Then you can use `trec_eval` and the text files to obtain performances results of the baseline method:
```sh
./trec_eval ./qrels.txt ./retrieved.txt
```

##### Advanced Method

After performing the first ranking, you can use `reranking.py` to apply a reranking thanks to a BERT model:
```sh
python3 reranking.py
```

This will output a file called `reranked.txt` in the same format as retrieved.txt. You can then perform a second evaluation with: 
```sh
./trec_eval ./qrels.txt ./reranked.txt
```

## Installation
Some packages are used, but the installation is straightforward.

[//]: # (Everythin after this will be hide)
   [trec_tools_link]: <https://github.com/joaopalotti/trectools>
   [transformers_link]: <https://www.sbert.net/>
