# This implementation of BM-25 using pyterrier does exactly the same thing as
# our own implementation of BM-25 in BM_25_retrieval.py, just in a faster way.
# We will use this version to generate the results on the whole test dataset and
# for the next parts of the project

import pyterrier as pt
import pandas as pd
import gzip
import shutil
import os as os

# Init pyterrier
pt.init()

# Get MS Marco passages used in TREC-2019
dataset = pt.get_dataset("trec-deep-learning-passages")

# Get corpus
pathCorpus = dataset.get_corpus()

# Get the index stemmed (Porter stemmer)
path = dataset.get_index("terrier_stemmed")
index = pt.IndexFactory.of(path)

# Get the queries
queries = dataset.get_topics("test-2019")
print("query examples")
print(queries)
print()

# Get the qrels
qrels = dataset.get_qrels("test-2019")
print("qrel examples:")
print(qrels)
print()

# Get the BM-25 model
bm25 = pt.BatchRetrieve(index, wmodel="BM25")

# Run BM-25 on the whole test dataset
pt.Experiment(
    [bm25],
    queries,
    qrels,
    eval_metrics=["map", "recip_rank", "ndcg"],
    save_dir="./",
    save_mode="overwrite",
    dataframe=True,
)

# Run BM-25 on a subset of queries
queries_uni = queries.loc[queries["qid"] == str(156493)]
pt.Experiment(
    [bm25],
    queries_uni,
    qrels,
    eval_metrics=["map", "recip_rank", "ndcg"],
    perquery=True,
    dataframe=True,
)

# Generating output files

with open("qrels.txt", "w+") as file:
    for index, query in qrels.iterrows():
        file.write(
            str(query["qid"])
            + "\t"
            + "0"
            + "\t"
            + str(query["docno"])
            + "\t"
            + str(query["label"])
            + "\n"
        )


with gzip.open("BR(BM25).res.gz", "rb") as f_in:
    with open("retrieved.txt", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
os.remove("BR(BM25).res.gz")
