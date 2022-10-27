import pyterrier as pt
import pandas as pd
from tqdm import tqdm
import math
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# This file implements BM-25, following the formula that we used in our course.
# It is working and you can try and compare results with other BM-25 implementations.
# We will not use it for the next parts of the project because it's too slow,
# Instead we will use pyterrier's version that is much faster and provide the same
# results. However we keep this version in our project to prove that we have fully
# understood the classes on BM-25


# Init pyterrier
pt.init()

# Get MS Marco passages used in TREC-2019
dataset = pt.get_dataset("trec-deep-learning-passages")
pathCorpus = dataset.get_corpus()
path = dataset.get_index("terrier_stemmed")
queries = dataset.get_topics("test-2019")
qrels = dataset.get_qrels("test-2019")
stemmer = PorterStemmer()  # init the Porter Stemmer

# Stemmer used to process the queries
stemmer = PorterStemmer()  # init the Porter Stemmer

print("query examples")
print(queries)
print()

print("qrel examples:")
print(qrels)
print()

# Example of a document
with pt.io.autoopen(dataset.get_corpus()[0], "rt") as corpusfile:
    for l in corpusfile:
        docno, passage = l.split("\t")
        if docno == "1851737":
            print("Example: document 1851737")
            print(passage)
            break

# Get index
index = pt.IndexFactory.of(path)
print("index infos")
print(index.getCollectionStatistics().toString())


# Get the meta index
meta = index.getMetaIndex()
# Inverted index imported from pyterrier
inv = index.getInvertedIndex()
# Get the lexicon
lex = index.getLexicon()
P = {}
# Get informations about a term such as term frequency in doc or doc_frequency
def getPostings(term: str):
    Postings = {}

    le = lex.getLexiconEntry(term)
    if le == None:
        Postings = []
    else:
        for posting in inv.getPostings(le):
            docno = meta.getItem("docno", posting.getId())
            Postings[posting.getId()] = posting.getFrequency()

    P[term] = Postings

    return Postings


# Our Implementation of BM25
# Hyperparameters
b = 0.75
k1 = 1.2
# Number of documents
N = index.getCollectionStatistics().getNumberOfDocuments()
# Average document length
avgdl = index.getCollectionStatistics().getAverageDocumentLength()


def BM25(q: str, doc_id):
    d = index.getDocumentIndex().getDocumentLength(doc_id)
    score = 0
    for term in q.split():
        term = term.lower()
        postings = P[term] if term in P else getPostings(term)
        nt = len(postings)
        if nt != 0:
            ctd = postings[doc_id] if doc_id in postings else 0
            score += (
                (ctd * (1 + k1))
                / (ctd + k1 * (1 - b + b * (d / avgdl)))
                * math.log(N / nt)
            )
    return score


print("BM-25 score for the query 'wolf color' and document-id 1792478:", end=" ")
print(BM25("wolf color", 1792478))
print()


# Baseline retrieval: get the top 1000 documents based on BM-25
def getTop1000(q: str):
    print("Original query: ", q)
    stop_words = set(stopwords.words("english"))
    q = " ".join([stemmer.stem(w) for w in q.split() if w not in stop_words])
    print("query stemmed by the Porter stemmer: ", end=" ")
    print(q)

    h = []
    with pt.io.autoopen(dataset.get_corpus()[0], "rt") as corpusfile:
        for l in tqdm(corpusfile):
            docno, passage = l.split("\t")
            score = BM25(q, int(docno))
            if score > 0:
                h += [{"doc_id": docno, "score": score}]
                h.sort(key=lambda x: x["score"], reverse=True)
                h = h[:1000]
    return h


# Testing BM-25 on one query (Or multiple if you want to evaluate)
qid = 156493
queryTest = queries.loc[queries["qid"] == str(qid)]

# qrels = qrels.loc[qrels["qid"] == "19335"]

# Get the top 1000 passages and output 2 files
# First one is retrieved.txt the ranking from the baseline method
# Second one is qrels.txt the ground truth
def retrieval(queries):
    # Get top 1000 base on baseline retrieval and output a file in TREC format
    def create_file():
        with open("retrieved.txt", "w+") as file:
            for index, query in queries.iterrows():
                top1000 = getTop1000(query["query"])

                for result, rank in zip(top1000, range(0, 1000)):
                    file.write(
                        str(query["qid"])
                        + "\t"
                        + "Q0"
                        + "\t"
                        + str(result["doc_id"])
                        + "\t"
                        + str(rank)
                        + "\t"
                        + str(result["score"])
                        + "\t"
                        + "myScore"
                        + "\n"
                    )

    # Get the ground truth from the test dataset and export a file TREC qrel format
    def create_qrels():
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

    create_file()
    create_qrels()


if __name__ == "__main__":
    print("Evaluating the queries...")
    retrieval(queryTest)
    print("qrels.txt and retrieved.txt generated !")
    print()
