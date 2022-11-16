from sentence_transformers import CrossEncoder
import pyterrier as pt
import pandas as pd

PATH_TO_TOP_1000 = "retrieved.txt"
OUTPUT_PATH = "reranked.txt"

# Init
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", 150)
pt.init()

dataset = pt.get_dataset("trec-deep-learning-passages")

# Get the previously retrieved top 1000 (by a baseline method)
retrieved = pd.read_csv(PATH_TO_TOP_1000, sep=" ")
retrieved.columns = ["qid", "Q0", "docID", "rank", "score", "system"]
print(retrieved.dtypes)
print(retrieved.head(n=5))
print()

# Get the queries
queries = dataset.get_topics("test-2020")
queries = queries.astype({"qid": "int64", "query": "string"})
print(queries.dtypes)
print("query examples")
print(queries.head(n=5))
print()


# Get the text corpus
pathCorpus = dataset.get_corpus()
print(pathCorpus[0])
print("Load CSV...")
corpus = pd.read_csv(pathCorpus[0], sep="\t")
corpus.columns = ["docno", "text"]
corpus = corpus.astype({"text": "string"})
print("corpus examples:")
print(corpus.head(n=5))
print()

model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2", max_length=512)


def getReranked(qid):
    querytext = queries.loc[queries["qid"] == qid].iloc[0]["query"]
    print("query text: ", querytext)
    docIds = retrieved.loc[retrieved["qid"] == qid]["docID"]
    print(docIds.head(n=5))
    docs = corpus.loc[corpus["docno"].isin(docIds)]
    print(docs.head(n=5))
    print()

    print("Predict...")
    couples = [(querytext, docText) for docText in docs["text"]]
    scores = model.predict(couples)
    print(scores)

    print("Sort...")
    sorted_indices = [i[0] for i in sorted(enumerate(scores), key=lambda x: -x[1])]

    top = docs.iloc[sorted_indices]
    return top


s = ""
numberquery = 0
for qid in retrieved["qid"].unique():
    print(numberquery, " query processed ...")
    numberquery += 1
    top = getReranked(qid)
    i = 0
    for index, row in top.iterrows():
        s += (
            str(qid)
            + " "
            + "Q0"
            + " "
            + str(row["docno"])
            + " "
            + str(i)
            + " "
            + str(1 / (i + 1))
            + " "
            + "BERT"
            + "\n"
        )
        i += 1


with open(OUTPUT_PATH, "w+") as file:
    file.write(s)
