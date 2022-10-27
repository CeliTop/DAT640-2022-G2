import sys
from trectools import TrecQrel, TrecRun, TrecEval

arguments = sys.argv

print("format: evaluate.py retrieved.txt qrels.txt")
print("Note: all files should be on the same local path")

retrievedPath = "retrieved.txt"
qrelsPath = "qrels.txt"
if len(arguments) == 1:
    print(
        "Assuming the files are retrieved.txt and qrels.txt if no parameters are used"
    )
elif len(arguments) == 3:
    retrievedPath = arguments[1]
    qrelsPath = arguments[2]
else:
    print("Print wrong numbers of parameters !")
    exit()

print()


def evaluation():
    r1 = TrecRun(retrievedPath)

    qrelstrec = TrecQrel(qrelsPath)

    te = TrecEval(r1, qrelstrec)

    p100 = te.get_precision(depth=100)
    rr = te.get_reciprocal_rank(depth=100)
    ndcg = te.get_ndcg(depth=100)
    map_ = te.get_map(depth=100)
    print("relevent docs: ", te.get_relevant_documents())
    print("retrieved docs: ", te.get_retrieved_documents())
    print("retrieved relevant docs: ", te.get_relevant_retrieved_documents())
    print()
    print("P@100 : \t", p100)
    print("MRR@100 : \t", rr)
    print("NDCG@100 : \t", ndcg)
    print("MAP@100 : \t", map_)
    print()


if __name__ == "__main__":
    evaluation()
