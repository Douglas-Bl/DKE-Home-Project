from SPARQLWrapper import SPARQLWrapper, JSON
from eval.eval import evaluate
from src.main.python.classifier import create_classifier
from src.main.python.test_classifier import test_classification
from joblib import dump, load


def main():

    sparql = SPARQLWrapper(
            "https://data.gesis.org/claimskg/sparql"
        )
    sparql.setReturnFormat(JSON)


    # create classifier and get dicts with author and mention score
    clf, authors, mentions = create_classifier(sparql)

    # save classifier
    #dump(clf, 'decision_tree.joblib')

    # load saved classifier
    #clf = load('decision_tree.joblib')

    # write predicted ratings of test set into output_data
    test_classification(sparql, clf, authors, mentions)

    # evaluate classification on test data
    evaluate()


if __name__ == "__main__":
    main()
