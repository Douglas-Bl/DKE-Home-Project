import os

from SPARQLWrapper import SPARQLWrapper, JSON
from sklearn import tree
import csv


def create_classifier(sparql):

    # get claims of training set to be able to ignore those when training
    test_ids = set()
    with open(os.getcwd() + "/../../../eval/gold.csv", newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            test_ids.add(row[0])

    ##### fine-tunable parameters
    ## how many query results
    training_size = 10000
    ## weights for author and mention score
    score_true = 1
    score_false = -1
    score_mixture = 0.3
    score_other = -0.4
    ## parameters for decision tree
    max_depth = 7
    min_samples_leaf = 39

    sparql = SPARQLWrapper(
        "https://data.gesis.org/claimskg/sparql"
    )
    sparql.setReturnFormat(JSON)

    # query to get claims with relevant information for training
    query = """
                        PREFIX itsrdf:<https://www.w3.org/2005/11/its/rdf#>
                            PREFIX schema:<http://schema.org/>
                            PREFIX dbr:<http://dbpedia.org/resource/>
                            PREFIX nif:<http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#>
                            SELECT ?date ?claim ?author ?mention COUNT(?citation) AS ?citation_count ?rating WHERE {
    
                            ?claim_review a schema:ClaimReview ; schema:reviewRating ?rating ; schema:itemReviewed ?claim .
                            ?claim schema:mentions ?mentions .
                            ?mentions nif:isString ?mention .
                            ?claim schema:author ?author .
                            ?claim schema:datePublished ?date .
                            ?claim_review schema:datePublished ?date_review BIND (year(?date_review) AS ?year) . FILTER (?year >= 2019) FILTER (?year <= 2021) .
                            OPTIONAL {?claim schema:citation ?citation} .
                            FILTER regex(?rating, "http://data.gesis.org/claimskg/rating/normalized", "i")
    
                            } ORDER BY ?claim LIMIT """ + str(training_size) + """
    
                        """

    sparql.setQuery(query)

    try:
        ret = sparql.queryAndConvert()

        authors = {}
        mentions = {}
        claims = set()

        for r in ret["results"]["bindings"]:
            # get data from row
            author = r["author"]["value"]
            mention = r["mention"]["value"]
            rating = r["rating"]["value"]
            claim = r["claim"]["value"]

            # do not use claims of the test set for training
            if claim in test_ids:
                continue

            # normalize rating of the claim
            if rating.endswith("TRUE"):
                rating = "TRUE"
            elif rating.endswith("FALSE"):
                rating = "FALSE"
            elif rating.endswith("MIXTURE"):
                rating = "MIXTURE"
            elif rating.endswith("OTHER"):
                rating = "OTHER"

            # save for every author how many claims they have in total
            # as well as how often their rating was TRUE, FALSE, MIXTURE or OTHER
            if author not in authors and claim not in claims:
                authors[author] = {rating: 1, "TOTAL": 1}
                for val in {"TRUE", "FALSE", "MIXTURE", "OTHER"}:
                    if val not in authors[author]:
                        authors[author][val] = 0
                claims.add(claim)
            elif claim not in claims:
                authors[author][rating] += 1
                authors[author]["TOTAL"] += 1
                claims.add(claim)

            # save for every mention how often it was part of a claim in total
            # as well as how often it was part of a claim with the rating TRUE, FALSE, MIXTURE or OTHER

            if mention not in mentions:
                mentions[mention] = {rating: 1, "TOTAL": 1}
                for val in {"TRUE", "FALSE", "MIXTURE", "OTHER"}:
                    if val not in mentions[mention]:
                        mentions[mention][val] = 0
            else:
                mentions[mention][rating] += 1
                mentions[mention]["TOTAL"] += 1

        # calculate a score for each mention that represents the tendency with which rating the mention appears
        for mention in mentions:
            score = (mentions[mention]["TRUE"] * score_true + mentions[mention]["FALSE"] * score_false + mentions[mention][
                "MIXTURE"] * score_mixture
                     + mentions[mention]["OTHER"] * score_other) / mentions[mention]["TOTAL"]
            mentions[mention]["score"] = score

        # calculate a score for each author that represents which rating their claims tend to receive
        for author in authors:
            score = (authors[author]["TRUE"] * score_true + authors[author]["FALSE"] * (score_false) + authors[author][
                "MIXTURE"] * score_mixture +
                     authors[author]["OTHER"] * score_other) / authors[author]["TOTAL"]
            authors[author]["score"] = score

        # array for features [author_score, mention_score, citation_count, year, month]
        X = []
        # array for labels [RATING_VALUE]
        Y = []

        for r in ret["results"]["bindings"]:
            # get data from row
            author = r["author"]["value"]
            mention = r["mention"]["value"]
            rating = r["rating"]["value"]
            date = r["date"]["value"]
            citation_count = int(r["citation_count"]["value"])
            claim = r["claim"]["value"]

            # do not use claims of the test set for training
            if claim in test_ids:
                continue

            # normalize rating of the claim
            if rating.endswith("TRUE"):
                rating = "TRUE"
            elif rating.endswith("FALSE"):
                rating = "FALSE"
            elif rating.endswith("MIXTURE"):
                rating = "MIXTURE"
            elif rating.endswith("OTHER"):
                rating = "OTHER"

            # date = [year, month]
            date = [int(date[:4]), int(date[5:7])]

            """
            if author not in authors and mention not in mentions:
                X.append([0, 0, citation_count, date[0], date[1]])
            elif author not in authors:
                X.append([0, mentions[mention]["score"], citation_count, date[0], date[1]])
            elif mention not in mentions:
                X.append([authors[author]["score"], 0, citation_count, date[0], date[1]])
            else:
                X.append([authors[author]["score"], mentions[mention]["score"], citation_count, date[0], date[1]])
            Y.append(rating)
            """

            # prepare training data with their labels
            X.append([authors[author]["score"], mentions[mention]["score"], citation_count, date[0], date[1]])
            Y.append(rating)

        # classify using a Decision Tree
        clf = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        clf = clf.fit(X, Y)

        return clf, authors, mentions

    except Exception as e:
        print(e)
