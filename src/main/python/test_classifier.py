import os
import csv


def test_classification(sparql, clf, authors, mentions):

    # get claims of training set
    test_ids = set()
    with open(os.getcwd() + "/../../../eval/gold.csv", newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            test_ids.add(row[0])

    # dict with all features needed for prediction of a claim
    claim_dict = {}

    for claim_id in test_ids:

        # query to relevant information for single current claim
        test_query = """
                            PREFIX itsrdf:<https://www.w3.org/2005/11/its/rdf#>
                                PREFIX schema:<http://schema.org/>
                                PREFIX dbr:<http://dbpedia.org/resource/>
                                PREFIX nif:<http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#>
                                SELECT ?date ?claim ?claimText ?author ?mention COUNT(?citation) AS ?citation_count ?rating WHERE {
    
                                ?claim_review schema:itemReviewed <""" + claim_id + """> .
                                ?claim_review schema:reviewRating ?rating ; schema:itemReviewed ?claim .
                                ?claim schema:text ?claimText
    
                                OPTIONAL {?claim schema:author ?authorOpt}
                                OPTIONAL {?claim schema:mentions ?mentions .
                                          ?mentions nif:isString ?mentionOpt}
                                OPTIONAL {?claim schema:citation ?citation}
                                OPTIONAL {?claim schema:datePublished ?dateOpt}
                                
    
                                BIND(COALESCE(?mentionOpt, "") AS ?mention)
                                BIND(COALESCE(?dateOpt, "2020-06-15"^^xsd:date) AS ?date)
                                BIND(COALESCE(?authorOpt, "") AS ?author)
    
                                FILTER regex(?rating, "http://data.gesis.org/claimskg/rating/normalized", "i")
                                }
                            """

        sparql.setQuery(test_query)

        try:
            ret = sparql.queryAndConvert()

            for r in ret["results"]["bindings"]:
                # get data from row
                claim = r["claim"]["value"]
                claim_text = r["claimText"]["value"]
                author = r["author"]["value"]
                mention = r["mention"]["value"]
                rating = r["rating"]["value"]
                date = r["date"]["value"]
                citation_count = int(r["citation_count"]["value"])

                # normalize rating of the claim
                if rating.endswith("TRUE"):
                    rating = "TRUE"
                elif rating.endswith("FALSE"):
                    rating = "FALSE"
                elif rating.endswith("MIXTURE"):
                    rating = "NEITHER"
                elif rating.endswith("OTHER"):
                    rating = "NEITHER"

                # date = [year, month]
                date = [int(date[:4]), int(date[5:7])]

                #### store all features for current claim in claim_dict
                ## if both author and mention didn't appear in training set
                if author not in authors and mention not in mentions:
                    # if claim not initialized
                    if claim not in claim_dict:
                        claim_dict[claim] = {"author_score": 0, "mention_score": 0, "citation_count": citation_count,
                                             "mentions_count": 0, "rating": rating, "date": date, "text": claim_text}
                ## if only mention didn't appear in training set
                elif author not in authors:
                    # if claim not initialized
                    if claim not in claim_dict:
                        claim_dict[claim] = {"author_score": 0, "mention_score": mentions[mention]["score"],
                                             "citation_count": citation_count, "mentions_count": 1, "rating": rating,
                                             "date": date, "text": claim_text}
                    # if claim has multiple mentions
                    else:
                        claim_dict[claim]["mention_score"] += mentions[mention]["score"]
                        claim_dict[claim]["mentions_count"] += 1
                ## if only author didn't appear in training set
                elif mention not in mentions:
                    # if claim not initialized
                    if claim not in claim_dict:
                        claim_dict[claim] = {"author_score": authors[author]["score"], "mention_score": 0,
                                             "citation_count": citation_count, "mentions_count": 0, "rating": rating,
                                             "date": date, "text": claim_text}
                ## if both author and mention are in training set
                else:
                    # if claim not initialized
                    if claim not in claim_dict:
                        claim_dict[claim] = {"author_score": authors[author]["score"],
                                             "mention_score": mentions[mention]["score"], "citation_count": citation_count,
                                             "mentions_count": 1, "rating": rating, "date": date, "text": claim_text}
                    # if claim has multiple mentions
                    else:
                        claim_dict[claim]["mention_score"] += mentions[mention]["score"]
                        claim_dict[claim]["mentions_count"] += 1

        except Exception as e:
            print(e)

    # dict for claims with their predicted rating
    predictions = {}

    for claim in claim_dict:
        # use all features for prediction of current claim
        # because a claim can have multiple mentions the mention score is averaged
        predicted = clf.predict([[claim_dict[claim]["author_score"],
                                  (claim_dict[claim]["mention_score"] / claim_dict[claim][
                                      "mentions_count"])
                                  if claim_dict[claim]["mentions_count"] != 0 else 0,

                                  claim_dict[claim]["citation_count"],
                                  claim_dict[claim]["date"][0],
                                  claim_dict[claim]["date"][1]]])

        ## the following statements look redundant but they reformat the rating
        ## from an ndarray with the rating into only a string of the rating
        if predicted == "TRUE":
            predicted = "TRUE"

        elif predicted == "FALSE":
            predicted = "FALSE"

        # the model is trained to also predict MIXTURE and OTHER but in the
        # evaluation set these two ratings are combined into the rating NEITHER
        elif predicted == "MIXTURE" or predicted == "OTHER":
            predicted = "NEITHER"

        # store claim_id with the claim description and the predicted rating
        predictions[claim] = [claim_dict[claim]["text"], predicted]

    # save predictions into csv file
    with open(os.getcwd() + "/../../../output_data/predictions.csv", 'w', newline='\n', encoding="utf-8") as new_csv_file:
        writer = csv.writer(new_csv_file)
        for claim in predictions:
            # every row is in the format claim_id, claim_description, predicted_rating
            writer.writerow((claim, predictions[claim][0], predictions[claim][1]))

        new_csv_file.close()
