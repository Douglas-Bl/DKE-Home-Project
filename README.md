# DKE 2022 Home Project
This repository contains the home project for the Data and Knowledge Engineering class 2022 at Heinrich Heine University.
The project exercise can be completed at home, using the expertise and skill sets acquired in the DKE 2022 lecture.
Use this repository to prepare your solution.


## Folder Structure

- **assignment**: contains a more detailed description of the task and the required output. 
- **src**: add your code here in the suitable subfolder(s) (depending on whether you use Python or Java). 
- **test_data**:  the test data will be added to this folder. Please refer to the task description in the assignment folder for information on the format. The dummy_ids.csv file contains dummy data in the required format. 
- **output_data**: folder that must be populated with the results of the home project. Please refer to the task description in the assignment folder for information on the format. The file dummy_predictions.csv contains dummy data in the required format. 

## Train and Test
- src\python\classifier.py includes the SPARQL Query to receive the training data along with all relevant information
  as well the data preparation to be able to train a decision tree.
- src\python\test_classifier.py queries the claims from the test set and predicts their rating by using the trained model
- eval\eval.py evaluates the classification of the test set
- src\python\main.py uses all of the above to first get the training data and train the model, then it classifies the test
  claims and lastly the classification results are evaluated by calculating the accuracy, precision and recall.

## Description of the classifier
Following is a list of the features that are used for classification as well as an explanation of their rationale.

- **year and month**: The year and month in which the claim was published. For example there could be time periods, like 
  before some big election, where a lot of misinformation is spread.
- **citation_count**: The amount of citations that a claim has could be correlated to whether the statement is true or
  false. For example, it could be that false claims have less or even no citations at all, while true claims often have
  a lot of citations.
- **author_score**: A big deciding factor to intuitively guess if a claim is true or false is to look at the ratings of
  other claims that are coming from the same author. If the author is claiming a lot of false things chances are that
  their  next claim is also false, vice versa if they often tell the truth it is more likely to be true as well.
  The author score is a metric to illustrate this, where the closer it is to -1 the greater the tendency of the author
  to claim false things, and the closer it is to +1 the greater the tendency of them stating the truth.
- **mention_score**: There may be a correlation between certain things, events or persons and the rating of a claim that
  mentions them. For example, it could be that the more influential and famous a person is, the more claims about this 
  person that are false are circulating. This could also apply to country or even world-wide events. Similar to the
  author score, the mention score is a metric to illustrate how often the mention appears in a false or true claim.