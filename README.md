# 6th-12th Grade Argument Feedback
### Purpose
This project's purpose is to categorize argumentative elements in student writing as "effective," "adequate," or "ineffective." A model will be trained on data that represents the 6th-12th grade population of the United States of America. This model will cut down on bias and help in giving students feedback on their argumentative writing, which will in turn help students become more confident, creative, and proficient writers. With the use of automated guidance, students will improve their critical thinking and civic engagement skills as they practice argumentative writing.

## Data 
We got our data from the "Feedback Prize - Predicting Effective Arguments" Kaggle competetion. The goal of the competition was to classify a given arguement into either Ineffective, Adequate or Effective.

Although the data was extremely imbalanced with Adequate having more than double the amount of both Ineffective and Effective.
![Effectiveness Piechart](GitImages/EffectivenessImbalance.png?raw=true "Effectiveness Piechart")

Each argument was also classified into one of 7 discourse types with a distribution as follows
![Discourse Type Piechart](GitImages/typeImbalance.png?raw=true "Discourse Type Piechart")

## Pre-Processing

### Data Cleaning
* Converted all the words to lower case. 
* Removed trailing and leading spaces. 
* Removed all non-alphabetic characters with the Regular Expression library.
* Stemmed the text in the dataset with the NLTK library. 
* We decided to keep the stopwords because we discovered it improved our model’s accuracy.

### Vectorizing
For our vectorization, we decided to use the TfidfVectorizer function on the cleaned data. The number of words that came out was 10,191.

The most popular word for each level of effectiveness is illustrated by this word cloud
![Effectiveness Word Cloud](GitImages/wordCloud.png?raw=true "Effectiveness Word Cloud]")

### One-Hot Encoding
We used one-hot encoding to preprocess the categorical features for our machine model.

### Oversampling
We performed oversampling on the training data to combat data imbalance. 
