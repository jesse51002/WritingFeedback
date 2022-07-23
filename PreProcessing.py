import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler

train = pd.read_csv("./Dataset/train.csv")

print("Categories Before: ", train['discourse_effectiveness'].value_counts())

train.drop(['discourse_id', 'essay_id'] , axis=1, inplace=True)

# ########## Clean data
def cleanText(df):

    stopWords = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    lengthArr = []
    wordCountArr = []
    sentCountArr = []

    for index, row in df.iterrows():

        curText = row.discourse_text

        # lower case conversion
        curText = curText.lower()
        # removes all the trailing and leading spaces
        curText = curText.strip()

        # Saves the lengths of the writing from the model
        lengthArr.append(len(curText))
        wordCountArr.append(len(curText.split()))
        #
        sentCountArr.append(len([x for x in re.split(r"[\n\.\?\!]+", curText) if len(x) > 0]))

        #Removes stop words
        def remove_stop(x):
            return " ".join([word for word in str(x).split() if word not in stopWords])

        curText = remove_stop(curText)

        # removing all non alpha numeric char
        curText = re.sub(r'[^a-z0-9 ]+', '', curText)
        # removing "..." (multiple periods in a row)
        curText = re.sub(r'([.])\1+', '', curText)
        # stems the text
        curText = stemmer.stem(curText)

        # removing multiple spaces in a row
        curText = re.sub(r'(\s\s)+', ' ', curText)

        # replaces the text
        df.at[index, 'discourse_text'] = curText

    df['StringLength'] = lengthArr
    df['WordCount'] = wordCountArr
    df['SentenceCount'] = sentCountArr


cleanText(train)

# Vectorized strings
countVec = CountVectorizer(
    ngram_range=(1,2),
    min_df=15
)

# Fits the vectorized with train data
train_vectors = countVec.fit_transform(train['discourse_text'])

# Gets a list of all the words in the vector
vector_features = countVec.get_feature_names()
# print("Vector features: ", vector_features)  # Prints all the words fit intoz the in the vectorizer
print("Feature Counts: ", len(vector_features), "\n\n")  # Prints the amount of words in the vectorizer
# Converts the vectorized data matrix to array
train_vec_arr = train_vectors.toarray()
# Puts the vectorized data into the dataframe
train_vec_dataframe = pd.DataFrame(data=train_vec_arr, columns=vector_features)

# One hot encodes discourse type
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_categorical_OneHot_train = pd.DataFrame(OH_encoder.fit_transform(train[['discourse_type']]))

def combineDataFrame(dfOg, restDfs):
    df = dfOg.copy()

    # drops the text column as it has been vectorized and type since it's been one hot encoded
    df.drop(['discourse_text', 'discourse_type'], inplace=True, axis=1)

    if 'discourse_effectiveness'in df:
        # ordinally encodes effectivness
        df['discourse_effectiveness'] = dfOg["discourse_effectiveness"].replace(
            {"Ineffective": 0, "Adequate": 1, "Effective": 2}
        )

    for curDf in restDfs:
        df = pd.concat([df, curDf], axis=1)

    return df


# Gets the combined and fully cleaned model
trainFullyCombined = combineDataFrame(train, [X_categorical_OneHot_train, train_vec_dataframe])

# Performs oversampling
ros = RandomOverSampler(sampling_strategy='auto', random_state=0)

processedY = trainFullyCombined['discourse_effectiveness']
trainFullyCombined.drop(['discourse_effectiveness'], axis=1, inplace=True)

xResampled, yResampled = ros.fit_resample(trainFullyCombined, processedY)

trainFullyProcessed = xResampled
trainFullyProcessed['discourse_effectiveness'] = yResampled

print("Categories After: ", train['discourse_effectiveness'].value_counts())

print(trainFullyProcessed.head())
trainFullyProcessed.to_csv('./Dataset/trainFullyProcessed.csv', index=False)


# Check if every letter is capitalized after every sentence
def capitalize(text):
    sentenceArr = [x for x in re.split(r"[\n\.\?\!]+", text) if len(x) > 0]
    i = 0
    upper_count = 0
    for sentence in sentenceArr:
        while i < len(sentence):
            if sentence[i].isalpha():
                if sentence[i].isupper():
                    upper_count = upper_count + 1
                else:
                    break
            i = i + 1
    return (upper_count / len(sentenceArr)) * 100


print(capitalize(text))
