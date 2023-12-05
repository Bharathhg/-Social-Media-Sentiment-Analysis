import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')

tokenizer = RegexpTokenizer(r"\w+")
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()

def getCleanedText(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenizing
    tokens = tokenizer.tokenize(text)

    # Remove stopwords and apply stemming
    new_tokens = [token for token in tokens if token not in en_stopwords]
    stemmed_tokens = [ps.stem(token) for token in new_tokens]

    # Join the cleaned tokens to form the cleaned text
    clean_text = " ".join(stemmed_tokens)

    return clean_text

def train_model(X_train, y_train):
    # Preprocess the data
    X_clean = [getCleanedText(text) for text in X_train]

    # Vectorize the data
    cv = CountVectorizer(ngram_range=(1, 2))
    X_vec = cv.fit_transform(X_clean).toarray()

    # Train the model
    mn = MultinomialNB()
    mn.fit(X_vec, y_train)

    return mn, cv

def predict_with_ml_model(model, cv, text):
    # Clean the input text
    cleaned_text = getCleanedText(text)

    # Vectorize the cleaned text
    text_vectorized = cv.transform([cleaned_text]).toarray()

    # Predict sentiment
    prediction = model.predict(text_vectorized)

    return prediction[0]  # Assuming a single prediction for a single input

if __name__ == '__main__':
    # Sample data
    X_train = ["This was really awesome an awesome movie",
               "Great movie! I liked it a lot",
               "Happy Ending! Awesome Acting by hero",
               "loved it!",
               "Bad not upto the mark",
               "Could have been better",
               "really Dissapointed by the movie"]

    y_train = ["positive", "positive", "positive", "positive", "negative", "negative", "negative"]

    # Train the model
    model, cv = train_model(X_train, y_train)

    # Example of using the trained model for prediction
    X_test = ["it was bad"]
    Xt_clean = [getCleanedText(text) for text in X_test]
    Xt_vec = cv.transform(Xt_clean).toarray()
    y_pred = model.predict(Xt_vec)

    print("Predicted class:", y_pred[0])
