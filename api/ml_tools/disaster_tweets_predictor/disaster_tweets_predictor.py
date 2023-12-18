import pickle
import numpy as np
import os
import re


utilities_path = os.path.join(os.getcwd(), "api", "ml_tools", "utilities", "disaster_tweets_predictor")

## creating transformer
class TweetTransformer():
    def __init__(self, word2idx, vectors):
        self.word2idx = word2idx
        self.vectors = vectors

    ## function to preprocessing of text
    def process_text(self, text):
        ## patterns to remove
        rem_pat_1 = "([@]|https?:)\S*"
        rem_pat_2 = "&\S+;"
        rem_pat_3 = "\[\d+:\d+.+\]" ## removing timestamp. eg. [01:04 UTC]
        rem_pat_4 = "[\-_.+#]" ## to remove symbols (make sure to bring last to avoid affecting first two patterns)
        combined_rem_pat = f"({rem_pat_1})|({rem_pat_2})|({rem_pat_3})|({rem_pat_4})"
    
        text = re.sub(combined_rem_pat, "", text) ## removing text that match patterns
        text = text.strip() ## removing trailing white spaces
        text = text.lower() ## lowercasing
    
        return text
    
    
    def full_text_process(self, text):
        text = self.process_text(text)
        return text


    def transform(self, X):
        
        if isinstance(X, str):
            X = [self.full_text_process(X)]
            
        N = len(X)
        X = [self.full_text_process(x) for x in X]
            
        transformed_x = np.zeros((N, self.vectors.shape[1]), dtype=np.float32)
        for i in range(N):
            mat = []
            line = X[i].lower().split()
            for word in line:
                if word in self.word2idx:
                    mat.append(self.vectors[self.word2idx[word]])

            if len(mat) > 0: transformed_x[i] = np.mean(mat, axis=0)
            else: print(f"Sentence at index:{i} has no word in the vector dictionary")

        return np.array(transformed_x)


def predict(text):
    ## creating transformer
    transformer = None
    word_vector_dict = None

    with open(os.path.join(utilities_path, "disaster_tweets_word_vectors.pickle"), "rb") as file:
        word_vector_dict = pickle.load(file)

    transformer = TweetTransformer(**word_vector_dict)

    ## loading classifier
    with open(os.path.join(utilities_path, "disaster_tweets_classifier.pickle"), "rb") as file:
        model = pickle.load(file)
        
    result = {}
    prediction = model.predict(transformer.transform(text))
    result["prediction"] = int(prediction[0])
    result["probability"] = float(np.max(model.predict_proba(transformer.transform(text)), 1)[0])

    return result
