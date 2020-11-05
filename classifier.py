import nltk


# Main function
def main():
    positives, negatives = load_data('Positive_reviews.txt',
                                     'Negative_reviews.txt')

    # Getting all the words and positives and negatives
    words = set()
    for word in positives:
        words.update(word)
    for word in negatives:
        words.update(word)

    # Getting the training set
    training_set = []
    training_set.extend(get_feature_set(positives, words, 'positive'))
    training_set.extend(get_feature_set(negatives, words, 'negative'))

    # Providing the data set for training
    classifier = nltk.NaiveBayesClassifier.train(training_set)

    # Classifying sentence on the trained model
    sentence = input('Please enter the sentence you want to check\n')
    sentence_words = extract_words(sentence)
    coordinate = format_change(words, sentence_words)
    probabilty = classifier.prob_classify(coordinate)
    for i in probabilty.samples():
        print(i, ': ', probabilty.prob(i))


# changing the input sentence into the required format
def format_change(words, sentence_words):
    return {word: (word in sentence_words) for word in words}


# Getting feature for training
def get_feature_set(data, words, type):
    feature = []
    for group in data:
        feature.append((
            {word: (word in group) for word in words},
            type
        ))
    return feature


# Loading data
def load_data(positive, negative):
    data = [positive, negative]
    result = []
    for file in data:
        with open(file) as f:
            result.append([extract_words(line)
                           for line in f.read().splitlines()])
    return result


# Extracting words
def extract_words(line):
    return set(word.lower() for word in nltk.word_tokenize(line)
               if any(c.isalpha() for c in word))


if __name__ == '__main__':
    main()
