import nltk
import os
from collections import Counter


# The main function
def main():
    data = load_data('holmes')

    # Finding the ngrams
    grams = [('unigrams', Counter(nltk.ngrams(data, 1))),
             ('bigrams', Counter(nltk.ngrams(data, 2))),
             ('trigrams', Counter(nltk.ngrams(data, 3)))]

    # Printing information
    for name, items in grams:
        print('The number of {} are: {}'.format(name, sum(items.values())))
        print('The most common are: ')
        for word, num in items.most_common(10):
            print(word, '\t', num)
        print()
        print()


# Load the contents into required variable
def load_data(directory):
    contents = []
    for file in os.listdir(directory):
        with open(os.path.join(directory, file)) as f:
            contents.extend(
                [word.lower() for word in nltk.word_tokenize(f.read())
                 if any(c.isalpha() for c in word)])
    return contents


if __name__ == '__main__':
    main()
