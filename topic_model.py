import os
import nltk
import math


# Main function
def main():
    corpus = load_data('federalist')
    topic_modelling(corpus)


# Finding relevant topics by ordering on the basis of frequency
def topic_modelling(folder):
    for num, file in enumerate(folder):
        print(num, ': ', file)
        print('The most common five key words are: ')
        key_words = find_key_words(file, folder)
        for i in key_words:
            print('\t', i)
        print()


# Finding key words
def find_key_words(file, folder):
    word_order = []
    for word in folder[file]:

        # Finding number of documents containing the word
        no_doc_con_word = 0
        for file_1 in folder:
            if word in folder[file_1]:
                no_doc_con_word += 1

        # Calculating the value for ordering
        constant = math.log(len(folder) / no_doc_con_word)
        word_order.append((folder[file][word] * constant, word))
    word_order = sorted(word_order, reverse=True)[:5]
    return [item[1] for item in word_order]


# Getting thw whole file and frequency
def load_data(directory):
    # finding words to ignore
    with open('function-words.txt') as f:
        ignore_words = [word.lower() for word in nltk.word_tokenize(f.read())
                        if word.isalpha()]

    folder = dict()
    for file in os.listdir(directory):
        with open(os.path.join(directory, file)) as f:
            # Finding words present in a file
            words = [word.lower() for word in
                     nltk.word_tokenize(f.read())
                     if word.isalpha()]

        # Getting the frequency
        frequency = dict()
        for word in words:
            if word in ignore_words:
                continue
            frequency[word] = 1 if (word not in frequency) else (frequency[word]
                                                                 + 1)
        folder[file] = frequency
    return folder


if __name__ == '__main__':
    main()
