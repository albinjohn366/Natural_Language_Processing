import nltk
import sys
import os
import string
from nltk.corpus import stopwords
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 3


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    dictionary = dict()
    for file in os.listdir(directory):
        with open(os.path.join(directory, file), encoding="utf8") as f:
            dictionary[file] = f.read()
    return dictionary


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    return [word.lower() for word in nltk.word_tokenize(document)
            if word not in string.punctuation and word not in
            stopwords.words('english')]


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """

    dictionary = dict()

    for document in documents:
        for word in documents[document]:

            # Calculate only if the word's IDF is not calculated
            if word not in dictionary:
                num_doc_con_word = 0

                # Checking the count of documents containing the word
                for document_1 in documents:
                    if word in documents[document_1]:
                        num_doc_con_word += 1

                constant = math.log(len(documents) / num_doc_con_word)
                dictionary[word] = constant

    return dictionary


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    order = []
    for file in files:
        value = 0
        for word in query:

            # checking if the word is in the file
            if word in files[file]:
                value += (idfs[word] * files[file].count(word))
        order.append((value, file))
    order = sorted(order, reverse=True)
    return_file = []
    for i in range(n):
        return_file.append(order[i][1])
    return return_file


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    order = []
    for sentence in sentences:
        tdf_sum = 0
        word_count = 0

        # For each word in the query
        for word in query:
            # If the word is in the sentence
            if word in sentences[sentence]:
                tdf_sum += idfs[word]
                word_count += 1

        order.append(
            (tdf_sum, (word_count / len(sentences[sentence])), sentence))

    order = sorted(order, reverse=True)
    return_file = []
    for i in range(n):
        return_file.append(order[i][2])
    return return_file


if __name__ == "__main__":
    main()
