import markovify


# Main function
def main():
    text = load_text('shakespeare.txt')
    speech_model = markovify.Text(text)

    # Printing 10 sentences
    for i in range(10):
        print(speech_model.make_sentence())
        print()


# Loading text
def load_text(file):
    with open(file) as f:
        contents = f.read()
        return contents


if __name__ == '__main__':
    main()
