from nltk.corpus import stopwords
from collections import defaultdict
import pandas as pd
import string
import re

class BagOfWords:
    def __init__(self, vocabulary_size: int = 1000, output_sequence_length: int = 200, extra_stopwords: list = []):
        """Initializes the BagOfWords object
        Args:
            max_tokens (int): Maximum vocabulary size
            extra_stopwords (list): Additional stopwords to exclude from the vocabulary
        """
        self.vocabulary = defaultdict(int)
        self.stopwords = set(stopwords.words("english")).union(set(extra_stopwords))
        self.vocabulary_size = vocabulary_size
        self.output_sequence_length = output_sequence_length

    def adapt(self, data: pd.Series) -> None:
        """Finds unique words in the training data and stores them by most frequent
        Args:
            data (Pandas Series): multiple rows of text data to be processed
        Returns:
            dict: A dictionary with words as keys and their frequencies as values
        """
        for text in data:
            tokens = self._tokenize(text)
            for i, word in enumerate(tokens):
                # remove stopwords like "the", "is", "and", etc.
                if word in self.stopwords:
                    continue

                self.vocabulary[word] += 1

        self.vocabulary = self.vocabulary.sorted(key=lambda item: item[1], reverse=True)[:self.vocabulary_size]
        self.vocabulary = self.vocabulary.insert(0, '<unk>')  # add unknown token at index 1
        self.vocabulary = self.vocabulary.insert(0, '')  # add padding token at index 0
        print(self.vocabulary[:10])

    def bag(self, data: str) -> dict:
        """Transforms text into integer sequences
        Breaks text into tokens and maps them to their corresponding index values in the vocabulary.
        Pads the sequences to a fixed length of output_sequence_length. 
        Breaks automatically if the length exceeds output_sequence_length.
        Args:
            data (str): input text to be processed
        Returns:
            list[int]: A list of list of integers representing the bag of words
        """
        tokens = self._tokenize(data)
        bag = []
        for word in tokens:
            if len(bag) >= self.output_sequence_length:
                break
            if word in self.vocabulary and self.vocabulary[word] > 0:
                bag.append(self.vocabulary.index(word))
            else:
                bag.append(self.vocabulary.index('<unk>'))
        return bag

    def empty(self) -> None:
        """Resets the bag of words to an empty state"""
        self.vocabulary.clear()

    def _tokenize(self, text: str) -> list:
        """Tokenizes the input text and removes unwanted characters
        Args:
            text (str): input text to be tokenized
        Returns:
            list: A list of tokens (words) after processing
        """
        text = text.replace("\n", " ").replace("\r", " ")  # remove newlines
        text = text.translate(
            str.maketrans("", "", string.punctuation)
        )  # remove punctuation
        text = re.compile(r"<[^>]+>").sub("", text)  # remove HTML tags
        text = text.lower()  # convert to lowercase
        tokens = text.split()
        return tokens


if __name__ == "__main__":
    BOW = BagOfWords()
    print(BOW.vocabulary)  # Display the bag of words
