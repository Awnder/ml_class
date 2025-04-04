from nltk.corpus import stopwords
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
        self.vocabulary = {}
        self.stopwords = set(stopwords.words("english")).union(set(extra_stopwords))
        self.vocabulary_size = vocabulary_size
        self.output_sequence_length = output_sequence_length

    def adapt(self, data: pd.Series) -> None:
        """Processes the input training data to build a vocabulary of unique words sorted by frequency.
        Args:
            data (pd.Series): A Pandas Series containing multiple rows of text data to be processed.
        """
        for text in data:
            tokens = self._tokenize(text)
            for i, word in enumerate(tokens):
                # remove stopwords like "the", "is", "and", etc.
                if word in self.stopwords:
                    continue

                self.vocabulary[word] = self.vocabulary.get(word, 0) + 1

        # Sort the vocabulary by frequency in descending order and limit to vocabulary_size
        sorted_vocabulary = sorted(self.vocabulary.items(), key=lambda item: item[1], reverse=True)
        self.vocabulary = dict(sorted_vocabulary[:self.vocabulary_size])

    def bag(self, data: list) -> list[list[int]]:
        """Transforms text into integer sequences.
        For each input text, breaks it into tokens and creates a binary sequence (bag of words).
        Adds a 1 to the sequence if the token is in the vocabulary, otherwise adds a 0.
        Pads the sequence with 0s to ensure a fixed length of `output_sequence_length`.
        Truncates the sequence if it exceeds `output_sequence_length`.
        Args:
            data (list): A list of input text strings to be processed.
        Returns:
            list[list[int]]: A list of binary sequences, where each sequence represents 
                the presence (1) or absence (0) of vocabulary words in the corresponding input text.
        """
        bag_of_words = []

        for text in data:
            bag = []
            tokens = self._tokenize(text)

            for word in tokens:
                if len(bag) >= self.output_sequence_length:
                    break
                
                if word in self.vocabulary:
                    # If the word is in the vocabulary, add 1 to the bag
                    bag.append(1)
                else:
                    bag.append(0)

            if len(bag) < self.output_sequence_length:
                # If the bag is shorter than output_sequence_length, pad with 0s
                bag.extend([0] * (self.output_sequence_length - len(bag)))

            bag_of_words.append(bag)

        return bag_of_words
    
    # def bag(self, data: str) -> list[int]:
    #     """Transforms a single text into an integer sequence.
    #     For the input text, breaks it into tokens and creates a binary sequence (bag of words).
    #     Adds a 1 to the sequence if the token is in the vocabulary, otherwise adds a 0.
    #     Pads the sequence with 0s to ensure a fixed length of `output_sequence_length`.
    #     Truncates the sequence if it exceeds `output_sequence_length`.
    #     Args:
    #         data (str): Input text string to be processed.
    #     Returns:
    #         list[int]: A binary sequence representing the presence (1) or absence (0) of vocabulary words in the input text.
    #     """
    #     bag = []
    #     tokens = self._tokenize(data)

    #     for word in tokens:
    #         if len(bag) >= self.output_sequence_length:
    #             break
            
    #         if word in self.vocabulary:
    #             # If the word is in the vocabulary, add 1 to the bag
    #             bag.append(1)
    #         else:
    #             bag.append(0)

    #     if len(bag) < self.output_sequence_length:
    #         # If the bag is shorter than output_sequence_length, pad with 0s
    #         bag.extend([0] * (self.output_sequence_length - len(bag)))

    #     return bag

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
