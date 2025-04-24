
# Note: The purpose of the strange spacing in this file is to make the document
#       more readable once it is printed by keeping functions on the same page.

from os.path import exists
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from collections import Counter, defaultdict
import random
import re

MODEL_FILENAME = 'textgen_lstm.pth'
SEQUENCE_LENGTH = 5
TEXT_FILENAMES = {
    #'<|SCREWTAPE LETTERS|>': 'cslewis_screwtape_letters.txt',
    #'<|MERE CHRISTIANITY|>': 'cslewis_mere_christianity.txt',
    '<|THE LAST BATTLE|>': 'cslewis_last_battle.txt',
    '<|THE LION, THE WITCH, AND THE WARDROBE|>': 'cslewis_lion_witch_wardrobe.txt',
    '<|PRINCE CASPIAN|>': 'cslewis_prince_caspian.txt',
    '<|THE SILVER CHAIR|>': 'cslewis_silver_chair.txt',
    '<|THE MAGICIANS NEPHEW|>': 'cslewis_magicians_nephew.txt',
    '<|THE HORSE AND HIS BOY|>': 'cslewis_horse_and_his_boy.txt',
    '<|THE VOYAGE OF THE DAWN TREADER|>': 'cslewis_voyage_dawn_treader.txt'
}

class LSTMTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        out, hidden = self.lstm(embeds, hidden)
        out = self.fc(out[:, -1, :])  # take output from last LSTM cell
        return out, hidden

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")








def train_model(text_filenames):

    # Read in all of the training files as one giant string
    full_text = ""
    for tag, filepath in text_filenames.items():
        with open(filepath, 'r', encoding='utf-8') as f:
            raw = f.read() #.lower() makes it longer and more rare tokens, but nicer output

            # Split into paragraphs (using 2+ newlines)
            paragraphs = [p.strip() for p in raw.split('\n\n') if p.strip()]

            # Add <END> token at the end of each paragraph
            processed = [f"{tag} {p} <END>" for p in paragraphs]
            full_text += "\n".join(processed) + "\n"
    print(full_text[:100] + ' ... ' + full_text[-100:])

    # Tokenize the words and build two important data structures.
    #  1) word2idx: given a word, what is it's index in the vocabulary
    #  2) idx2word: given an index, what is the corresponding word

    # split on whitespace and punctuation '(\W)', and reinsert the punctuation '\1' as separate tokens
    # words = re.sub(r"(\W)", r" \1 ", word).split()
    #     words = re.sub(r"(\W)(?<![<>])", r" \1 ", word).split() doesn't include <>
    punctuation = "~!\"#$%&'()*+,-./:;=?@[\\]^_`{|}~" # except <>
    words = []
    word = []
    ignore_special_words = False

    # iterate through each character in the text and build words, which are added to the words list
    # make punctuation a separate token and ignore special phrases between <>
    for char in full_text:
        if char in punctuation and not ignore_special_words: # create a punctuation token
            if word:
                words.append(''.join(word))
                word = []
            words.append(char)
        elif char == '<':
            ignore_special_words = True
        elif char == '>':
            ignore_special_words = False
        elif (char == '\n' or char == ' ') and not ignore_special_words: # create a word token
            if word:
                words.append(''.join(word))
                word = []
        else:
            if not ignore_special_words:
                word.append(char)

    word_counts = Counter(words)
    vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    vocab_size = len(vocab)
    print(f'Vocabulary size: {vocab_size}')
    print(f'Lookup Tables: {len(word2idx)} and {len(idx2word)} elements')

    # Create sequences of N words, but use tokens instead of the actual words
    sequences = []
    for i in range(SEQUENCE_LENGTH, len(words)):
        seq = words[i-SEQUENCE_LENGTH:i+1]  # includes the label
        sequences.append([word2idx[word] for word in seq])
    print(f'There are {len(sequences)} word sequences of length {SEQUENCE_LENGTH}')
    print(f"{sequences[:10]} ... {sequences[-10:]}")
    print(" ".join([idx2word[idx] for idx in sequences[0]]))

    # We are training only, no testing
    random.shuffle(sequences)
    sequences = torch.tensor(sequences)
    X = sequences[:, :-1]
    y = sequences[:, -1]
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    model = LSTMTextGenerator(vocab_size, embed_dim=128, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 50
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs, _ = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    print("Training complete!")
    return (model, word2idx, idx2word, optimizer)



def save_model(model, word2idx, idx2word, optimizer, filename=MODEL_FILENAME):
    
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'word2idx': word2idx,
        'idx2word': idx2word,
    }, filename)

    print(f"Model saved as '{filename}'")



def load_model(filename=MODEL_FILENAME):

    checkpoint = torch.load(filename, map_location=device)
    model_state = checkpoint['model_state']
    optimizer = checkpoint['optimizer_state']
    word2idx = checkpoint['word2idx']
    idx2word = checkpoint['idx2word']

    model = LSTMTextGenerator(
        vocab_size=len(word2idx),
        embed_dim=128,
        hidden_dim=256
    ).to(device)

    model.load_state_dict(model_state)
    return (model, word2idx, idx2word, optimizer)



if exists(MODEL_FILENAME):
    print("Loading model")
    model, word2idx, idx2word, _ = load_model(MODEL_FILENAME)
else:
    print("Training model")
    model, word2idx, idx2word, optimizer = train_model(TEXT_FILENAMES)
    save_model(model, word2idx, idx2word, optimizer)


# Text generation, a sequence of words of a certain length
def generate_text(seed_text, next_words=30, temperature=1.0):
    model.eval()
    words = seed_text.split()
    state = None

    for _ in range(next_words):
        # Prepare input tensor
        input_seq = [word2idx.get(w, 0) for w in words[-SEQUENCE_LENGTH:]]
        input_tensor = torch.tensor(input_seq).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, state = model(input_tensor, state)

            # Apply temperature
            logits = logits.squeeze() / temperature
            probs = F.softmax(logits, dim=-1)

            # Sample from the adjusted distribution
            next_idx = torch.multinomial(probs, num_samples=1).item()

        next_word = idx2word.get(next_idx, "<unk>")
        if next_word == "<END>":
            break

        words.append(next_word)
    
    return ' '.join(words)


user_prompt = ''
while True:
    user_prompt = input('> ')
    if user_prompt.lower() in ['quit', 'exit']:
        break
    ai_response = generate_text(user_prompt, next_words=200)
    ai_response = ai_response.replace(user_prompt, '')
    print(ai_response)
