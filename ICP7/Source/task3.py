import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

sentence = open('input.txt', encoding="utf8").read()

# Tokenization
stokens = nltk.sent_tokenize(sentence)
wtokens = nltk.word_tokenize(sentence)

print("\n============== Word  Tokenization ==============\n")
print(wtokens)
print("\n============== Sentence  Tokenization ==============\n")
print(stokens)


#streeming
print("\n============== Stremming ==============\n")
stemmed_output = ' '.join([ps.stem(w) for w in w_tokens])
print("-------Stemming")
print(stemmed_output)
print("\n")


# POS

print("\n============== POS ==============\n")
n_pos = nltk.pos_tag(w_tokens)
print("Parts of Speech :", n_pos)

# Lemmatization
print("\n============== Lemmatization ==============\n")
lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in w_tokens])
print("----------Lemmatization")
print(lemmatized_output)
print("\n")

# Trigram
print("\n============== Trigram ==============\n")
from nltk.util import ngrams
token = nltk.word_tokenize(sentence)

n = 0
for s in stokens:
    n = n + 1
    if n < 2:
        token = nltk.word_tokenize(s)
        bigrams = list(ngrams(token, 2))
        trigrams = list(ngrams(token, 3))
        print("The text:", s, "\nword_tokenize:", token, "\nbigrams:", bigrams, "\ntrigrams", trigrams)

print("\n============== Named Entity Recognition ==============\n")
# Named Entity Recognition
from nltk import word_tokenize, pos_tag, ne_chunk
n = 0
for s in stokens:
    n = n + 1
    if n < 2:
        print(ne_chunk(pos_tag(word_tokenize(s))))