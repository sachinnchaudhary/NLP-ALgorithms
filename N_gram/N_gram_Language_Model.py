import re

from collections import Counter


def tokenize(corpus):

    return re.findall(r"\b\w+\b", corpus.lower())


         
def unigram(word, corpus): 

      tokens = tokenize(corpus)
      word_count = Counter(tokens)
      len = __builtins__.len
      total_words = len(tokens)
      return word_count[word] / total_words 


def bigram(corpus, prev_word, word):

     tokens = tokenize(corpus)
     bigrams = list(zip(tokens[:-1], tokens[1:]))
     bigram_count = Counter(bigram)
     prev_word_count = tokens.count(prev_word.lower())
     return bigram_count[(prev_word.lower(), word.lower())] / prev_word_count if prev_word_count else 0

def trigram(corpus, word1, word2, word3):
     tokens = tokenize(corpus)
     trigrams = list(zip(tokens[:-2], tokens[1:-1], tokens[2:]))
     count_trigram = Counter(trigram)
     prev_bigram_count = Counter(list(zip(tokens[:-1], tokens[1:])))
     bigram_key = prev_bigram_count[(word1.lower(), word2.lower())]

     return count_trigram[(word1.lower(), word2.lower(), word3.lower())] / bigram_key if bigram_key else 0

def fourgram(corpus, word1, word2, word3, word4):
     tokens  = tokenize(corpus)
     fourgrams = list(zip(tokens[:-3], tokens[1:-2], tokens[2:-1], tokens[3:]))
     count_fourgram = Counter(fourgram)
     prev_trigram_count = Counter(list(zip(tokens[:-2], tokens[1:-1], tokens[2:])))
     trigram_key = prev_trigram_count[(word1.lower(), word2.lower(), word3.lower())]

     return count_fourgram[(word1.lower(), word2.lower(), word3.lower(), word4.lower())] / trigram_key if trigram_key else 0

corpus = "Prof. Emily Clark is a computational linguist at Google Brain, where she’s leading the Language Understanding group since June 2019. Her team's recent breakthroughs include Transformer-based models like BERT, RoBERTa, and GPT-3. She has authored 48 peer-reviewed papers in conferences such as NAACL, AAAI, IJCAI, and ACL. She completed her MSc in Computational Linguistics at MIT in 2015 and PhD at UC Berkeley in 2018. Reach her at emily.clark@google.com, follow her on Twitter (@EmilyNLP), or visit her page emilyclark.ai for detailed publications."


print("choose n-gram model: \n1. unigram\n2. bigram\n3. trigram\n4. fourgram ")
choice = int(input("select: 1 , 2, 3 ,4: "))


if choice == 1:
  word = input("enter the word for unigram prob: ")
  prob = unigram(word, corpus)
  print(f"this is the prob for word {prob}")


elif choice == 2:
    word1 = input("enter the previous word for bigram prob: ")
    word2 = input("enter the target word for bigram prob: ")
    prob = bigram(corpus, word1, word2)
    print(f"bigram probability {word1}: {prob: .4f}")


elif choice == 3:
    word1 = input("enter the first word for probability: ")
    word2 = input("enter the second word for probability: ")
    word3 = input("enter the target word for probability: ")   
    
    prob = trigram(corpus, word1, word2, word3)
    print(f"trigram probability for {word3}: {prob: .4f}")

elif choice == 4:
  word1 = input("enter the first word for probability: ")
  word2 = input("enter the second word for probability: ")
  word3 = input("enter the third word for probability: ")
  word4 = input("enter the target word for probability: ")

  prob = fourgram(corpus, word1, word2 ,word3, word4)
  print(f'fourgram prob: {word4}:  {prob: .4f}')



else:
   print("plz choice 1,2,3")
