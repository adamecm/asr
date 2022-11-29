import numpy as np

path1 = "./trigram_model/train.txt"
path2 = "./trigram_model/cestina"

def count_unigrams(txt_source,count_dict):
  for line in txt_source:
    for word in line.split():
      if word == "<s>":
        continue
      if word not in LEGAL_WORDS:
        word = "<unk>"
      if word not in count_dict.keys():
        count_dict[word] = 1  
      else:
        count_dict[word] += 1

def ngrams_in_senetence(n,sentence):
  list_of_ngrams = []
  words = sentence.split()
  words = sentence.split()
  for idx in range(len(words)):
    if words[idx] not in LEGAL_WORDS:
      words[idx] = "<unk>"
    if idx <= n-2:
      continue
    tmp = []
    for i in reversed(range(n)):
      tmp.append(words[idx-i])
    list_of_ngrams.append(tuple(tmp))
  
  return list_of_ngrams

# def bigrams_in_senetence(sentence):
#   list_of_bigrams = []
#   words = sentence.split()
#   for idx in range(len(words)):
#     if idx == 0:
#       continue
#     tmp = " ".join([words[idx-1],words[idx]])
#     list_of_bigrams.append(tmp)
  
#   return list_of_bigrams

# def trigrams_in_senetence(sentence):
  list_of_trigrams = []
  words = sentence.split()
  for idx in range(len(words)):
    if idx <= 1:
      continue
    tmp = " ".join([words[idx-2],words[idx-1],words[idx]])
    list_of_trigrams.append(tmp)
  
  return list_of_trigrams


def prune_ngrams(count_dict):
  keys = list(count_dict.keys())
  for key in keys:
    if count_dict[key] < 2:
      count_dict.pop(key)
  return count_dict

def count_bigrams(txt_source,count_dict):
  for line in txt_source:
    bigrams = ngrams_in_senetence(2,line)
    for bigram in bigrams:
      if bigram not in count_dict.keys():
        count_dict[bigram] = 1  
      else:
        count_dict[bigram] += 1


def count_trigrams(txt_source,count_dict):
  for line in txt_source:
    bigrams = ngrams_in_senetence(3,line)
    for trigram in bigrams:
      if trigram not in count_dict.keys():
        count_dict[trigram] = 1  
      else:
        count_dict[trigram] += 1


LEGAL_WORDS = set()
with open(path2,"r",encoding="windows-1250") as f:
  for line in f.readlines():
    LEGAL_WORDS.add(line.strip("\n"))
LEGAL_WORDS.add("</s>")
LEGAL_WORDS.add("<s>")

source_text = list()
unigram_count = dict()
bigram_count = dict()
trigram_count = dict()
with open(path1,"r",encoding="windows-1250") as f:
  for line in f.readlines():
    line = "<s> " + line.strip("\n") + " </s>"
    source_text.append(line)



print("########################UNIGRAMY########################")
## unigrams
count_unigrams(source_text,unigram_count)
# print("unk",unigram_count["<unk>"])
number_of_words = sum(unigram_count.values())
print("nou",number_of_words)



number_of_unigrams = len(unigram_count)
print("nou",number_of_unigrams)



unigram_ML = unigram_count.copy()

for word in unigram_count.keys():
  unigram_ML[word] = np.log10(unigram_count[word]/number_of_words)

print("uni a :",unigram_ML["a"])
print("uni <unk> :", unigram_ML["<unk>"])
print("uni </s> :",unigram_ML["</s>"])
print()
print("########################BIGRAMY########################")
##bigrams
count_bigrams(source_text,bigram_count)
number_of_bigrams = len(bigram_count)
print("nob",number_of_bigrams)

# print(bigram_count)
bigram_ML = bigram_count.copy()
# exit()
# for bigram in bigram_count.keys():
#   temp = 0
#   for word in bigram.split()[:-1]:
#     if word == "<s>":
#       temp = 2350
#       continue
#     temp += unigram_count[word]
#   bigram_ML[bigram] = np.log10(bigram_count[bigram]/temp)

for bigram in bigram_count.keys():
  word = bigram[0]
  if word == "<s>":
    denom = 2350
  else:
    denom = unigram_count[word]
  bigram_ML[bigram] = np.log10(bigram_count[bigram]/denom)

#     if word == "<s>":
#       temp = 2350
#       continue
#     temp += unigram_count[word]

print("bi <s> <unk> :", bigram_ML[("<s>", "<unk>")])
print("bi a o :", bigram_ML[("a", "o")])
print()
print("########################TRIGRAMY########################")
##trigrams
count_trigrams(source_text,trigram_count)
number_of_trigrams = len(trigram_count)
print("not",number_of_trigrams)
trigram_count = prune_ngrams(trigram_count)
number_of_trigrams = len(trigram_count)
print("not pruned",number_of_trigrams)
# print(trigram_count)
# print(bigram_count)
trigram_ML = trigram_count.copy()
# exit()

# for trigram in trigram_count.keys():
#   temp = 0
#   for word in trigram.split()[:-1]:
#     if word == "<s>":
#       temp = 2350
#       continue
#     temp += unigram_count[word]
#   bigram_ML[bigram] = np.log10(bigram_count[bigram]/temp)

for trigram in trigram_count.keys():
  wordset = tuple(trigram[0:2])
  # if word == "<s>":
  #   denom = 2350
  #   continue
  denom = bigram_count[wordset]
  trigram_ML[trigram] = np.log10(trigram_count[trigram]/denom)

# for word in trigram_count.keys():
#   trigram_ML[word] = np.log10(trigram_count[word]/number_of_trigrams)

print("tri <s> <unk> <unk> :", trigram_ML[("<s>", "<unk>", "<unk>")])
print("tri a tak se :", trigram_ML[("a", "tak", "se")])
print("tri a za této :", trigram_ML[("a", "za", "této")])
# print("bi a o :", trigram_ML["a o"])
