string = "<s> já karel a pepa </s>"
lst = string.split()
def bigrams_in_senetence(sentence):
  list_of_bigrams = []
  words = sentence.split()
  for idx in range(len(words)):
    if idx == 0:
      continue
    tmp = " ".join([words[idx-1],words[idx]])
    list_of_bigrams.append(tmp)
  
  return list_of_bigrams

def trigrams_in_senetence(sentence):
  list_of_trigrams = []
  words = sentence.split()
  for idx in range(len(words)):
    if idx <= 1:
      continue
    tmp = " ".join([words[idx-2],words[idx-1],words[idx]])
    list_of_trigrams.append(tmp)
  
  return list_of_trigrams

def ngrams_in_senetence(n,sentence):
  list_of_ngrams = []
  words = sentence.split()
  for idx in range(len(words)):
    if idx <= n-2:
      continue
    tmp = []
    for i in reversed(range(n)):
      tmp.append(words[idx-i])
    list_of_ngrams.append(" ".join(tmp))
  
  return list_of_ngrams


# print(ngrams_in_senetence(1,string))
# print(ngrams_in_senetence(2,string))
# print(bigrams_in_senetence(string))
# print(ngrams_in_senetence(3,string))
# print(trigrams_in_senetence(string))
# print(ngrams_in_senetence(4,string))

print(tuple(["a","b","b"]))

a = {"a":1,"b":2}

print(a)
a["a"] = a.get("a",0)+1
print(a)
a["c"] = a.get("c",0)+1
print(a)