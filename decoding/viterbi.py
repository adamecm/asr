import copy
import numpy as np
import re

LEAVES_PATH = "./leaves.txt"
VOCAB_PATH = "./vocab"
ARPA_PATH = "./lm_arpa.txt"
OBSERVATIONS_PATH = "./00170005_14.txt"

LEAVES_ENCODING = "windows-1250"
VOCAB_ENCODING = "windows-1250"
ARPA_ENCODING = "utf-8"
OBSERVATIONS_ENCODING = "windows-1250"

BETA = 5
LGAMMA = -np.log10(0.01)

char_idx = dict()
trans_probs = list()
wordlist = list()
with open(LEAVES_PATH,"r",encoding=LEAVES_ENCODING) as file:
  for idx,line in enumerate(file.readlines()):
    splitline = line.split()
    char_idx[splitline[0]] = idx
    trans_probs.append((-np.log10(float(splitline[1])),-np.log10(float(splitline[2]))))
word_net = list()

with open(VOCAB_PATH,"r",encoding=VOCAB_ENCODING) as file:
  for line in file.readlines():
    temp = line.split("		")
    wordlist.append(temp[0])
    phonemes = temp[1]
    word_net.append([char_idx[x] for x in phonemes.split()])
  word_net.append([char_idx["#"]])
# print(word_net)
wordlist.append("#")
observations_B = list()

with open(OBSERVATIONS_PATH,"r",encoding=OBSERVATIONS_ENCODING) as file:
  for line in file.readlines():
    observations_B.append([-np.log10(float(x)) for x in line.split()])


language_model = dict()
with open(ARPA_PATH,"r",encoding=ARPA_ENCODING) as file:
  lm = "".join(file.readlines()) #this should be file.read() instead but that does not work
  # chunks = lm.split("\n\n")
  # all_unigrams = re.findall("\d+[.]\d+ [\w|<]+\n",lm)
  all_unigrams = re.findall("-?\d+.\d+ \S+\n",lm)
  chunk_dict = dict()
  for element in all_unigrams:
    tmp = element.strip("\n").split()
    chunk_dict[tmp[1]] = float(tmp[0])
  language_model["unigrams"] = chunk_dict

  all_bigrams = re.findall("-?\d+.\d+ \S+ \S+\n",lm)
  chunk_dict = dict()
  for element in all_bigrams:
    tmp = element.strip("\n").split()
    chunk_dict[(tmp[1],tmp[2])] = float(tmp[0])
  language_model["bigrams"] = chunk_dict
  

  all_trigrams = re.findall("-?\d+.\d+ \S+ \S+ \S+\n",lm)
  chunk_dict = dict()
  for element in all_trigrams:
    tmp = element.strip("\n").split()
    chunk_dict[(tmp[1],tmp[2],tmp[3])] = float(tmp[0])
  language_model["trigrams"] = chunk_dict

unigrams = language_model["unigrams"]

unigrams = dict()
with open("./lm.txt","r",encoding=ARPA_ENCODING) as file:
  for line in all_unigrams:
    tmp = line.split()
    unigrams[tmp[1]] = float(tmp[0])
  
  

#####init#####
phi_net = copy.deepcopy(word_net)
# min_path = np.infty
for w in range(len(phi_net)):
  phi_net[w][0] = [observations_B[0][word_net[w][0]],0] #first row (time 0), token number 0
  #-np.log10(acoustic_model[0][phi_net[path_idx][0]])
  for phoneme_idx in range(1,len(phi_net[w])):
    phi_net[w][phoneme_idx] = [np.infty,0]



print("t=1 a(a) = ", phi_net[0][0])
print("t=1 a(aby) = ", phi_net[1][0])
print("t=1 # = ", phi_net[-1][0])
print()
tokens = list()
for t in range(1,len(observations_B)): #t = 2 to T (392), every observation vector
  old_phi_net = copy.deepcopy(phi_net)
  
  # ends = [[(phi[-1][0] + trans_probs[word[-1]][1] - BETA*language_model.get(wordlist[idx],language_model["<unk>"])), phi[-1][1]] for phi,word,idx in zip(old_phi_net,word_net,range(len(word_net)))]
  ends = [[(phi[-1][0] + trans_probs[word[-1]][1] + LGAMMA - BETA*unigrams.get(wordlist[idx],0)), phi[-1][1]] for phi,word,idx in zip(old_phi_net,word_net,range(len(word_net)))]
  # ends = [[(phi[-1][0] + trans_probs[word[-1]][1] - BETA*language_model.get(wordlist[idx],0 if wordlist[idx]=="#" else language_model["<unk>"])), phi[-1][1]] for phi,word,idx in zip(old_phi_net,word_net,range(len(word_net)))]
  # ends = [[(phi[-1][0] + trans_probs[word[-1]][1]), phi[-1][1]] for phi,word,idx in zip(old_phi_net,word_net,range(len(word_net)))]
  min_cost_val = min(ends)
  min_cost_idx = ends.index(min_cost_val)
  tokens.append((wordlist[min_cost_idx],min_cost_val[1]))
  min_cost_val[1] = len(tokens) - 1 
  # min_cost_val[0] = min_cost_val[0] + LGAMMA - BETA*language_model.get(wordlist[idx],0 if wordlist[idx]=="#" else language_model["<unk>"])
  # min_cost_val[0] = min_cost_val[0] + LGAMMA
  
  for w in range(len(phi_net)): #every word
    word = word_net[w]
    cost = min(min_cost_val,[old_phi_net[w][0][0]+trans_probs[word[0]][0], old_phi_net[w][0][1]])
    phi_net[w][0] = [cost[0] + observations_B[t][word[0]], cost[1]]
    #1 to end of each word
    for j in range(1,len(word)):
      prev_phi = [old_phi_net[w][j-1][0] + trans_probs[word[j-1]][1],old_phi_net[w][j-1][1]] #prev + trans
      current_phi = [old_phi_net[w][j][0] + trans_probs[word[j]][0],old_phi_net[w][j][1]] #same + loop
      cost = min(prev_phi,current_phi)
      phi_net[w][j] = [cost[0] + observations_B[t][word[j]], cost[1]]
  
  print(f"END MIN COST:", min_cost_val)
  print(f"TIME {t+1}: a(a) = ", phi_net[0])
  print(f"TIME {t+1} a(aby) = ", phi_net[1])
  print(f"TIME {t+1} # = ", phi_net[-1])
  print()

  
  

# ends = [[(phi[-1][0] + trans_probs[word[-1]][1] - BETA*language_model.get(wordlist[idx],language_model["<unk>"])), phi[-1][1]] for phi,word,idx in zip(old_phi_net,word_net,range(len(word_net)))]
# ends = [[(phi[-1][0] + trans_probs[word[-1]][1] + LGAMMA - BETA*language_model.get(wordlist[idx],0)), phi[-1][1]] for phi,word,idx in zip(old_phi_net,word_net,range(len(word_net)))]

ends = [[(phi[-1][0] + trans_probs[word[-1]][1] + LGAMMA - BETA*unigrams.get(wordlist[idx],0)), phi[-1][1]] for phi,word,idx in zip(old_phi_net,word_net,range(len(word_net)))]
final_min_val = min(ends)
final_min_idx = ends.index(final_min_val)
# tokens.append([wordlist[min_cost_idx],min_cost_val[1]])
print("Minimal cost = ",final_min_val[0])

previous_token_index = final_min_val[1]
spoken_words = [wordlist[final_min_idx]]
while(previous_token_index != 0):
    spoken_words.append(tokens[previous_token_index][0])
    previous_token_index = tokens[previous_token_index][1]
    # print(previous_token)
spoken_words.reverse()
print("Result:")
print(spoken_words)

#beta = 5
# gamma = 0.01