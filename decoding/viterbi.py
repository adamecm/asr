import copy
import numpy as np


beta = 5
lgamma = -np.log10(0.01)
char_idx = dict()
trans_probs = list()
wordlist = list()
with open("./leaves.txt","r",encoding="windows-1250") as file:
  for idx,line in enumerate(file.readlines()):
    splitline = line.split()
    char_idx[splitline[0]] = idx
    trans_probs.append((-np.log10(float(splitline[1])),-np.log10(float(splitline[2]))))
word_net = list()
wordlist.append("#")
with open("./vocab","r",encoding="windows-1250") as file:
  for line in file.readlines():
    temp = line.split("		")
    wordlist.append(temp[0])
    phonemes = temp[1]
    word_net.append([char_idx[x] for x in phonemes.split()])
  word_net.append([char_idx["#"]])
# print(word_net)

acoustic_model = list()
with open("./00170005_14.txt","r",encoding="windows-1250") as file:
  for line in file.readlines():
    acoustic_model.append([-np.log10(float(x)) for x in line.split()])

#####init#####
phi_net = copy.deepcopy(word_net)
# min_path = np.infty
for w in range(len(phi_net)):
  phi_net[w][0] = acoustic_model[0][word_net[w][0]] #first row (time 0)
  #-np.log10(acoustic_model[0][phi_net[path_idx][0]])
  for phoneme_idx in range(1,len(phi_net[w])):
    phi_net[w][phoneme_idx] = np.infty



print("t=1 a(a) = ", phi_net[0][0])
print("t=1 a(aby) = ", phi_net[1][0])
print("t=1 # = ", phi_net[-1][0]) 

token_net = [np.zeros(len(x)) for x in phi_net]
tokens = list()
for t in range(1,len(acoustic_model)): #t = 2 to T (392), every vector from acoustic model
  old_phi_net = copy.deepcopy(phi_net)
  old_token_net = copy.deepcopy(token_net)

  ends = [(x[-1] + trans_probs[y[-1]][1]) for x,y in zip(old_phi_net,word_net)]
  min_val = min(ends)
  min_word = np.argmin(ends)

  tokens.append(min_word)
    

  for w in range(len(phi_net)): #every word
    word = word_net[w]
    phi_net[w][0] = min(min_val,old_phi_net[w][0]+trans_probs[word[0]][0])+acoustic_model[t][word[0]]
    #1 to end of each word
    for j in range(1,len(word)):

      prev_phi = old_phi_net[w][j-1] + trans_probs[word[j-1]][1] #prev + trans
      current_phi = old_phi_net[w][j] + trans_probs[word[j]][0] #same + loop
    
      phi_net[w][j] = min(prev_phi,current_phi) + acoustic_model[t][word[j]]
  
    


  if t == 1:
    print("t=2 a(a) = ", phi_net[0][0])
    # print("t=2 a(aby) = ", phi_net[1][0])
    print("t=2 # = ", phi_net[-1][0])
  if t==391:
    print("t=392 a(a) = ", phi_net[0][0])
    # print("t=392 a(aby) = ", phi_net[1][0])
    print("t=392 # = ", phi_net[-1][0])


chars = {val:key for key,val in char_idx.items()}
print(np.unique(tokens))

for x in np.unique(tokens):
  print(wordlist[x])

ends = [(x[-1] + trans_probs[y[-1]][1]) for x,y in zip(phi_net,word_net)]
final_min_val = min(ends)
print("Minimal cost = ",final_min_val)


#beta = 5
# gamma = 0.01