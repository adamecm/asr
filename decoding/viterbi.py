import copy
import numpy as np
import arpa

BETA = 5
LGAMMA = -np.log10(0.01)

char_idx = dict()
trans_probs = list()
wordlist = list()
with open("./leaves.txt","r",encoding="windows-1250") as file:
  for idx,line in enumerate(file.readlines()):
    splitline = line.split()
    char_idx[splitline[0]] = idx
    trans_probs.append((-np.log10(float(splitline[1])),-np.log10(float(splitline[2]))))
word_net = list()

with open("./vocab","r",encoding="windows-1250") as file:
  for line in file.readlines():
    temp = line.split("		")
    wordlist.append(temp[0])
    phonemes = temp[1]
    word_net.append([char_idx[x] for x in phonemes.split()])
  word_net.append([char_idx["#"]])
# print(word_net)
wordlist.append("#")
observations_B = list()
with open("./00170005_14.txt","r",encoding="windows-1250") as file:
  for line in file.readlines():
    observations_B.append([-np.log10(float(x)) for x in line.split()])


language_model = dict()
with open("./lm.txt","r",encoding="utf-8") as file:
  for line in file.readlines():
    temp = line.split()
    language_model[temp[1]] = float(temp[0])



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

tokens = list()
for t in range(1,len(observations_B)): #t = 2 to T (392), every observation vector
  old_phi_net = copy.deepcopy(phi_net)
  
  ends = [[(phi[-1][0] + trans_probs[word[-1]][1] - BETA*language_model.get(wordlist[idx],language_model["<unk>"])), phi[-1][1]] for phi,word,idx in zip(old_phi_net,word_net,range(len(word_net)))]
  # ends = list()
  # for i, word in enumerate(word_net):
  #   ends.append([old_phi_net[i][-1][0] + trans_probs[word[-1]][1] - BETA * language_model.get(wordlist[i], language_model["<unk>"]) , old_phi_net[i][-1][1]])
  #   # ends.append(old_phi_net[i][-1][0] + trans_probs)
  min_cost_val = min(ends)
  # min_cost_idx = np.argmin(ends,axis = 0)[0]
  min_cost_idx = ends.index(min_cost_val)

  tokens.append([wordlist[min_cost_idx],min_cost_val[1]])

  min_cost_val[1] = len(tokens) - 1 
  min_cost_val[0] += LGAMMA


  for w in range(len(phi_net)): #every word
    word = word_net[w]
    # min_cost = min(min_cost_val,[old_phi_net[w][0][0]+trans_probs[word[0]][0],old_phi_net[w][0][1]])
    phi_net[w][0] = min(min_cost_val,[old_phi_net[w][0][0]+trans_probs[word[0]][0],old_phi_net[w][0][1]])
    # phi_net[w][0] = [min_cost[0] + observations_B[t][word[0]],min_cost[1]]
    phi_net[w][0][0] += observations_B[t][word[0]]
    #1 to end of each word
    for j in range(1,len(word)):

      prev_phi = [old_phi_net[w][j-1][0] + trans_probs[word[j-1]][1],old_phi_net[w][j-1][1]] #prev + trans
      current_phi = [old_phi_net[w][j][0] + trans_probs[word[j]][0],old_phi_net[w][j][1]] #same + loop

      phi_net[w][j] = min(prev_phi,current_phi)
      phi_net[w][j][0] += observations_B[t][word[j]]




  if t == 1:
    print("t=2 a(a) = ", phi_net[0][0])
    # print("t=2 a(aby) = ", phi_net[1][0])
    print("t=2 # = ", phi_net[-1][0])
  if t==391:
    print("t=392 a(a) = ", phi_net[0][0])
    # print("t=392 a(aby) = ", phi_net[1][0])
    print("t=392 # = ", phi_net[-1][0])


# chars = {val:key for key,val in char_idx.items()}
# print(np.unique(tokens))

# for x in np.unique(tokens):
#   print(wordlist[x])

# ends = [(x[-1] + trans_probs[y[-1]][1]) for x,y in zip(phi_net,word_net)]
# ends = [(phi[-1][0] + trans_probs[word[-1]][1] + BETA*language_model.get(wordlist[idx],language_model["<unk>"])) for phi,word,idx in zip(old_phi_net,word_net,range(len(word_net)))]

# ends = [[(phi[-1][0] + trans_probs[word[-1]][1] - BETA*language_model.get(wordlist[idx],language_model["<unk>"])), phi[-1][1]] for phi,word,idx in zip(old_phi_net,word_net,range(len(word_net)))]
ends = [[(phi[-1][0] + trans_probs[word[-1]][1] - BETA*language_model.get(wordlist[idx],language_model["<unk>"])), phi[-1][1]] for phi,word,idx in zip(old_phi_net,word_net,range(len(word_net)))]
final_min_val = min(ends)
# # final_min_idx = np.argmin(ends,axis = 0)[0]
final_min_idx = ends.index(final_min_val)
# tokens.append([wordlist[min_cost_idx],min_cost_val[1]])
# print("Minimal cost = ",final_min_val[0])

final_min_val = min_cost_val
final_min_idx = min_cost_idx

i = final_min_val[1]
posloupnost = [wordlist[final_min_idx]]
while(i>0):
    posloupnost.append(tokens[i][0])
    i = tokens[i][1]
    print(i)
posloupnost.reverse()
print(posloupnost)

#beta = 5
# gamma = 0.01