import copy
import numpy as np

char_idx = dict()
trans_probs = list()

with open("./leaves.txt","r",encoding="windows-1250") as file:
  for idx,line in enumerate(file.readlines()):
    splitline = line.split()
    char_idx[splitline[0]] = idx
    trans_probs.append((-np.log10(float(splitline[1])),-np.log10(float(splitline[2]))))
word_net = list()

with open("./vocab","r",encoding="windows-1250") as file:
  for line in file.readlines():
    phonemes = line.split("		")[1]
    word_net.append([char_idx[x] for x in phonemes.split()])
  word_net.append([char_idx["#"]])
# print(word_net)

acoustic_model = list()
with open("./00170005_14.txt","r",encoding="windows-1250") as file:
  for line in file.readlines():
    acoustic_model.append([-np.log10(float(x)) for x in line.split()])

# a = []
# for i in range(38):
#   for j in range(38):
#     #print(i,j, acoustic_model[0][i]+trans_probs[i][0]+acoustic_model[1][j]+trans_probs[j][1])
#     a.append((i,j, acoustic_model[0][i]+trans_probs[i][0]+acoustic_model[1][j]+trans_probs[j][0]))

# print(sorted(a,key=lambda x: x[2]))
# exit()
def mins(best_path):
  cost = 0
  for state in best_path:
    cost += acoustic_model[0][state]+trans_probs[state][1]
#####init#####
phi_net = copy.deepcopy(word_net)
# min_path = np.infty
for w in range(len(phi_net)):
  phi_net[w][0] = acoustic_model[0][word_net[w][0]] #first row (time 0)
  #-np.log10(acoustic_model[0][phi_net[path_idx][0]])
  for phoneme_idx in range(1,len(phi_net[w])):
    phi_net[w][phoneme_idx] = np.infty

# old_phi_net = copy.deepcopy(phi_net)
min_path = np.argmin(acoustic_model[-1])
min_val = acoustic_model[0][min_path] + trans_probs[min_path][1]
##### calc #####
min_cost = []
t = 0
for t in range(1,len(acoustic_model)): #t = 2 to T (392), every vector from acoustic model
  # ends = []
  for w in range(len(phi_net)): #every word
    # if phi_net[w][-1] < min_path:
    #   min_path = w#phi_net[w][-1]

    word = word_net[w]
    for j in range(1,len(word)):
      # temp = phi_net[w][j]


      prev_char = word[j-1]
      current_char = word[j]

      prev_phi = phi_net[w][j-1] + trans_probs[prev_char][1] #prev + trans
      current_phi = phi_net[w][j] + trans_probs[current_char][0] #same + loop
     
      phi_net[w][j] = min(prev_phi,current_phi) + acoustic_model[t][current_char]
      # old_phi_net[w][j] = temp
      # phi_net[w][j] = phi_net[w][j]
    # ends.append(phi_net[w][-1])
  # ends = [x[-1] for x in phi_net]
  ends = [(x[-1] + trans_probs[y[-1]][1]) for x,y in zip(phi_net,word_net)]
  min_path = np.argmin(ends)
  min_state = word_net[min_path][-1]
  min_val = ends[min_path] + trans_probs[min_state][1]
  
  for w in range(len(phi_net)):
    # temp = phi_net[w][0]

    prev_char = word_net[min_path][-1]
    current_char = word_net[w][0]

    prev_phi = phi_net[min_path][-1] + trans_probs[prev_char][1]
    # prev_phi = min_val + trans_probs[prev_char][0]
    current_phi = phi_net[w][0] + trans_probs[current_char][0]
    phi_net[w][0] = min(prev_phi,current_phi) + acoustic_model[t][current_char]

    # old_phi_net[w][0] = temp



final_min_val = min([x[-1] for x in phi_net])
print(final_min_val)

print(min_val)

"""
list pravdepodobnosti

uz je vsechno jako -log10 => je treba pricitat a ne odcitat

min( phi_{j-1}(t-1)+A(j-1,j), phi_{j}(t-1)+A(j,j))


pres vsechna t (radek souboru)
  pres vsechna slova
    vypocet koncu (ty delsi kraty netreba pocitat)

  pres vsechna slova
    pres vsechny fonemy
      if zacatek
        koncova predchozi ppst
      else
        predchozi ppst

      vzorec ze souboru




for t in range(len(acoustic_model)):
  [print(i,x) for i,x in enumerate()]

"""
