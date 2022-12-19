import copy
import numpy as np
# net = list()
# with open("./vocab","r",encoding="windows-1250") as file:
#   for line in file.readlines():
#     phonemes = line.split("		")[1]
#     net.append([x for x in phonemes.split()])

# print(net)

# leaves = dict()
# with open("./leaves.txt","r",encoding="windows-1250") as file:
#   for line in file.readlines():
#     splitline = line.split()
#     leaves[splitline[0]] = float(splitline[1])

# print(leaves)
# ###### V2
# leaves = dict()
# with open("./leaves.txt","r",encoding="windows-1250") as file:
#   for line in file.readlines():
#     splitline = line.split()
#     leaves[splitline[0]] = float(splitline[1])

# net = list()
# with open("./vocab","r",encoding="windows-1250") as file:
#   for line in file.readlines():
#     phonemes = line.split("		")[1]
#     net.append([leaves[x] for x in phonemes.split()])

# print(net)

###### V3

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

# def init_phi(net,acoustic_model):
#   phi_net = copy.deepcopy(net)
#   for path_idx in range(len(phi_net)):
#     phi_net[path_idx][0] = acoustic_model[0][phi_net[path_idx][0]] #-np.log10(acoustic_model[0][phi_net[path_idx][0]])
#     for phoneme_idx in range(1,len(phi_net[path_idx])):
#       phi_net[path_idx][phoneme_idx] = np.infty
      
#   return phi_net


# #   loop = phi_net_old[word][state] + leaves[net[word][state]][0]
# #   trans = phi_net_old[word][state-1] + leaves[net[state -1]][1]


# # def calc_phi(state_net, phi_net, acoustic_model, trans_probs):
# #   for t in range(1,len(acoustic_model)): #přes řádky souboru
# #     print(t)
# #     for word in range(1,len(phi_net)):
# #       for state in range(len(phi_net[word])):
# #         loop = phi_net[word][state] + (1-trans_probs[state_net[word][state]])
# #         trans = phi_net[word][state-1] + trans_probs[state_net[word][state-1]]

# #         phi_net[word][state] = min(trans,loop) + acoustic_model[t][state_net[state][0]]

# #   return phi_net

# def calc_phi(state_net, phi_net, phi_net_old, b_vec, b_vec_old, trans_probs):
#   ends = list()

#   for word in phi_net:
#     pass

#   for word in range(1,len(phi_net)):
#     for state in range(len(phi_net[word])):
#       net_state = state_net[state][0]
#       loop = phi_net_old[word][state] + trans_probs[net_state]
#       trans = phi_net_old[word][state-1] + (1-trans_probs[net_state-1])
      
#       phi_net[word][state] = min(trans,loop) + b_vec[net_state]

#   return phi_net

# phi_net = init_phi(word_net, acoustic_model)
# # print(phi_net)

# # phi_net = calc_phi(net, phi_net, acoustic_model, trans_probs)
# print(phi_net)
# # single_phi(net, phi_net, acoustic_model, trans_probs)
# # print(round(1-float(splitline[1])-float(splitline[2]),6))

# for t in range(1, len(acoustic_model)):
#   phi_net_old = phi_net
#   phi_net = calc_phi(word_net, phi_net, phi_net_old, acoustic_model[t], acoustic_model[t-1], trans_probs)
#   print(phi_net[0])
#   print(phi_net[-1])
#   print("-------")
# # x = []
# # for state in x:

# init phi net

phi_net = copy.deepcopy(word_net)
# min_path = np.infty
for w in range(len(phi_net)):
  phi_net[w][0] = acoustic_model[0][word_net[w][0]] #first row (time 0)
  #-np.log10(acoustic_model[0][phi_net[path_idx][0]])
  for phoneme_idx in range(1,len(phi_net[w])):
    phi_net[w][phoneme_idx] = np.infty

old_phi_net = copy.deepcopy(phi_net)

# calc
for t in range(1,len(acoustic_model)):
  for w in range(len(phi_net)):
    # if phi_net[w][-1] < min_path:
    #   min_path = w#phi_net[w][-1]

    word = word_net[w]
    for j in range(1,len(word)):
      temp = phi_net[w][j]


      prev_char = word[j-1]
      current_char = word[j]

      prev_phi = old_phi_net[w][j-1] + trans_probs[prev_char][1]
      current_phi = old_phi_net[w][j] + trans_probs[current_char][0]

      minimum = np.min([prev_phi,current_phi])

      #old_phi_net
      
      phi_net[w][j] = minimum + acoustic_model[t][current_char]
      old_phi_net[w][j] = temp
      pass
  
  ends = [x[-1] for x in phi_net]
  min_path = np.argmin(ends)
  min_val = np.min(ends)

  for w in range(len(phi_net)):
    temp = phi_net[w][0]

    prev_char = word_net[min_path][-1]
    current_char = word_net[w][0]

    prev_phi = old_phi_net[w][0] + trans_probs[current_char][0]
    current_phi = old_phi_net[w][0] + trans_probs[current_char][0]
    phi_net[w][0] = min(prev_phi,current_phi) + acoustic_model[t][current_char]

    old_phi_net[w][0] = temp




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
