"""
Baum-Welch re-estimation algorithm implementation for KKY/ARŘ lectures
"""

from re import A
import numpy as np
from pyparsing import alphas

"""A_DICT = {
         "mean": [-1.820163e+000, 2.948045e-001, -9.102286e-001, -5.090310e-001, -4.833414e-001, -2.789246e-001, -2.483530e-001, -1.946086e-002, -2.488011e-001, -1.734584e-001, -1.326621e-001, -6.957074e-002, -6.207554e-002],
         "variance": [2.533771e+000, 3.355462e-001, 2.346364e-001, 1.186939e-001, 9.580594e-002, 6.385045e-002, 5.294375e-002, 4.059501e-002, 5.438076e-002, 3.317861e-002, 2.332679e-002, 1.959105e-002, 1.796214e-002]
         }

N_DICT = {
         "mean": [-3.409288e+000, 6.246719e-001, -4.743754e-002, -2.178264e-001, -5.912049e-001, -3.309700e-001, -3.621980e-001, -1.818327e-001, -3.115039e-001, -1.627000e-001, -1.629415e-001, -1.313724e-001, -1.055320e-001],
         "variance": [1.853842e+000, 2.350074e-001, 2.670902e-001, 1.030515e-001, 8.692522e-002, 5.001630e-002, 8.236017e-002, 4.526421e-002, 3.839431e-002, 2.944827e-002, 2.747891e-002, 1.762550e-002, 2.038012e-002]
         }

O_DICT = {
         "mean": [-2.774132e+000, 8.278207e-001, -5.067130e-001, -7.707845e-001, -4.396908e-001, -3.018584e-001, -4.529741e-001, -1.182087e-001, -5.049667e-002, -7.423452e-002, -1.431837e-001, -1.037804e-001, -6.737377e-002],
         "variance": [2.095176e+000, 2.614865e-001, 1.857337e-001, 1.567194e-001, 6.444157e-002, 6.257857e-002, 4.931529e-002, 5.150662e-002, 3.487385e-002, 2.563643e-002, 2.315338e-002, 1.673202e-002, 1.605465e-002]
         }"""

EMPTY_DICT = {
              "mean": np.zeros(13),
              "variance": np.zeros(13)
             }

A_probabilities = [
[0.000000e+000, 1.000000e+000, 0.000000e+000, 0.000000e+000, 0.000000e+000],
[0.000000e+000, 0.500000e+000, 0.500000e+000, 0.000000e+000, 0.000000e+000], 
[0.000000e+000, 0.000000e+000, 0.500000e+000, 0.500000e+000, 0.000000e+000], 
[0.000000e+000, 0.000000e+000, 0.000000e+000, 0.500000e+000, 0.500000e+000], 
[0.000000e+000, 0.000000e+000, 0.000000e+000, 0.000000e+000, 0.000000e+000]] 


loaded_text = np.loadtxt("./trainingHMM/test_1.txt")

flat_start_mean = np.average(loaded_text,axis=0)
flat_start_variance = [1.7448, 0.2294, 0.2884, 0.2020, 0.0420, 0.0620, 0.1028, 0.0230, 0.0137, 0.0298, 0.0173, 0.0076, 0.0085]# np.var(loaded_text,axis=0)

A_DICT = {
         "mean": flat_start_mean,
         "variance": flat_start_variance
         }


N_DICT = {
         "mean": flat_start_mean,
         "variance": flat_start_variance
         }
O_DICT = {
         "mean": flat_start_mean,
         "variance": flat_start_variance
         }

from acoustic_model import *
#print(A_DICT)


b_list = []
for segment in loaded_text:
  row = [0]
  #for fonem in [EMPTY_DICT,A_DICT,N_DICT,O_DICT,EMPTY_DICT]:
  for fonem in [A_DICT,N_DICT,O_DICT]:
    #print(fonem["mean"])
    row.append(b_func(segment,fonem["mean"],fonem["variance"]))
    #exit()
  row.append(0)
  b_list.append(row)
  #print(row)
  #exit()

alphas,afp = alpha_func(33,5,A_probabilities,b_list)
print(np.log(afp))
#print(loaded_text[0])
betas,bfp = beta_func(33,5,A_probabilities,b_list)
print(np.log(bfp))
#exit()
#print(betas)
#for j in range(3):
num_sum = 0
den_sum = 0



mu_j = []

# print(np.shape(alphas))
# print(np.shape(betas))
# print(np.shape(loaded_text))
for j in range(1,4):
  for t in range(33):
    #print(alphas[t][j],"\n",betas[t][j],"\n",loaded_text[t],"\n")
    num_sum += alphas[t][j]*betas[t][j]*loaded_text[t]
    den_sum += alphas[t][j]*betas[t][j]
  mu_j.append(num_sum/den_sum)

#print(mu_j)
for x,i in zip([A_DICT,N_DICT,O_DICT],range(3)):
  x["mean"] = mu_j[i]

mu_j = []

b_list = []
for segment in loaded_text:
  row = [0]
  #for fonem in [EMPTY_DICT,A_DICT,N_DICT,O_DICT,EMPTY_DICT]:
  for fonem in [A_DICT,N_DICT,O_DICT]:
    #print(fonem["mean"])
    row.append(b_func(segment,fonem["mean"],fonem["variance"]))
    #exit()
  row.append(0)
  b_list.append(row)
  #print(row)
  #exit()

alphas,afp = alpha_func(33,5,A_probabilities,b_list)
print(np.log(afp))
#print(loaded_text[0])
betas,bfp = beta_func(33,5,A_probabilities,b_list)
print(np.log(bfp))
#exit()
#print(betas)
#for j in range(3):
num_sum = 0
den_sum = 0


# print(np.shape(alphas))
# print(np.shape(betas))
# print(np.shape(loaded_text))
for j in range(1,4):
  for t in range(33):
    #print(alphas[t][j],"\n",betas[t][j],"\n",loaded_text[t],"\n")
    num_sum += alphas[t][j]*betas[t][j]*loaded_text[t]
    den_sum += alphas[t][j]*betas[t][j]
  mu_j.append(num_sum/den_sum)

# print(mu_j)

""" SECOND RE-ESTIMATION"""
for x,i in zip([A_DICT,N_DICT,O_DICT],range(3)):
  x["mean"] = mu_j[i]

mu_j = []

b_list = []
for segment in loaded_text:
  row = [0]
  #for fonem in [EMPTY_DICT,A_DICT,N_DICT,O_DICT,EMPTY_DICT]:
  for fonem in [A_DICT,N_DICT,O_DICT]:
    #print(fonem["mean"])
    row.append(b_func(segment,fonem["mean"],fonem["variance"]))
    #exit()
  row.append(0)
  b_list.append(row)
  #print(row)
  #exit()

alphas,afp = alpha_func(33,5,A_probabilities,b_list)
print(np.log(afp))
#print(loaded_text[0])
betas,bfp = beta_func(33,5,A_probabilities,b_list)
print(np.log(bfp))
#exit()
#print(betas)
#for j in range(3):
num_sum = 0
den_sum = 0


# print(np.shape(alphas))
# print(np.shape(betas))
# print(np.shape(loaded_text))
for j in range(1,4):
  for t in range(33):
    #print(alphas[t][j],"\n",betas[t][j],"\n",loaded_text[t],"\n")
    num_sum += alphas[t][j]*betas[t][j]*loaded_text[t]
    den_sum += alphas[t][j]*betas[t][j]
  mu_j.append(num_sum/den_sum)

  """ THIRD RE-ESTIMATION"""
for x,i in zip([A_DICT,N_DICT,O_DICT],range(3)):
  x["mean"] = mu_j[i]

mu_j = []

b_list = []
for segment in loaded_text:
  row = [0]
  #for fonem in [EMPTY_DICT,A_DICT,N_DICT,O_DICT,EMPTY_DICT]:
  for fonem in [A_DICT,N_DICT,O_DICT]:
    #print(fonem["mean"])
    row.append(b_func(segment,fonem["mean"],fonem["variance"]))
    #exit()
  row.append(0)
  b_list.append(row)
  #print(row)
  #exit()

alphas,afp = alpha_func(33,5,A_probabilities,b_list)
print(np.log(afp))
#print(loaded_text[0])
betas,bfp = beta_func(33,5,A_probabilities,b_list)
print(np.log(bfp))
#exit()
#print(betas)
#for j in range(3):
num_sum = 0
den_sum = 0


# print(np.shape(alphas))
# print(np.shape(betas))
# print(np.shape(loaded_text))
for j in range(1,4):
  for t in range(33):
    #print(alphas[t][j],"\n",betas[t][j],"\n",loaded_text[t],"\n")
    num_sum += alphas[t][j]*betas[t][j]*loaded_text[t]
    den_sum += alphas[t][j]*betas[t][j]
  mu_j.append(num_sum/den_sum)

""" FOURTH RE-ESTIMATION"""
for x,i in zip([A_DICT,N_DICT,O_DICT],range(3)):
  x["mean"] = mu_j[i]

mu_j = []

b_list = []
for segment in loaded_text:
  row = [0]
  #for fonem in [EMPTY_DICT,A_DICT,N_DICT,O_DICT,EMPTY_DICT]:
  for fonem in [A_DICT,N_DICT,O_DICT]:
    #print(fonem["mean"])
    row.append(b_func(segment,fonem["mean"],fonem["variance"]))
    #exit()
  row.append(0)
  b_list.append(row)
  #print(row)
  #exit()

alphas,afp = alpha_func(33,5,A_probabilities,b_list)
print(np.log(afp))
#print(loaded_text[0])
betas,bfp = beta_func(33,5,A_probabilities,b_list)
print(np.log(bfp))
#exit()
#print(betas)
#for j in range(3):
num_sum = 0
den_sum = 0


# print(np.shape(alphas))
# print(np.shape(betas))
# print(np.shape(loaded_text))
for j in range(1,4):
  for t in range(33):
    #print(alphas[t][j],"\n",betas[t][j],"\n",loaded_text[t],"\n")
    num_sum += alphas[t][j]*betas[t][j]*loaded_text[t]
    den_sum += alphas[t][j]*betas[t][j]
  mu_j.append(num_sum/den_sum)

  """ 6TH RE-ESTIMATION"""
for x,i in zip([A_DICT,N_DICT,O_DICT],range(3)):
  x["mean"] = mu_j[i]

mu_j = []

b_list = []
for segment in loaded_text:
  row = [0]
  #for fonem in [EMPTY_DICT,A_DICT,N_DICT,O_DICT,EMPTY_DICT]:
  for fonem in [A_DICT,N_DICT,O_DICT]:
    #print(fonem["mean"])
    row.append(b_func(segment,fonem["mean"],fonem["variance"]))
    #exit()
  row.append(0)
  b_list.append(row)
  #print(row)
  #exit()

alphas,afp = alpha_func(33,5,A_probabilities,b_list)
print(np.log(afp))
#print(loaded_text[0])
betas,bfp = beta_func(33,5,A_probabilities,b_list)
print(np.log(bfp))
#exit()
#print(betas)
#for j in range(3):
num_sum = 0
den_sum = 0


# print(np.shape(alphas))
# print(np.shape(betas))
# print(np.shape(loaded_text))
for j in range(1,4):
  for t in range(33):
    #print(alphas[t][j],"\n",betas[t][j],"\n",loaded_text[t],"\n")
    num_sum += alphas[t][j]*betas[t][j]*loaded_text[t]
    den_sum += alphas[t][j]*betas[t][j]
  mu_j.append(num_sum/den_sum)

  """ SECOND RE-ESTIMATION"""
for x,i in zip([A_DICT,N_DICT,O_DICT],range(3)):
  x["mean"] = mu_j[i]

mu_j = []

b_list = []
for segment in loaded_text:
  row = [0]
  #for fonem in [EMPTY_DICT,A_DICT,N_DICT,O_DICT,EMPTY_DICT]:
  for fonem in [A_DICT,N_DICT,O_DICT]:
    #print(fonem["mean"])
    row.append(b_func(segment,fonem["mean"],fonem["variance"]))
    #exit()
  row.append(0)
  b_list.append(row)
  #print(row)
  #exit()

alphas,afp = alpha_func(33,5,A_probabilities,b_list)
print(np.log(afp))
#print(loaded_text[0])
betas,bfp = beta_func(33,5,A_probabilities,b_list)
print(np.log(bfp))
#exit()
#print(betas)
#for j in range(3):
num_sum = 0
den_sum = 0


# print(np.shape(alphas))
# print(np.shape(betas))
# print(np.shape(loaded_text))
for j in range(1,4):
  for t in range(33):
    #print(alphas[t][j],"\n",betas[t][j],"\n",loaded_text[t],"\n")
    num_sum += alphas[t][j]*betas[t][j]*loaded_text[t]
    den_sum += alphas[t][j]*betas[t][j]
  mu_j.append(num_sum/den_sum)