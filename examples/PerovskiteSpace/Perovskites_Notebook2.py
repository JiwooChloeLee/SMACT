
# coding: utf-8

# In[22]:

# get_ipython().magic(u'matplotlib inline')
import smact.lattice as lattice
import smact
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from os import path


# In[3]:

site_A = lattice.Site([0,0,0],[+1,+2,+3])
site_B = lattice.Site([0.5,0.5,0.5],[+5,+4,+3,+2])
site_C = lattice.Site([0.5,0.5,0.5],[-2,-1])
perovskite = lattice.Lattice([site_A,site_B,site_C],space_group=221)


# In[4]:

search = smact.ordered_elements(3,87)


# In[5]:

A_list = []
B_list = []
C_list = [['O',-2,1.35],['S',-2,1.84],['Se',-2,1.98],['F',-1,1.285],['Br',-1,1.96],['I',-1,2.2]]
for element in search:
    with open(path.join(smact.data_directory, 'shannon_radii.csv'), 'rU') as f:
        reader = csv.reader(f)
        r_shannon=False
        for row in reader:
            if row[2]=="12_n" and row[0]==element and int(row[1]) in site_A.oxidation_states:
                A_list.append([row[0],row[1],row[4]])
            if row[2]=="6_n" and row[0]==element and int(row[1]) in site_B.oxidation_states:
                B_list.append([row[0],row[1],row[4]])



# In[6]:

charge_balanced = []
goldschmidt_cubic = []
goldschmidt_ortho = []
a_too_large = []
A_B_similar = []
pauling_perov = []
anion_stats = []
for C in C_list:
    anion_hex = 0
    anion_cub = 0
    anion_ort = 0
    for B in B_list:
        for A in A_list:
            if B[0] != A[0]:        
                if C[0] != A[0] and C[0] != B[0]:
                    if int(A[1])+int(B[1])+3*int(C[1]) == 0:
                         charge_balanced.append([A[0],B[0],C[0]])
                         paul_a = smact.Element(A[0]).pauling_eneg
                         paul_b = smact.Element(B[0]).pauling_eneg
                         paul_c = smact.Element(C[0]).pauling_eneg
                         electroneg_makes_sense = smact.pauling_test([A[1],B[1],C[1]], [paul_a,paul_b,paul_c])
                         if electroneg_makes_sense:
                             pauling_perov.append([A[0],B[0],C[0]])
                         tol = (float(A[2]) + C[2])/(np.sqrt(2)*(float(B[2])+C[2]))
                         if tol > 1.0:
                            a_too_large.append([A[0],B[0],C[0]])
                            anion_hex = anion_hex+1
                         if tol > 0.9 and tol <= 1.0:
                            goldschmidt_cubic.append([A[0],B[0],C[0]])
                            anion_cub = anion_cub + 1
                         if tol >= 0.71 and tol < 0.9:
                            goldschmidt_ortho.append([A[0],B[0],C[0]])
                            anion_ort = anion_ort + 1
                         if tol < 0.71:
                            A_B_similar.append([A[0],B[0],C[0]])
    anion_stats.append([anion_hex,anion_cub,anion_ort]) 


# In[33]:

print anion_stats
colours=['#991D1D','#8D6608','#857070']
matplotlib.rcParams.update({'font.size': 22})
plt.pie(anion_stats[5],labels=['Hex','Cubic','Ortho']
        ,startangle=90,autopct='%1.1f%%',colors=colours)
plt.axis('equal')
plt.savefig('I-perovskites.png')


# In[18]:



print 'Number of possible charge neutral perovskites from', search[0], 'to', search[len(search)-1], '=', len(charge_balanced)
print 'Number of Pauling senseibe perovskites from', search[0], 'to', search[len(search)-1], '=', len(pauling_perov)
print 'Number of possible cubic perovskites from', search[0], 'to', search[len(search)-1], '=', len(goldschmidt_cubic)
print 'Number of possible ortho perovskites from', search[0], 'to', search[len(search)-1], '=', len(goldschmidt_ortho)
print 'Number of possible hexagonal perovskites from', search[0], 'to', search[len(search)-1], '=', len(a_too_large)
print 'Number of possible non-perovskites from', search[0], 'to', search[len(search)-1], '=', len(A_B_similar)



# In[20]:

print goldschmidt_cubic


# In[ ]:



