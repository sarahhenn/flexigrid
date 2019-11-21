# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:22:27 2019

@author: Chrissi
"""


import pickle

print("Start loading")
pickle.load("EFH_2005.pkl")

#with open('out/cache/' +hashed_url, 'rb') as pickle_file:  #from stackoverflow
#    content = pickle.load(pickle_file)

print("Results loaded")


