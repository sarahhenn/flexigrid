# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:58:38 2019

@author: Chrissi
"""
import numpy as np
import xlsxwriter
from random import randint


option = {'model' : "liu" } #random, #liu


S = 300         #in W
S_rand = {}      #in W

R = 0.642        #in Ohm
R_rand = {}      #in Ohm
r = 0.642
X_rand = {}      #in Ohm
X = 0.083

P = 0.9*S        #in W
P_rand = {}      #in W

Q = 0.4359*S     #in W
Q_rand = {}      #in W

U = {}           #in V
U_start = 400    #in V
U_as = {}        #in V
deltaU = []      #in V
deltaU_as = []
deltaU_sum = {}
U_sq = {}

numOfLoads = 50
x = np.arange(0, numOfLoads)

U[0] = U_start
U_as[0] = U_start
U_sq[0] = U_start*U_start
diff = {}
U_square = []
U_square.append(U_start*U_start)
U_node = {}
deltaU_liu = []
Liu_V = 2*(R*P+X*Q)


for r in x:
        S_rand[r] = (randint(35000,35000))
        R_rand[r] = (randint(51,51))/1000
        P_rand[r] = 0.9*S_rand[r]
        Q_rand[r] = 0.4359*S_rand[r]

if option["model"] == "random":
    print("Random Power per Load and Resistance per line ")   
    
    for n in x:  
        deltaU.append (S_rand[n]*R_rand[n]/U[n])
        deltaU_as.append (S_rand[n]*R_rand[n]/U_start)
        U[n+1] = U[n]-deltaU[n]
        deltaU_sum[n] = np.sum(deltaU_as)
        U_as[n+1] = U_start-deltaU_sum[n]
        diff[n] = (U_as[n]-U[n])/U_start*100

elif option["model"] == "liu":
    
    for n in x:  
        U_square.append (U_square[n-1]-Liu_V)
        deltaU_as.append (S_rand[n]*R_rand[n]/U_start)
        U_sq[n+1] = U_sq[n] - Liu_V
        deltaU.append (S*R/U[n])
        U[n+1] = U[n]-deltaU[n]
        U_node[n] = (U_sq[n])**(1/2)
        diff[n] = (U_node[n]-U[n])/U_start*100
       
else:
    for n in x:  
        deltaU.append (S*R/U[n])
        U[n+1] = U[n]-deltaU[n]
        U_as[n] = U_start-x[n]*deltaU[0] 
        diff[n] = (U_as[n]-U[n])/U_start*100


print("Results ready...")

#workbook = xlsxwriter.Workbook('Spannungsvergleich.xlsx')
#worksheet = workbook.add_worksheet()
#
#results=[]
#
#if option["random"]:
#
#    for i in x:
#        results.insert(i, (deltaU[i],deltaU_as[i],U_sq[i],diff[i]))
#        
#    row = 1
#    col = 0
#	
#
#    worksheet.write(0, 0, "ΔU_tatsächlich pro Leitungsstück")
#    worksheet.write(0, 1, "ΔU_annahme pro Leitungsstück")
#    worksheet.write(0, 2, "U in V")
#    worksheet.write(0, 3, "Differenz zw. Angenommener und Berechneter Spannung")
#
#    for a,b,c,d in (results):
#    
#        worksheet.write(row, col, a)
#        worksheet.write(row, col + 1, b)
#        worksheet.write(row, col + 2, c)
#        worksheet.write(row, col + 3, d)
#        row += 1
#elif option["liu"]:
#    for i in x:
#        results.insert(i, (U_sq[i]))
#   
#    row = 1
#    col = 0
#
#    worksheet.write(0, 0, "ΔU pro Leitungsstück")
#    
#    for a in (results):
#        
#        worksheet.write(row, col, a)
#        row += 1
#    
#else:
#    for i in x:
#        results.insert(i, (U_sq[i],deltaU[i],diff[i]))
#   
#    row = 1
#    col = 0
#
#    worksheet.write(0, 0, "ΔU_tatsächlich pro Leitungsstück")
#    worksheet.write(0, 1, "U in V")
#    worksheet.write(0, 2, "Differenz zw. Angenommener und Berechneter Spannung")
#
#
#    for a,b,c in (results):
#        
#        worksheet.write(row, col, a)
#        worksheet.write(row, col + 1, b)
#        worksheet.write(row, col + 2, c)
#        row += 1
#
#   
#print("Workbook generated")
#
#workbook.close()
#
#
