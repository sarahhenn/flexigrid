# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:58:38 2019

@author: Chrissi
"""
#import numpy as np
import xlsxwriter
#import xlrd
import run_kerber_net as kerber
#from random import randint

(net, nodes, nodelines, lineLength, lineCurrent, res_per_line, r, x, res) = kerber.net_values()

option = {'model' : "net" } #random, #liu, #net

#%% Define Variables

S = 4500         #in W

P = 0.9*S        #in W
Q = 0.4359*S     #in W

U_liu_sq = {}
U_liu = {}
U_noe = {}
U_act = {}

U_liu_sq[0] = 400*400
U_liu[0] = 400
U_noe[0] = 400
U_act[0] = 400
U_liu_sq[1] = 400*400
U_liu[1] = 400
U_noe[1] = 400
U_act[1] = 400


for [n,m] in nodelines:
    U_liu_sq[m] = U_liu_sq[n] - 2*(r[n,m]*P+x[n,m]*Q)
    if U_liu_sq[m] < 0:
        #U_liu[m] = 0
        U_liu[m] = (-1)*((-1)*U_liu_sq[m])**(1/2)
    else:
        U_liu[m] = (U_liu_sq[m])**(1/2)
    #U_noe[m] = U_noe[n] - (S*res[n,m]*lineLength[n,m])/U_noe[1]
    U_noe[m] = U_noe[n] - (S*0.91/(lineCurrent[n,m]*1000))
    #U_act[m] = U_act[n] - S*res[n,m]/U_act[n]
    U_act[m] = U_act[n] - (S*res[n,m]*lineLength[n,m])**(1/2)

#%% Write in Workbook

print("Results ready...")

workbook = xlsxwriter.Workbook('Spannungsvergleich.xlsx')
worksheet = workbook.add_worksheet()

#workbook = xlrd.open_workbook('Spannungsvergleich.xlsx')
#worksheet = workbook.add_worksheet()

results=[]


for i in nodes["grid"]:
    results.insert(i, (U_act[i], U_noe[i], U_liu[i]) )
    
row = 2
col = 0
worksheet.write(0, 0, "Leistung: " + str(S/1000) + " kW")	
worksheet.write(1, 0, "U_actual")
worksheet.write(1, 1, "U_noever")
worksheet.write(1, 2, "U_liu")


for a,b,c in (results):

    worksheet.write(row, col, a)
    worksheet.write(row, col + 1, b)
    worksheet.write(row, col + 2, c)

    row += 1
   
print("Workbook generated")

workbook.close()
#
#
