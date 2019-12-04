# -*- coding: utf-8 -*-
"""
@author: she
"""
import pickle 

def read_results(name_results, name_dist):
    results = {}
    
    with open( name_results + ".pkl", "rb") as f_in:
        results["inputs_clustered"] = pickle.load(f_in)
        
    
    with open ( name_dist + '.pkl', "rb") as fin:
        results["loads_with"] = pickle.load(fin)                        #01
        results["line_to_loads"] = pickle.load(fin)                     #02
        results["loads_per_branch"] = pickle.load(fin)                  #03
        results["loads_per_net_with"] = pickle.load(fin)                #04
#        results["line_to_loads"] = pickle.load(fin)                     #05
#        results["line_to_loads"] = pickle.load(fin)                     #06
#        results["line_to_loads"] = pickle.load(fin)                     #07
#        results["line_to_loads"] = pickle.load(fin)                     #08
        
        
    
    with open ( name_results + '.pkl', "rb") as fin:
        results["1_ObjVal"] = pickle.load(fin)                          #01
        results["2_Runtime"] = pickle.load(fin)                         #02
        results["3_MIPGap"] = pickle.load(fin)                          #03
        results["res_powerTrafoLoad"] = pickle.load(fin)                #04
        results["res_powerTrafoInj"] = pickle.load(fin)                 #05
        results["res_powerLine"] = pickle.load(fin)                     #06
        results["res_capacity"] = pickle.load(fin)                      #07
        results["res_powerCh"] = pickle.load(fin)                       #08
        results["res_powerDis"] = pickle.load(fin)                      #09
        results["res_SOC"] = pickle.load(fin)                           #10
        results["res_SOC_init"] = pickle.load(fin)                      #11
        results["res_powerLoad"] = pickle.load(fin)                     #12
        results["res_powerInj"] = pickle.load(fin)                      #13
        results["res_powerInjPV"] = pickle.load(fin)                    #14
        results["res_powerInjBat"] = pickle.load(fin)                   #15
        results["res_powerUsePV"] = pickle.load(fin)                    #16
        results["res_powerUseBat"] = pickle.load(fin)                   #17
        results["res_powerPV"] = pickle.load(fin)                       #18
        results["res_powerPlug"] = pickle.load(fin)                     #19
        results["res_c_inv"] = pickle.load(fin)                         #20
        results["res_c_om"] = pickle.load(fin)                          #21
        results["res_c_dem"] = pickle.load(fin)                         #22
        results["res_c_fix"] = pickle.load(fin)                         #23
        results["res_rev"] = pickle.load(fin)                           #24
        results["res_c_dem_grid"] = pickle.load(fin)                    #25
        results["res_rev_grid"] = pickle.load(fin)                      #26
        results["res_c_node"] = pickle.load(fin)                        #27
        results["res_c_total_nodes"] = pickle.load(fin)                 #28
        results["res_c_total_grid"] = pickle.load(fin)                  #29
        results["res_emission_nodes"] = pickle.load(fin)                #30
        results["res_emission_grid"] = pickle.load(fin)                 #31
        results["nodes"] = pickle.load(fin)                             #32
        results["nodeLines"] = pickle.load(fin)                         #33
        results["res_actHP"] = pickle.load(fin)                         #34
        results["res_powerHP"] = pickle.load(fin)                       #35
        results["res_powerEH"] = pickle.load(fin)                       #36
        results["res_SOC_tes"] = pickle.load(fin)                       #37
        results["res_SOC_init_tes"] = pickle.load(fin)                  #38
        results["res_ch_tes"] = pickle.load(fin)                        #39
        results["res_dch_tes"] = pickle.load(fin)                       #40
        results["res_heatHP"] = pickle.load(fin)                        #41
        results["res_heatEH"] = pickle.load(fin)                        #42
        results["res_voltLine"] = pickle.load(fin)                      #43
        results["res_voltNode"] = pickle.load(fin)                      #44
        results["res_powerHPGrid"] = pickle.load(fin)                   #45
        results["res_powerHPPV"] = pickle.load(fin)                     #46
        results["res_powerHPBat"] = pickle.load(fin)                    #47
        results["res_powerEHGrid"] = pickle.load(fin)                   #48
        results["res_powerEHPV"] = pickle.load(fin)                     #49
        results["res_powerEHBat"] = pickle.load(fin)                    #50
        results["res_exBat"] = pickle.load(fin)                         #51
        results["res_actBat"] = pickle.load(fin)                        #52
        results["res_ev_load"] = pickle.load(fin)
        results["res_ev_inj"] = pickle.load(fin)


#    with open ("results/"+ name + '.pkl', "rb") as fin:
#        results["1_ObjVal"] = pickle.load(fin) 
#        results["2_Runtime"] = pickle.load(fin)
#        results["3_MIPGap"] = pickle.load(fin)
#        results["res_powerTrafoLoad"] = pickle.load(fin)
#        results["res_powerTrafoInj"] = pickle.load(fin)
#        results["res_powerLine"] = pickle.load(fin)
#        results["res_capacity"] = pickle.load(fin)
#        results["res_powerCh"] = pickle.load(fin)
#        results["res_powerDis"] = pickle.load(fin)
#        results["res_SOC"] = pickle.load(fin)
#        results["res_SOC_init"] = pickle.load(fin)
#        results["res_powerLoad"] = pickle.load(fin)
#        results["res_powerInj"] = pickle.load(fin)
#        results["res_powerInjPV"] = pickle.load(fin)
#        results["res_powerInjBat"] = pickle.load(fin)
#        results["res_powerUsePV"] = pickle.load(fin)
#        results["res_powerUseBat"] = pickle.load(fin)
#        results["res_powerPV"] = pickle.load(fin)
#        results["res_powerPlug"] = pickle.load(fin)
#        results["res_c_inv"] = pickle.load(fin)
#        results["res_c_om"] = pickle.load(fin)
#        results["res_c_dem"] = pickle.load(fin)
#        results["res_c_fix"] = pickle.load(fin)
#        results["res_rev"] = pickle.load(fin)
#        results["res_c_dem_grid"] = pickle.load(fin)
#        results["res_rev_grid"] = pickle.load(fin)
#        results["res_c_node"] = pickle.load(fin)
#        results["res_c_total_nodes"] = pickle.load(fin)
#        results["res_c_total_grid"] = pickle.load(fin)
#        results["res_emission_nodes"] = pickle.load(fin)
#        results["res_emission_grid"] = pickle.load(fin)
#        results["nodes"] = pickle.load(fin)
#        results["res_actHP"] = pickle.load(fin)
#        results["res_powerHP"] = pickle.load(fin)
#        results["res_powerEH"] = pickle.load(fin)
#        results["res_SOC_tes"] = pickle.load(fin)
#        results["res_SOC_init_tes"] = pickle.load(fin)
#        results["res_ch_tes"] = pickle.load(fin)
#        results["res_dch_tes"] = pickle.load(fin)
#        results["res_heatHP"] = pickle.load(fin)
#        results["res_heatEH"] = pickle.load(fin)
#        results["res_powerHPGrid"] = pickle.load(fin)
#        results["res_powerHPPV"] = pickle.load(fin)
#        results["res_powerHPBat"] = pickle.load(fin)
#        results["res_powerEHGrid"] = pickle.load(fin)
#        results["res_powerEHPV"] = pickle.load(fin)
#        results["res_powerEHBat"] = pickle.load(fin)
#        results["res_exBat"] = pickle.load(fin)
#        results["res_actBat"] = pickle.load(fin)


    return results

