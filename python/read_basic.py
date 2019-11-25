# -*- coding: utf-8 -*-
"""
@author: she
"""
import pickle 

def read_results(name, options):
    results = {}
    
    with open(options["filename_results"], "rb") as f_in:
        results["inputs_clustered"] = pickle.load(f_in)
    
    with open ("results/"+ name + '.pkl', "rb") as fin:
        results["1_ObjVal"] = pickle.load(fin) 
        results["2_Runtime"] = pickle.load(fin)
        results["3_MIPGap"] = pickle.load(fin)
        results["res_powerTrafoLoad"] = pickle.load(fin)
        results["res_powerTrafoInj"] = pickle.load(fin)
        results["res_yTrafo"] = pickle.load(fin)
        results["res_powerLine"] = pickle.load(fin)
        results["res_capacity"] = pickle.load(fin)
        results["res_powerCh"] = pickle.load(fin)
        results["res_powerDis"] = pickle.load(fin)
        results["res_SOC"] = pickle.load(fin)
        results["res_SOC_init"] = pickle.load(fin)
        results["res_constraint_Inj"] = pickle.load(fin)
        results["res_constraint_Subtr"] = pickle.load(fin)
        results["res_exBat"] = pickle.load(fin)
        results["res_actBat"] = pickle.load(fin)
        results["res_powerInj"] = pickle.load(fin)
        results["res_powerSubtr"] = pickle.load(fin)
        results["res_powerSubtrTotal"] = pickle.load(fin)
        results["res_powerInjPV"] = pickle.load(fin)
        results["res_powerUsePV"] = pickle.load(fin)
        results["res_powerPVChBat"] = pickle.load(fin)
        results["res_powerNetLoad"] = pickle.load(fin)
        results["res_powerUseBat"] = pickle.load(fin)
        results["res_powerNetChBat"] = pickle.load(fin)
        results["res_powerNetDisBat"] = pickle.load(fin)
        if options["allow_apc_opti"]:
            results["res_apc_var"] = pickle.load(fin)
            results["res_apc_total"] = pickle.load(fin)
        results["res_constraint_apc"] = pickle.load(fin)
        results["res_powerGenRealMax"] = pickle.load(fin)
        results["res_powerGenReal"] = pickle.load(fin)
        results["res_powerGenCurt"] = pickle.load(fin)
        results["res_c_inv"] = pickle.load(fin)
        results["res_c_om"] = pickle.load(fin)
        results["res_c_dem"] = pickle.load(fin)
        results["res_c_fix"] = pickle.load(fin)
        results["res_rev"] = pickle.load(fin)
        results["res_c_dem_grid"] = pickle.load(fin)
        results["res_rev_grid"] = pickle.load(fin)
        results["res_c_node"] = pickle.load(fin)
        results["res_c_total_nodes"] = pickle.load(fin)
        results["res_c_total_grid"] = pickle.load(fin)
        results["res_emission_nodes"] = pickle.load(fin)
        results["res_emission_grid"] = pickle.load(fin)
        results["nodes"] = pickle.load(fin)
        results["res_actHP"] = pickle.load(fin)
        results["res_powerHP"] = pickle.load(fin)
        results["res_powerEH"] = pickle.load(fin)
        results["res_SOC_tes"] = pickle.load(fin)
        results["res_ch_tes"] = pickle.load(fin)
        results["res_dch_tes"] = pickle.load(fin)
        results["res_heatHP"] = pickle.load(fin)
        results["res_heatEH"] = pickle.load(fin)
        results["res_powerHPNet"] = pickle.load(fin)
        results["res_powerHPPV"] = pickle.load(fin)
        results["res_powerHPBat"] = pickle.load(fin)
        results["res_powerEHNet"] = pickle.load(fin)
        results["res_powerEHPV"] = pickle.load(fin)
        results["res_powerEHBat"] = pickle.load(fin)

    return results

