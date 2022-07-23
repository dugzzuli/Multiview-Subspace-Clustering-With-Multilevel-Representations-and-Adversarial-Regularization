#!/bin/bash
for learning_rate_main in 0.0001 0.00001
do
	for cost_ssc_param in 10 1 0.1 0.01 0.001 0.0001 0.00001
	do
		for reg_ssc_param in 0.1 0.01 0.001
		do
			for diver_param in 0.1 0.01 0.001
            do  
        
            echo "--learning_rate_main $learning_rate_main --cost_ssc_param $cost_ssc_param --reg_ssc_param $reg_ssc_param --diver_param $diver_param --IV_param $IV_param"

            python mainMLR_MV_UniversalGuass_single.py --learning_rate_main $learning_rate_main --cost_ssc_param $cost_ssc_param --reg_ssc_param $reg_ssc_param --diver_param $diver_param --IV_param $diver_param
            
            done
        done
    done
done