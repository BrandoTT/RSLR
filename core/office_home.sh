# office-home mixed noise sh (0.2 feature noise and 0.2 label noise)
# 12 task 
# # Pr --> Rw
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Pr_Rw'
# RecordFolder='officehome/Pr_Rw'

# python core/office_home.py \
#     --Log ${Log} \
#     --Noise ${Noise} \
#     --warmup ${warmup} \
#     --Epoch ${Epoch} \
#     --lr ${lr} \
#     --phase ${phase} \
#     --p_threshold ${p_threshold} \
#     --dset ${dset} \
#     --RecordFolder ${RecordFolder} \

# # Pr --> Ar
# Log='yes'
# Noise='sym'
# warmup='5' # 5: 49.83  # label noise ratio:0.2 
# Epoch='40'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Pr_Ar'
# RecordFolder='officehome/Pr_Ar/12_10'

# python core/office_home.py \
#     --Log ${Log} \
#     --Noise ${Noise} \
#     --warmup ${warmup} \
#     --Epoch ${Epoch} \
#     --lr ${lr} \
#     --phase ${phase} \
#     --p_threshold ${p_threshold} \
#     --dset ${dset} \
#     --RecordFolder ${RecordFolder} \
# Pr --> Cl
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='50'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Pr_Cl'
# RecordFolder='officehome/Pr_Cl'

# python core/office_home.py \
#     --Log ${Log} \
#     --Noise ${Noise} \
#     --warmup ${warmup} \
#     --Epoch ${Epoch} \
#     --lr ${lr} \
#     --phase ${phase} \
#     --p_threshold ${p_threshold} \
#     --dset ${dset} \
#     --RecordFolder ${RecordFolder} \
# Rw --> Pr
# Log='yes'
# Noise='sym'
# warmup='7'
# Epoch='40'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Rw_Pr'
# RecordFolder='officehome/Rw_Pr'

# python core/office_home.py \
#     --Log ${Log} \
#     --Noise ${Noise} \
#     --warmup ${warmup} \
#     --Epoch ${Epoch} \
#     --lr ${lr} \
#     --phase ${phase} \
#     --p_threshold ${p_threshold} \
#     --dset ${dset} \
#     --RecordFolder ${RecordFolder} \
# Rw --> Ar
# Log='yes'
# Noise='sym'
# warmup='3'  #  before: 7 0.5791  3:0.6316 
# Epoch='40'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Rw_Ar'
# RecordFolder='officehome/Rw_Ar'

# python core/office_home.py \
#     --Log ${Log} \
#     --Noise ${Noise} \
#     --warmup ${warmup} \
#     --Epoch ${Epoch} \
#     --lr ${lr} \
#     --phase ${phase} \
#     --p_threshold ${p_threshold} \
#     --dset ${dset} \
#     --RecordFolder ${RecordFolder} \

# # Rw --> Cl (未在本机运行)
# Log='yes'
# Noise='sym'
# warmup='7'
# Epoch='40'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Rw_Cl'
# RecordFolder='officehome/Rw_Cl'

# python core/office_home.py \
#     --Log ${Log} \
#     --Noise ${Noise} \
#     --warmup ${warmup} \
#     --Epoch ${Epoch} \
#     --lr ${lr} \
#     --phase ${phase} \
#     --p_threshold ${p_threshold} \
#     --dset ${dset} \
#     --RecordFolder ${RecordFolder} \

# Ar --> Pr
# Log='yes'
# Noise='sym'
# warmup='4'
# Epoch='40'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Ar_Pr'
# RecordFolder='officehome/Ar_Pr/12_7'

# python core/office_home.py \
#     --Log ${Log} \
#     --Noise ${Noise} \
#     --warmup ${warmup} \
#     --Epoch ${Epoch} \
#     --lr ${lr} \
#     --phase ${phase} \
#     --p_threshold ${p_threshold} \
#     --dset ${dset} \
#     --RecordFolder ${RecordFolder} \

############### label noise : 0.4 
# Ar --> Cl
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Ar_Cl'
# RecordFolder='officehome/Ar_Cl/label_noise'
# python core/office_home.py \
#     --Log ${Log} \
#     --Noise ${Noise} \
#     --warmup ${warmup} \
#     --Epoch ${Epoch} \
#     --lr ${lr} \
#     --phase ${phase} \
#     --p_threshold ${p_threshold} \
#     --dset ${dset} \
#     --RecordFolder ${RecordFolder} \

# # Ar --> Pr
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Ar_Pr'
# RecordFolder='officehome/Ar_Pr/label_noise'
# python core/office_home.py \
#     --Log ${Log} \
#     --Noise ${Noise} \
#     --warmup ${warmup} \
#     --Epoch ${Epoch} \
#     --lr ${lr} \
#     --phase ${phase} \
#     --p_threshold ${p_threshold} \
#     --dset ${dset} \
#     --RecordFolder ${RecordFolder} \
# # # Ar --> Rw
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Ar_Rw'
# RecordFolder='officehome/Ar_Rw/label_noise'
# python core/office_home.py \
#     --Log ${Log} \
#     --Noise ${Noise} \
#     --warmup ${warmup} \
#     --Epoch ${Epoch} \
#     --lr ${lr} \
#     --phase ${phase} \
#     --p_threshold ${p_threshold} \
#     --dset ${dset} \
#     --RecordFolder ${RecordFolder} \

# # # Cl --> Pr
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Cl_Pr'
# RecordFolder='officehome/Cl_Pr/label_noise'
# python core/office_home.py \
#     --Log ${Log} \
#     --Noise ${Noise} \
#     --warmup ${warmup} \
#     --Epoch ${Epoch} \
#     --lr ${lr} \
#     --phase ${phase} \
#     --p_threshold ${p_threshold} \
#     --dset ${dset} \
#     --RecordFolder ${RecordFolder} \

# # # Cl --> Rw
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Cl_Rw'
# RecordFolder='officehome/Cl_Rw/label_noise'
# python core/office_home.py \
#     --Log ${Log} \
#     --Noise ${Noise} \
#     --warmup ${warmup} \
#     --Epoch ${Epoch} \
#     --lr ${lr} \
#     --phase ${phase} \
#     --p_threshold ${p_threshold} \
#     --dset ${dset} \
#     --RecordFolder ${RecordFolder} \
# # # Pr --> Ar
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Pr_Ar'
# RecordFolder='officehome/Pr_Ar/label_noise'
# python core/office_home.py \
#     --Log ${Log} \
#     --Noise ${Noise} \
#     --warmup ${warmup} \
#     --Epoch ${Epoch} \
#     --lr ${lr} \
#     --phase ${phase} \
#     --p_threshold ${p_threshold} \
#     --dset ${dset} \
#     --RecordFolder ${RecordFolder} \

# # # Pr --> Cl
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Pr_Cl'
# RecordFolder='officehome/Pr_Cl/label_noise'
# python core/office_home.py \
#     --Log ${Log} \
#     --Noise ${Noise} \
#     --warmup ${warmup} \
#     --Epoch ${Epoch} \
#     --lr ${lr} \
#     --phase ${phase} \
#     --p_threshold ${p_threshold} \
#     --dset ${dset} \
#     --RecordFolder ${RecordFolder} \
# # # Pr --> Rw
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Pr_Rw'
# RecordFolder='officehome/Pr_Rw/label_noise'
# python core/office_home.py \
#     --Log ${Log} \
#     --Noise ${Noise} \
#     --warmup ${warmup} \
#     --Epoch ${Epoch} \
#     --lr ${lr} \
#     --phase ${phase} \
#     --p_threshold ${p_threshold} \
#     --dset ${dset} \
#     --RecordFolder ${RecordFolder} \

#  # # Rw --> Cl
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Rw_Cl'
# RecordFolder='officehome/Rw_Cl/label_noise'
# python core/office_home.py \
#     --Log ${Log} \
#     --Noise ${Noise} \
#     --warmup ${warmup} \
#     --Epoch ${Epoch} \
#     --lr ${lr} \
#     --phase ${phase} \
#     --p_threshold ${p_threshold} \
#     --dset ${dset} \
#     --RecordFolder ${RecordFolder} \

#  # # Rw --> Pr
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Rw_Pr'
# RecordFolder='officehome/Rw_Pr/label_noise'
# python core/office_home.py \
#     --Log ${Log} \
#     --Noise ${Noise} \
#     --warmup ${warmup} \
#     --Epoch ${Epoch} \
#     --lr ${lr} \
#     --phase ${phase} \
#     --p_threshold ${p_threshold} \
#     --dset ${dset} \
#     --RecordFolder ${RecordFolder} \
    
# # Cl --> Rw
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Cl_Rw'
# RecordFolder='officehome/Cl_Rw'

# python core/office_home.py \
#     --Log ${Log} \
#     --Noise ${Noise} \
#     --warmup ${warmup} \
#     --Epoch ${Epoch} \
#     --lr ${lr} \
#     --phase ${phase} \
#     --p_threshold ${p_threshold} \
#     --dset ${dset} \
#     --RecordFolder ${RecordFolder} \
# #Cl --> Ar
# Log='yes'
# Noise='sym'
# warmup='5' # before  5?  50.3
# Epoch='40'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Cl_Ar'
# RecordFolder='officehome/Cl_Ar/12_10'

# python core/office_home.py \
#     --Log ${Log} \
#     --Noise ${Noise} \
#     --warmup ${warmup} \
#     --Epoch ${Epoch} \
#     --lr ${lr} \
#     --phase ${phase} \
#     --p_threshold ${p_threshold} \
#     --dset ${dset} \
#     --RecordFolder ${RecordFolder} \
# # Cl --> Pr
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Cl_Pr'
# RecordFolder='officehome/Cl_Pr'

# python core/office_home.py \
#     --Log ${Log} \
#     --Noise ${Noise} \
#     --warmup ${warmup} \
#     --Epoch ${Epoch} \
#     --lr ${lr} \
#     --phase ${phase} \
#     --p_threshold ${p_threshold} \
#     --dset ${dset} \
#     --RecordFolder ${RecordFolder} \

############################################
# Supplyment TDKE Experiments 2022-05-10
############################################
################## Noise ratio: 0.2 ##################
# Rw→Pr
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Rw_Pr'
# RecordFolder='officehome/Rw_Pr/label_noise'
# noise_ra='0.2'
# python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
#     --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}
###########
# # Rw→Ar
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Rw_Ar'
# RecordFolder='officehome/Rw_Ar/label_noise'
# noise_ra='0.2'
# python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
#     --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}
# # Rw→Cl
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Rw_Cl'
# RecordFolder='officehome/Rw_Cl/label_noise'
# noise_ra='0.2'
# python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
#     --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}
# # Pr→Ar
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Pr_Ar'
# RecordFolder='officehome/Pr_Ar/label_noise'
# noise_ra='0.2'
# python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
#     --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}
# # Pr→Cl
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Pr_Cl'
# RecordFolder='officehome/Pr_Cl/label_noise'
# noise_ra='0.2'
# python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
#     --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}
# # Pr→Rw	
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Pr_Rw'
# RecordFolder='officehome/Pr_Rw/label_noise'
# noise_ra='0.2'
# python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
#     --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}
# # Cl→Ar	
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Cl_Ar'
# RecordFolder='officehome/Cl_Ar/label_noise'
# noise_ra='0.2'
# python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
#     --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}
# # Ar→Cl	
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Ar_Cl'
# RecordFolder='officehome/Ar_Cl/label_noise'
# noise_ra='0.2'
# python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
#     --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}
# # Ar→Pr	
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Ar_Pr'
# RecordFolder='officehome/Ar_Pr/label_noise'
# noise_ra='0.2'
# python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
#     --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}
# # Ar→Rw	
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Ar_Rw'
# RecordFolder='officehome/Ar_Rw/label_noise'
# noise_ra='0.2'
# python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
#     --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}
# # Cl→Pr	
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Cl_Pr'
# RecordFolder='officehome/Cl_Pr/label_noise'
# noise_ra='0.2'
# python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
#     --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}
# # Cl→Rw	
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Cl_Rw'
# RecordFolder='officehome/Cl_Rw/label_noise'
# noise_ra='0.2'
# python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
#     --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}


# ################## Noise ratio: 0.4 ##################
# Cl→Ar	
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Cl_Ar'
# RecordFolder='officehome/Cl_Ar/label_noise'
# noise_ra='0.4'
# python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
#     --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}

# # # Pr→Ar
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Pr_Ar'
# RecordFolder='officehome/Pr_Ar/label_noise'
# noise_ra='0.4'
# python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
#     --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}

# # Rw→Ar
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Rw_Ar'
# RecordFolder='officehome/Rw_Ar/label_noise'
# noise_ra='0.4'
# python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
#     --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}

# ################## Noise ratio: 0.6 ##################
# #Rw→Pr
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Rw_Pr'
# RecordFolder='officehome/Rw_Pr/label_noise'
# noise_ra='0.6'
# python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
#     --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}
# ###########
# # Rw→Ar
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Rw_Ar'
# RecordFolder='officehome/Rw_Ar/label_noise'
# noise_ra='0.6'
# python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
#     --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}
# # Rw→Cl
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Rw_Cl'
# RecordFolder='officehome/Rw_Cl/label_noise'
# noise_ra='0.6'
# python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
#     --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}
# # Pr→Ar
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Pr_Ar'
# RecordFolder='officehome/Pr_Ar/label_noise'
# noise_ra='0.6'
# python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
#     --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}
# # Pr→Cl
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Pr_Cl'
# RecordFolder='officehome/Pr_Cl/label_noise'
# noise_ra='0.6'
# python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
#     --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}
# # Pr→Rw	
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Pr_Rw'
# RecordFolder='officehome/Pr_Rw/label_noise'
# noise_ra='0.6'
# python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
#     --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}
# # Cl→Ar	
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Cl_Ar'
# RecordFolder='officehome/Cl_Ar/label_noise'
# noise_ra='0.6'
# python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
#     --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}
# # Ar→Cl	
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Ar_Cl'
# RecordFolder='officehome/Ar_Cl/label_noise'
# noise_ra='0.6'
# python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
#     --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}
# # Ar→Pr	
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Ar_Pr'
# RecordFolder='officehome/Ar_Pr/label_noise'
# noise_ra='0.6'
# python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
#     --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}
# # Ar→Rw	
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Ar_Rw'
# RecordFolder='officehome/Ar_Rw/label_noise'
# noise_ra='0.6'
# python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
#     --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}
# # Cl→Pr	
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Cl_Pr'
# RecordFolder='officehome/Cl_Pr/label_noise'
# noise_ra='0.6'
# python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
#     --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}
# # Cl→Rw	
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Cl_Rw'
# RecordFolder='officehome/Cl_Rw/label_noise'
# noise_ra='0.6'
# python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
#     --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}


#Rw→Pr
Log='yes'
Noise='sym'
warmup='5'
Epoch='30'
lr='0.01'
phase='noise'
p_threshold='0.6'
dset='home_Rw_Pr'
RecordFolder='officehome/Rw_Pr/label_noise/feature_test'
noise_ra='0.4'
python core/office_home.py --Log ${Log} --Noise ${Noise} --warmup ${warmup} --Epoch ${Epoch} --lr ${lr} --phase ${phase} \
    --p_threshold ${p_threshold} --dset ${dset} --RecordFolder ${RecordFolder} --noise_ratio ${noise_ra}