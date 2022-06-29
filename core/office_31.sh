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
# Epoch='30'
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
# warmup='3'
# Epoch='30'
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
# Epoch='30'
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
# warmup='3'
# Epoch='30'
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
# warmup='3'
# Epoch='30'
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
# # Ar --> Cl
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Ar_Cl'
# RecordFolder='officehome/Ar_Cl'

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
# # Ar --> Rw
# Log='yes'
# Noise='sym'
# warmup='5'
# Epoch='30'
# lr='0.01'
# phase='noise'
# p_threshold='0.6'
# dset='home_Ar_Rw'
# RecordFolder='officehome/Ar_Rw'

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
# Epoch='30'
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
#for rate in 0.5 0.6 0.7 0.8
# for rate in 0.8
# do
#     # A → W
#     Log='yes'
#     Noise='sym'
#     warmup='3'
#     Epoch='30'
#     lr='0.01'
#     phase='noise'
#     p_threshold='0.6'
#     dset='office_a_w'
#     RecordFolder='A_W/High_noise'
#     ratio=$rate

#     python core/office.py \
#         --Log ${Log} \
#         --Noise ${Noise} \
#         --warmup ${warmup} \
#         --Epoch ${Epoch} \
#         --lr ${lr} \
#         --phase ${phase} \
#         --p_threshold ${p_threshold} \
#         --dset ${dset} \
#         --RecordFolder ${RecordFolder} \
#         --noise_ratio ${ratio}
# done


#for rate in 0.5 0.6 0.7 0.8
# for rate in 0.8
# do
#     # A → D
#     Log='yes'
#     Noise='sym'
#     warmup='3'
#     Epoch='30'
#     lr='0.01'
#     phase='noise'
#     p_threshold='0.6'
#     dset='office_a_d'
#     RecordFolder='A_D/High_noise'
#     ratio=$rate

#     python core/office.py \
#         --Log ${Log} \
#         --Noise ${Noise} \
#         --warmup ${warmup} \
#         --Epoch ${Epoch} \
#         --lr ${lr} \
#         --phase ${phase} \
#         --p_threshold ${p_threshold} \
#         --dset ${dset} \
#         --RecordFolder ${RecordFolder} \
#         --noise_ratio ${ratio}
# done

# for rate in 0.5
# #for rate in 0.8
# do
#     # W → D
#     Log='yes'
#     Noise='sym'
#     warmup='5'
#     Epoch='30'
#     lr='0.01'
#     phase='noise'
#     p_threshold='0.6'
#     dset='office_w_d'
#     RecordFolder='W_D/High_noise'
#     ratio=$rate

#     python core/office.py \
#         --Log ${Log} \
#         --Noise ${Noise} \
#         --warmup ${warmup} \
#         --Epoch ${Epoch} \
#         --lr ${lr} \
#         --phase ${phase} \
#         --p_threshold ${p_threshold} \
#         --dset ${dset} \
#         --RecordFolder ${RecordFolder} \
#         --noise_ratio ${ratio}
# done

# for rate in 0.5 0.6 0.7 0.8
# #for rate in 0.8
# do
#     # W → A
#     Log='yes'
#     Noise='sym'
#     warmup='3'
#     Epoch='30'
#     lr='0.01'
#     phase='noise'
#     p_threshold='0.6'
#     dset='office_w_a'
#     RecordFolder='W_A/High_noise'
#     ratio=$rate

#     python core/office.py \
#         --Log ${Log} \
#         --Noise ${Noise} \
#         --warmup ${warmup} \
#         --Epoch ${Epoch} \
#         --lr ${lr} \
#         --phase ${phase} \
#         --p_threshold ${p_threshold} \
#         --dset ${dset} \
#         --RecordFolder ${RecordFolder} \
#         --noise_ratio ${ratio}
# done
####################################
# for rate in 0.5 0.6 0.7 0.8
# #for rate in 0.8
# do
#     # D → A
#     Log='yes'
#     Noise='sym'
#     warmup='3'
#     Epoch='30'
#     lr='0.01'
#     phase='noise'
#     p_threshold='0.6'
#     dset='office_d_a'
#     RecordFolder='D_A/High_noise'
#     ratio=$rate

#     python core/office.py \
#         --Log ${Log} \
#         --Noise ${Noise} \
#         --warmup ${warmup} \
#         --Epoch ${Epoch} \
#         --lr ${lr} \
#         --phase ${phase} \
#         --p_threshold ${p_threshold} \
#         --dset ${dset} \
#         --RecordFolder ${RecordFolder} \
#         --noise_ratio ${ratio}
# done
#for rate in 0.5 0.6 0.7 0.8
# for rate in 0.8
# do
#     # D → W
#     Log='yes'
#     Noise='sym'
#     warmup='3'
#     Epoch='30'
#     lr='0.01'
#     phase='noise'
#     p_threshold='0.8'
#     dset='office_d_w'
#     RecordFolder='D_W/High_noise'
#     ratio=$rate

#     python core/office.py \
#         --Log ${Log} \
#         --Noise ${Noise} \
#         --warmup ${warmup} \
#         --Epoch ${Epoch} \
#         --lr ${lr} \
#         --phase ${phase} \
#         --p_threshold ${p_threshold} \
#         --dset ${dset} \
#         --RecordFolder ${RecordFolder} \
#         --noise_ratio ${ratio}
# done
# for rate in 0.5 0.6 0.7 0.8
# do
#     # D --> W
#     Log='yes'
#     Noise='sym'
#     warmup='5'
#     Epoch='30'
#     lr='0.01'
#     phase='noise'
#     p_threshold='0.6'
#     dset='office_d_w'
#     RecordFolder='D_W/High_noise'
#     ratio=$rate

#     python core/office.py \
#         --Log ${Log} \
#         --Noise ${Noise} \
#         --warmup ${warmup} \
#         --Epoch ${Epoch} \
#         --lr ${lr} \
#         --phase ${phase} \
#         --p_threshold ${p_threshold} \
#         --dset ${dset} \
#         --RecordFolder ${RecordFolder} \
#         --noise_ratio ${ratio}
# done

# for rate in 0.5 0.6 0.7 0.8
# do
#     # D --> A
#     Log='yes'
#     Noise='sym'
#     warmup='5'
#     Epoch='30'
#     lr='0.01'
#     phase='noise'
#     p_threshold='0.6'
#     dset='office_d_a'
#     RecordFolder='D_A/High_noise'
#     ratio=$rate

#     python core/office.py \
#         --Log ${Log} \
#         --Noise ${Noise} \
#         --warmup ${warmup} \
#         --Epoch ${Epoch} \
#         --lr ${lr} \
#         --phase ${phase} \
#         --p_threshold ${p_threshold} \
#         --dset ${dset} \
#         --RecordFolder ${RecordFolder} \
#         --noise_ratio ${ratio}
# done

# for rate in 0.5 0.6 0.7 0.8
# do
#     # W --> A
#     Log='yes'
#     Noise='sym'
#     warmup='5'
#     Epoch='30'
#     lr='0.01'
#     phase='noise'
#     p_threshold='0.6'
#     dset='office_w_a'
#     RecordFolder='W_A/High_noise'
#     ratio=$rate

#     python core/office.py \
#         --Log ${Log} \
#         --Noise ${Noise} \
#         --warmup ${warmup} \
#         --Epoch ${Epoch} \
#         --lr ${lr} \
#         --phase ${phase} \
#         --p_threshold ${p_threshold} \
#         --dset ${dset} \
#         --RecordFolder ${RecordFolder} \
#         --noise_ratio ${ratio}
# done

# for rate in 0.5 0.6 0.7 0.8
# do
#     # W --> D
#     Log='yes'
#     Noise='sym'
#     warmup='5'
#     Epoch='30'
#     lr='0.01'
#     phase='noise'
#     p_threshold='0.6'
#     dset='office_w_d'
#     RecordFolder='W_D/High_noise'
#     ratio=$rate

#     python core/office.py \
#         --Log ${Log} \
#         --Noise ${Noise} \
#         --warmup ${warmup} \
#         --Epoch ${Epoch} \
#         --lr ${lr} \
#         --phase ${phase} \
#         --p_threshold ${p_threshold} \
#         --dset ${dset} \
#         --RecordFolder ${RecordFolder} \
#         --noise_ratio ${ratio}
# done

# ####  Ablation Study #######
# for rate in 0.4
# do
#     # A --> W
#     Log='yes'
#     Noise='sym'
#     warmup='3'
#     Epoch='33'
#     lr='0.01'
#     phase='noise'
#     p_threshold='0.6'
#     dset='office_a_w'
#     RecordFolder='office31/A_W/pseudo_labels'
#     ratio=$rate

#     python core/office.py \
#         --Log ${Log} \
#         --Noise ${Noise} \
#         --warmup ${warmup} \
#         --Epoch ${Epoch} \
#         --lr ${lr} \
#         --phase ${phase} \
#         --p_threshold ${p_threshold} \
#         --dset ${dset} \
#         --RecordFolder ${RecordFolder} \
#         --noise_ratio ${ratio}
# done
# for rate in 0.4
# do
#     # A --> D
#     Log='yes'
#     Noise='sym'
#     warmup='3'
#     Epoch='33'
#     lr='0.01'
#     phase='noise'
#     p_threshold='0.6'
#     dset='office_a_d'
#     RecordFolder='office31/A_D/pseudo_labels'
#     ratio=$rate

#     python core/office.py \
#         --Log ${Log} \
#         --Noise ${Noise} \
#         --warmup ${warmup} \
#         --Epoch ${Epoch} \
#         --lr ${lr} \
#         --phase ${phase} \
#         --p_threshold ${p_threshold} \
#         --dset ${dset} \
#         --RecordFolder ${RecordFolder} \
#         --noise_ratio ${ratio}
# done
# ################### restart -------
# for rate in 0.4
# do
#     # W --> A
#     Log='yes'
#     Noise='sym'
#     warmup='3'
#     Epoch='33'
#     lr='0.01'
#     phase='noise'
#     p_threshold='0.6'
#     dset='office_w_a'
#     RecordFolder='office31/W_A/pseudo_labels'
#     ratio=$rate

#     python core/office.py \
#         --Log ${Log} \
#         --Noise ${Noise} \
#         --warmup ${warmup} \
#         --Epoch ${Epoch} \
#         --lr ${lr} \
#         --phase ${phase} \
#         --p_threshold ${p_threshold} \
#         --dset ${dset} \
#         --RecordFolder ${RecordFolder} \
#         --noise_ratio ${ratio}
# done
# for rate in 0.4
# do
#     # W --> D
#     Log='yes'
#     Noise='sym'
#     warmup='3'
#     Epoch='33'
#     lr='0.01'
#     phase='noise'
#     p_threshold='0.6'
#     dset='office_w_d'
#     RecordFolder='office31/W_D/pseudo_labels'
#     ratio=$rate

#     python core/office.py \
#         --Log ${Log} \
#         --Noise ${Noise} \
#         --warmup ${warmup} \
#         --Epoch ${Epoch} \
#         --lr ${lr} \
#         --phase ${phase} \
#         --p_threshold ${p_threshold} \
#         --dset ${dset} \
#         --RecordFolder ${RecordFolder} \
#         --noise_ratio ${ratio}
# done
for rate in 0.4
do
    # D --> A
    Log='yes'
    Noise='sym'
    warmup='3'
    Epoch='33'
    lr='0.01'
    phase='noise'
    p_threshold='0.6'
    dset='office_d_a'
    RecordFolder='office31/D_A/pseudo_labels'
    ratio=$rate

    python core/office.py \
        --Log ${Log} \
        --Noise ${Noise} \
        --warmup ${warmup} \
        --Epoch ${Epoch} \
        --lr ${lr} \
        --phase ${phase} \
        --p_threshold ${p_threshold} \
        --dset ${dset} \
        --RecordFolder ${RecordFolder} \
        --noise_ratio ${ratio}
done
# for rate in 0.4
# do
#     # D --> W
#     Log='yes'
#     Noise='sym'
#     warmup='3'
#     Epoch='33'
#     lr='0.01'
#     phase='noise'
#     p_threshold='0.6'
#     dset='office_d_w'
#     RecordFolder='office31/D_W/pseudo_labels'
#     ratio=$rate

#     python core/office.py \
#         --Log ${Log} \
#         --Noise ${Noise} \
#         --warmup ${warmup} \
#         --Epoch ${Epoch} \
#         --lr ${lr} \
#         --phase ${phase} \
#         --p_threshold ${p_threshold} \
#         --dset ${dset} \
#         --RecordFolder ${RecordFolder} \
#         --noise_ratio ${ratio}
# done