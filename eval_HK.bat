@echo off
set run_name=HollowKnight_Silksong_2L256D4H_200k_seed1_9FPS_bs64_twls2_ibl32
set num_episode=10

python eval_HK.py ^
    -run_name "%run_name%" ^
    -env_name "HollowKnight_Silksong" ^
    -config_path "config_files/STORM_HK_Silksong.yaml" ^
    -num_episode %num_episode% 

pause

