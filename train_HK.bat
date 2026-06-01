@echo off
setlocal

set env_name=HollowKnight_Silksong

python -u train_HK.py ^
    -n "%env_name%_2L256D4H_200k_seed1_9FPS_bs64_twls2_ibl32" ^
    -seed 1 ^
    -config_path "config_files/STORM_HK_Silksong.yaml" ^
    -trajectory_path "D_TRAJ/%env_name%.pkl" ^
    -env_name "%env_name%" ^
    -resume_step 410192

endlocal
pause

