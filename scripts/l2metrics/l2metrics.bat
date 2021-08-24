@echo off

set dest_name=%1
set src_log=%1

set metrics=(c3_match c3_activation context_id_accuracy partial_id_accuracy cluster_quality)

set out_dir=work/results/l2metrics/%dest_name%
set src_dir=E:\dev\mount\data\dist\M15_Data_Drop_1\C3_configs\logs\%src_log%

for %%m in %metrics% do (
    python -m l2metrics -p %%m -o %%m -O %out_dir%/%%m -l %src_dir%
)
