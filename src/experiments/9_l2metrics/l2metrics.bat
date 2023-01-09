@echo off

set dest_name=%1
set src_log=%1

set metrics=(performance, art_match, art_activation)

set out_dir=work\results\9_l2metrics\l2metrics\%dest_name%
set src_dir=work\results\9_l2metrics\logs\%src_log%

for %%m in %metrics% do (
    python -m l2metrics -p %%m -o %%m -O %out_dir%/%%m -l %src_dir%
)
