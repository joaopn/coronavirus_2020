#!/bin/bash -e

python3 coronavirus.py

date_now=$(date +'%m/%d/%Y')

git add . -A
git commit --all -m "Update $date_now"
#git push