#!/bin/bash -e

while true 
do 
	git pull
	python3 coronavirus.py
	date_now=$(date +'%m/%d/%Y')
	git add . -A
	git commit --all -m "Update $date_now"
	git push
	echo "Updated for $date_now."
	sleep 1d; 
done

