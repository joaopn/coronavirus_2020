#!/bin/bash -e

while true 
do 
	git pull
	python3 coronavirus.py --plots age
	python3 coronavirus.py --plots countries
	date_now=$(date +'%m/%d/%Y')
	git add . -A
	git commit --all -m "Update github.io $date_now"
	git push
	echo "Updated github.io for $date_now."
	sleep 1d; 
done

