#!/bin/bash

for i in *.geo
do 
	echo $i
	order="1"
	if [[ $i == *'2ord'* ]]
	then 
		order="2"
	fi
	dimension="2"
	if [[ $i == *'1d'* ]]
	then 
		dimension="1"
	fi
	if [[ $i == *'3d'* ]]
	then
		dimension="3"
	fi

	gmsh $i -$dimension -order $order -o ${i/geo/msh}
	
done

