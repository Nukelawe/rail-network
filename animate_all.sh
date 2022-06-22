#!/bin/sh
while read p; do
	p=${p/file \'/}
	p=${p/\'/}
	python animate.py $p
done < concat.txt
