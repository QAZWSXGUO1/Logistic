import logRegres
import sys

dataArr, labelMat = logRegres.loadDataSet()
matrix = logRegres.gradAscent(dataArr, labelMat)

#print matrix[1]

for line in sys.stdin:
	line = line.strip()
	lines = line.split('\t');
	result =  float(matrix[0]) + float(lines[0]) * float(matrix[1]) + float(lines[1]) * float(matrix[2])
	print "%s\t%f"%(line, result)
