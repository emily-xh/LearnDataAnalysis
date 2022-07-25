# load the data from CSV file
# map the string value to int
#
import csv
import numpy as np

def CSVLoader(file):
    with open(file, newline='') as csvfile:
        csvdata = csv.reader(csvfile, delimiter=',', quotechar='|')
        data, position_dict, col = [], {}, 0
        for line in csvdata:
            data.append(line)
        for item in data[0]:
            position_dict[item] = col
            col += 1
    return position_dict, data[1:]

def getUniqueVal(position, data):
    s = set()
    for d in data:
        s.add(d[position])
    return s

def mapValue(valdict, val):
    try:
        return (float)(val)
    except:
        return valdict[val]

def getAttribute(key_list, position_dict, data):
    valdict = {}
    for key in key_list:
        s = getUniqueVal(position_dict[key], data)
        ct = 0
        for val in (list)(s):
            valdict[val] = ct
            ct +=1
    fea = []
    for d in data:
        tmp = []
        for key in key_list:
            val = mapValue(valdict, d[position_dict[key]])
            tmp.append(val)
        fea.append(tmp)
    return np.array(fea)