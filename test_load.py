import csv

maxData = 10#Change this
dataString = []
string_max_length = 0
with open('validated.tsv') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    #header = (client_id path sentence up_votes down_votes age gender accent locale segment)
    next(reader)#skip header
    for row in reader:
        if len(dataString) >= maxData:
            break
        sentence = row[2].split(" ")
        print("sentence: ", sentence)
        dataString.append(sentence)
        string_max_length = max(len(sentence), string_max_length)
print("string_max_length:", string_max_length)
print("len(dataString):", len(dataString))
