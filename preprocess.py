from nltk import PorterStemmer

file = open("D:/IIT Delhi/Semester 5/COL764/Project/Datasets/cleanTweets_25_500000.txt", 'r')
writeFile = open("preprocessedTweets.txt", 'w')

count = 0
porter = PorterStemmer()
while file:
    line = file.readline()
    if line is '':
        break
    writeFile.write(line)
    line = file.readline()
    terms = line.split()
    newLine = ''
    for i in terms:
        if '#' in i or '@' in i or i.startswith('http'):
            continue
        newLine += porter.stem(i)+' '
    newLine += '\n'
    writeFile.write(newLine)
    # print(newLine)
    count+=1
    if count%10000==0:
        print(count)

writeFile.close()
file.close()