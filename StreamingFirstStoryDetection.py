from datetime import datetime
from pygtrie import CharTrie
import numpy as np
from scipy import spatial
import math
import time
import matplotlib.pyplot as plt

file = open("preprocessedTweets.txt",'r')

'''
TF : 1+log(tf)
IDF : log(D/df)
'''

mu, sigma = 0, 1
# loss probability = 2.5% or 0.025
delta = 0.025
K = 5
L = int(math.log(delta, 1 - 0.5**K))

# N is top terms based on IDF score
N = 300

# threshold
t = 0.2

# array of hash tables (Tries)
# hashtable bucket contains list of documents sorted based on timestamps, stored as (Doc, timestamp), sorted in increasing
# or of time stamp, newest document appended at the end of the list

hash_table_array = [CharTrie() for i in range(L)]
# L*k
hyperplanes = [[np.random.normal(mu, sigma, N) for i in range(K)] for j in range(L)]

# print(hyperplanes)

inverted_index = CharTrie()
DocumentVectors = CharTrie()

# need to decide appropriate data structure to store the docIDs sorted with most recent timestamps
# not needed, any new document is latest, hence insert it at index 0, (but better if inserted at the end for amortized
# complexity of O(1)
time_series = []
# also need to regulate the size of time series object

# Threads, contains sets of documents
Threads = []

# First Stories
FirstStory = []

count = 0
curTime = time.time()
x_values = []
y_values = []
while file:
    line = file.readline()
    if line == '' or count>1000:
        break
    if count%50==0:
        print(count)
        x_values.append(count)
        y_values.append(time.time()-curTime)
        curTime = time.time()

    line = line.split(',')
    docID = line[-1].rstrip(' \r\n').lstrip(' ')
    time_string = line[2].split()
    time_string2 = time_string[3].split(':')
    timeStamp = datetime(int(time_string[-1]), 3, int(time_string[2]), int(time_string2[0]), int(time_string2[1]),
                 int(time_string2[2]))

    # need to limit number of documents in timeseries (either to top 2000, or to last 10 days)
    if len(time_series)>0:
        if (timeStamp - time_series[0][1]).days > 10:
            time_series.pop(0)

    # newest document at the end of the list
    time_series.append((docID, timeStamp))
    terms = file.readline().split()
    # term -> list(idf, size, document_dictionary{docID: termFreq})
    for i in terms:
        if inverted_index.has_node(i) == 1:
            temp = inverted_index[i]
            dictionary = temp[2]
            if dictionary.has_node(docID) == 1:
                dictionary[docID] += 1
            else:
                dictionary[docID] = 1
                temp[1] += 1
        else:
            inverted_index[i] = [0, 1, CharTrie()]
            inverted_index[i][2][docID] = 1
    count += 1

    # calculate IDF weight from inverted index
    top_terms = []
    for i in inverted_index:
        inverted_index[i][0] = np.log(inverted_index[i][1]/count)
        top_terms.append((i,inverted_index[i][0]))

    top_terms.sort(key=lambda x: x[1])
    # print(top_terms)

    top_terms = top_terms[:N]

    # now make document vectors
    # given top n specific terms from which to retrieve this document's vector
    DocumentVectors[docID] = np.array([0 for j in range(N)])

    # update all the document vectors currently present
    for k in DocumentVectors:
        docArray = DocumentVectors[k]
        j = 0
        for i in top_terms:
            temp = inverted_index[i[0]]
            temp = temp[2]
            if temp.has_node(k):
                docArray[j] = (1 + np.log(temp[k]))*i[1]
            j += 1

    S = CharTrie()
    for i in range(L):
        string = ''
        # generate k bit string
        arr = DocumentVectors[docID]
        for j in range(K):
            if np.dot(hyperplanes[i][j], arr) < 0:
                string += '0'
            else:
                string += '1'
        hash_table = hash_table_array[i]
        if hash_table.has_key(string):
            for k in hash_table[string]:
                if S.has_key(k[0]):
                    S[k[0]] += 1
                else:
                    S[k[0]] = 1
            minDate = hash_table[string][0]
            # if any oldest document greater than 30 days, and list size exceeded (dynamic size), then remove it
            if (timeStamp - minDate[1]).days > 30:
                hash_table[string].remove(minDate)
            # because new document is definitely new and latest, hence add it to the front of the list
            hash_table[string].append((docID, timeStamp))
            # hash_table[string].sort(key=lambda x: x[1])
        else:
            hash_table[string] = [(docID, timeStamp)]

    # find top 3L most collided elements for docID, and then find cosine similarity among all of them and return
    top_elements = list(S.items())
    top_elements.sort(key=lambda x: x[1])
    top_elements = top_elements[:3 * L]

    # find max_cosine similarity
    minDistance = float('+inf')
    docVec = DocumentVectors[docID]
    nearestNeighbour = None
    for i in top_elements:
        if not DocumentVectors.has_node(i[0]):
            DocumentVectors[i[0]] = np.array([0 for j in range(N)])
            docArray = DocumentVectors[i[0]]
            j = 0
            for k in top_terms:
                temp = inverted_index[i[0]]
                temp = temp[2]
                if temp.has_node(i[0]):
                    docArray[j] = (1 + np.log(temp[i[0]])) * k[1]
                j += 1

        tempVec = DocumentVectors[i[0]]
        # need to decide between np.
        tempDistance = spatial.distance.cosine(docVec, tempVec)
        if minDistance > tempDistance:
            nearestNeighbour = i[0]
            minDistance = tempDistance

    if minDistance > t:
        # apply variance reduction
        for k in time_series[:len(time_series)-1]:
            if not DocumentVectors.has_node(k[0]):
                DocumentVectors[k[0]] = np.array([0 for j in range(len(top_terms))])
                docArray = DocumentVectors[k[0]]
                j = 0
                for i in top_terms:
                    temp = inverted_index[i[0]]
                    temp = temp[2]
                    if temp.has_node(k[0]):
                        docArray[j] = (1 + np.log(temp[k[0]])) * i[1]
                    j += 1

            tempDistance = spatial.distance.cosine(DocumentVectors[docID], DocumentVectors[k[0]])
            if tempDistance < minDistance:
                nearestNeighbour = k[0]
                minDistance = tempDistance

        if minDistance <= t:
            if nearestNeighbour is None:
                Threads.append({docID})
                FirstStory.append((docID, timeStamp))
            for i in Threads:
                if nearestNeighbour in i:
                    i.add(docID)
                    break
        else:
            # add to new thread
            Threads.append({docID})
            FirstStory.append((docID,timeStamp))

    else:
        # add it to thread
        if nearestNeighbour is None:
            Threads.append({docID})
            FirstStory.append((docID, timeStamp))
        else:
            for i in Threads:
                if nearestNeighbour in i:
                    i.add(docID)
                    break


# print Threads
for i in Threads:
    print(i)

# print first stories
for i in FirstStory:
    print(i, end=", ")

file.close()

fig = plt.figure()
ax = fig.gca()
plt.title("Time take to process every 50 tweets")
ax.set_xlabel("Tweet Number")
ax.set_ylabel("Time Taken")
ax.plot(x_values,y_values,color='green')
# ax.legend()
fig.canvas.draw()
plt.savefig('Plot_Streaming')
plt.show()
