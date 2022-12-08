import numpy as np
from mat4py import loadmat
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import math
from scipy.spatial import distance
data = loadmat('emnist-digits-150.mat')
a = data['dataset']
b = a['DigitImage']m
images = b['images']
labels = b['labels']
digits = np.array(images)
bwdigits = digits
image = digits[2]
image = image.T
fimage = np.array(image, dtype='float')
pixels = image.reshape((28, 28))
plt.imshow(pixels)
plt.show()
# extracting features from pca method 
pca=PCA(n_components=150)
pca.fit(digits)
NewDigits=pca.transform(digits)
#implementing Eqivalence relation function 
def Rfunc(X1 , X2 ,q):
    if np.linalg.norm(X1-X2) == 0 :
        return 1
    sigma = 1 / np.linalg.norm(X1-X2)

    a = 0
    for i in range (len(X1)):
        a = a + pow((abs(X1[i]-X2[i])),q)
    a = pow(a,1/q)
    a =(sigma*a)
    a = math.floor(a * 10 ** 3) / 10 ** 3
    R = 1 - a
    R = math.floor(R * 10 ** 3) / 10 ** 3
    return R
table = [[0 for x in range(len(NewDigits))] for y in range(len(NewDigits))] 
for i in range(len(NewDigits)):
    for j in range(len(NewDigits)):
        a = Rfunc(NewDigits[i],NewDigits[j],3)
        table[i][j] = a
        
table = np.array(table)

def RoR(Rtable):
    Rtable= list(Rtable)
    flag=True
    Rprime=Rtable
    temp=Rtable
    while flag:
        Rtable=Rprime
        temp=Rtable
        max1=0
        for k in range(150):
            for i in range(150):
                max=0
                for j in range(150):
                     if min(Rtable[k][i],Rtable[i][j])>max1:
                        max1=min(Rtable[k][j],Rtable[j][i])
                        Rprime[k][j]=max1


        if temp==Rprime:
            flag=False
        return Rprime
    # RP = [[0 for x in range(len(R))] for y in range(len(R))] 
    # temp = []
    # for k in range(len(R)):
    #     for i in range(len(R)):
    #         for j in range(len(R)):
    #             temp.append(max(min(R[k][i] , R[i][j])))
    #         print(temp)    
        
            
TransetiveMatrix = RoR(table)

def calcAlphacut(cut , table):
    alpha = []
    non = []
    for i in range(len(table)):
        if r[0][i]<cut:
            alpha.append(i)
        else:
            non.append(i)
    
    print('the alpha for',cut,'is = ')
    print(alpha)
    print('the alpha for none is = ')
    print(non)


calcAlphacut(0.32,TransetiveMatrix)
calcAlphacut(0.35,TransetiveMatrix)
calcAlphacut(0.4,TransetiveMatrix)
calcAlphacut(0.45,TransetiveMatrix)
calcAlphacut(0.48,TransetiveMatrix)
calcAlphacut(0.8,TransetiveMatrix)

print('matrix of fuzzy compatibility relation')
print(table)

print('table of fuzzy transitive closer ')
TC = np.array(TransetiveMatrix)
print(TC)