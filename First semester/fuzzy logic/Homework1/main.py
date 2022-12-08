import numpy as np
import scipy.io as sc
test=sc.loadmat('emnist-digits-150.mat')

a=test.get('dataset')

image=a[0][0][0][0][0][0]
label=a[0][0][0][0][0][1]
image=np.array(image)
label=np.array(label)
count=0
f=[]
for i in range(len(image)):
    for j in range(len(image[0])):
         if  image[i][j]>0:
            count=count+image[i][j]/256
    f.append(count/784)
    count=0
f=np.array(f)

f1=[]
count=0
for i in range(len(image)):
    for j in range(len(image[0])):
         if  image[i][j]>0:
            count=count+1
    f1.append(count/784)
    count=0
f1=np.array(f1)

f2=[]
count=0
for i in range(len(image)):
    for j in range(len(image[0])):
         if  image[i][j]==0:
            count=count+1
    f2.append(count/784)
    count=0
f2=np.array(f2)


f4=[]

for i in range(len(f)):
    f4.append(3*(f[i]**2+f1[i]**2)/(f[i]+f1[i]+f2[i]))

r=[]
for i in range(150):
    r.append([])
    for j in range(150):
        r[i].append(1-(abs(f4[i]-f4[j])))
k=True
rt=r
r2=r
while k:
    r=rt
    r2=r
    max1=0
    for k in range(150):
        for i in range(150):
            max=0
            for j in range(150):
                 if min(r[k][i],r[i][j])>max1:
                    max1=min(r[k][j],r[j][i])
                    rt[k][j]=max1


    if r2==rt:
        k=False
list0=[]
list1=[]
for i in range(150):
   if r[0][i]<0.75:
       list0.append(i)
   else:
       list1.append(i)
print('alfa cut<75')
print(list0)
print(list1)
list3=[]
list4=[]
for i in range(len(list0)):
    if r[list0[0]][list0[i]]<0.85:
        list3.append(list0[i])
    else:
        list4.append(list0[i])
list5=[]
list6=[]
for i in range(len(list1)):
    if r[list1[0]][list1[i]]<0.85:
        list5.append(list1[i])
    else:
        list6.append(list1[i])
print('alfa cut<85')
print(list3)
print(list4)
print(list5)
print(list6)
list7=[]
list8=[]
for i in range(len(list3)):
    if r[list3[0]][list3[i]]<0.9:
        list7.append(list3[i])
    else:
        list8.append(list3[i])
list9=[]
list10=[]
for i in range(len(list4)):
    if r[list4[0]][list4[i]]<0.9:
        list9.append(list4[i])
    else:
        list10.append(list4[i])
list11=[]
list12=[]
for i in range(len(list5)):
    if r[list5[0]][list5[i]]<0.9:
        list11.append(list5[i])
    else:
        list11.append(list5[i])
list13=[]
list14=[]
for i in range(len(list6)):
    if r[list6[0]][list6[i]]<0.9:
        list13.append(list6[i])
    else:
        list14.append(list6[i])

print('alfa cut<0.9')
print(list7)
print(list8)
print(list9)
print(list10)
print(list11)
print(list12)
print(list13)
print(list14)



list15=[]
list16=[]
list17=[]
list18=[]
list19=[]
list20=[]
list21=[]
list22=[]
for i in range(len(list10)):
    if r[list10[0]][list10[i]]<0.95:
        list15.append(list10[i])
    else:
        list16.append(list10[i])

for i in range(len(list11)):
    if r[list11[0]][list11[i]]<0.95:
        list17.append(list11[i])
    else:
        list18.append(list11[i])
for i in range(len(list12)):
    if r[list12[0]][list12[i]]<0.95:
        list19.append(list12[i])
    else:
        list20.append(list12[i])
print('alfa cut<0.95')
print(list13)
print(list14)
print(list15)
print(list16)
print(list17)
print(list18)
print(list19)
print(list20)
list23=[]
list24=[]
for i in range(len(list14)):
    if r[list14[0]][list14[i]]<0.97:
        list23.append(list14[i])
    else:
        list24.append(list14[i])
list25=[]
list26=[]
for i in range(len(list15)):
    if r[list15[0]][list15[i]]<0.97:
        list25.append(list15[i])
    else:
        list26.append(list15[i])
list27=[]
list28=[]
for i in range(len(list16)):
    if r[list16[0]][list16[i]]<0.97:
        list27.append(list16[i])
    else:
        list28.append(list16[i])
list29=[]
list30=[]
for i in range(len(list17)):
    if r[list17[0]][list17[i]]<0.97:
        list29.append(list17[i])
    else:
        list30.append(list17[i])
list31=[]
list32=[]
for i in range(len(list18)):
    if r[list18[0]][list18[i]]<0.97:
        list31.append(list18[i])
    else:
        list32.append(list18[i])
print('alfa cut<0.97')
print(list23)
print(list24)
print(list25)
print(list26)
print(list27)
print(list28)
print(list29)
print(list30)
print(list31)
print(list32)
print(list7)
print(list8)



