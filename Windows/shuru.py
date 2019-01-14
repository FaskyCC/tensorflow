import csv
import pandas as pd

ref = {'0':'daisy','1':'dandelion','2':'rose','3':'sunflower','4':'tulip'}
lines=[line.strip().split() for line in open(r'C:/Users/Fa/Desktop/res.txt')]
from tkinter import _flatten
a = list(_flatten(lines))
b = []
for num in a:
    b.append(ref[num])


# # print(a)
name = ['Expected']
test = pd.DataFrame(columns=name, data=b)
# print(test)
test.to_csv(r'C:/Users/Fa/Desktop/Test3/res.csv')


# with open("F:/flower/output.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(a)
