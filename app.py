from imblearn.under_sampling import TomekLinks
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from numpy import where

_FILENAME = "Full data.xlsx"

# read xlsx file and save to a file as csv.
data_xlsx = pd.read_excel(_FILENAME, dtype=str, index_col=None)
data_xlsx.to_csv('data.csv', encoding='utf-8', index=False)

# read from the csv file.
data = pd.read_csv("data.csv", header=0, encoding='utf-8')
data_ = data.copy()
print( "data : ",  data.Classes.value_counts() )
counter = data['Classes']
# scatter_(data[['Phase A', 'Phase B', 'Phase C']], counter )

plt.figure()
pd.value_counts( data['Classes']).plot.bar(title= "Before Tomek Link")
plt.show()

# define sampling method
method = TomekLinks()
d, t = method.fit_resample( data[['Phase A', 'Phase B', 'Phase C']], data['Classes'] )
# print( t.value_counts() )

# run many times to achieve good balance
for i in range(5):
    d, t = method.fit_resample( d, t )


print( t.value_counts() )
plt.figure()
pd.value_counts(t).plot.bar(title="After Tomek Link")
plt.show()


new_ = d.copy()
new_['Classes'] = t
new_.to_csv("result.csv", index=False, encoding='utf-8')
new_.to_excel("result.xlsx")


print("done...")