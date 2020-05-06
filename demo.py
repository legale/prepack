import pandas as pd
import numpy as np
from prepack import prepack as pp


# Print list in 2 columns
lst = ['first',
       'второй тоже достаточно длинный столбец, его едва можно уместить на экране.',
       'третий',
       'четвертый столбец очень длинный, гораздо длиннее обычного, посмотрим на результат'
       ]

print('print 2 cols p2c:\n', pp.p2c(lst))

# dataframes levenshtein merge
cols = ['col']
a = pd.DataFrame([['top'],['god'],['fog']], columns = cols)
b = pd.DataFrame([['to p'],['mod'],['dog']], columns = cols)

c = pp.levenshtein_merge(a,b,left_on='col', right_on='col')
print('dfa:\n', a)
print('dfb:\n', b)
print('dataframes levenshtein merged:\n', c)