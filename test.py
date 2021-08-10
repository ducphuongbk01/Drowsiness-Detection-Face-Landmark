import numpy as np
lst3 = [[5,6,7],[8,9,10]]
lst = np.array([1,2,3])
lst2 = np.array([1,2,3])
lst = np.concatenate((lst, lst2))
lst4 = np.array(lst3)
lst5 = [lst, lst2]

print (lst5[1])