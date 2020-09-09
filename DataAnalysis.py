

import matplotlib.pyplot as plt
import numpy as np

data16 = [0.3229178935154512, 0.8667891621589661]
data24 = [0.29797789505571914, 0.8813051581382751]
data32 = [0.3192911538362969, 0.8689129948616028]

plt.figure(figsize=[3,6])
x = data16
y = data24
z = data32
plt.hist([x, y,z])