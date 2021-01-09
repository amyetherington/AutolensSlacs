import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('resolution_test', sep='\s+', header=None)
data_2 = pd.read_csv('resolution_test_slope_1_5', sep='\s+', header=None)
data_3 = pd.read_csv('resolution_test_slope_2_5', sep='\s+', header=None)

res = data[0]
radius = data[1]
res_2 = data_2[0]
radius_2 = data_2[1]
res_3 = data_3[0]
radius_3 = data_3[1]

plt.figure()
plt.scatter(res, radius, marker='+', label='isothermal')
plt.scatter(res_2[:22], radius_2[:22], marker='x', label='power law 1.5')
plt.scatter(res_3, radius_3, marker='.', label='power law 2.5')
plt.xlabel('Resolution')
plt.ylabel('Einstein Radius')
plt.legend()
plt.savefig('/Users/dgmt59/Documents/Plots/Crit_curves_resolution/res.png', bbox_inches='tight', dpi=300)


plt.figure()
plt.scatter(res_2[-15:], radius_2[-15:], marker='x')
plt.xlabel('Resolution')
plt.ylabel('Einstein Radius')
plt.show()
