import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('threshold_test', sep='\s+', header=None)
data_2 = pd.read_csv('threshold_test_slope_1_5', sep='\s+', header=None)
data_3 = pd.read_csv('threshold_test_slope_2_5', sep='\s+', header=None)

thresh = data[0]
radius = data[1]
thresh_2 = data_2[0]
radius_2 = data_2[1]
thresh_3 = data_3[0]
radius_3 = data_3[1]

plt.figure()
plt.scatter(thresh, radius, marker='+', label='isothermal')
plt.scatter(thresh_2[:22], radius_2[:22], marker='x', label='power law 1.5')
plt.scatter(thresh_3, radius_3, marker='.', label='power law 2.5')
plt.xlabel('Threshold')
plt.ylabel('Einstein Radius')
plt.legend()
plt.savefig('/Users/dgmt59/Documents/Plots/Crit_curves_resolution/threshold.png', bbox_inches='tight', dpi=300)


plt.show()