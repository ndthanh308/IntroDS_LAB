import pandas as pd
import numpy as np  # noqa
import matplotlib.pyplot as plt

df_groups = pd.read_csv("C:\\Users\\pgall\\OneDrive - The University of Chicago\\Github\\zemax_tools\\S4cam\\groupedCameras\\TMP\\current_designs\\8mm_clearance_elliptical_ap\\groups_info\\85cam_groups.csv")  # noqa
df_fn_elliptical = pd.read_csv('C:\\Users\\pgall\\OneDrive - The University of Chicago\\Github\\zemax_tools\\S4cam\\groupedCameras\\TMP\\current_designs\\8mm_clearance_elliptical_ap\\f_numbers\\f_numbers.csv')  # noqa
df_fn_circular = pd.read_csv('C:\\Users\\pgall\\OneDrive - The University of Chicago\\Github\\zemax_tools\\S4cam\\groupedCameras\\TMP\\current_designs\\8mm_clearance_circular_ap\\f_numbers\\f_numbers.csv')  # noqa

z_ellip = df_fn_elliptical.f_num.values
z_circ = df_fn_circular.f_num.values

s = pd.Series(z_ellip)
table = s.describe(percentiles=[.10, .50, .90])
table = table.to_latex(float_format="%1.2f")
table = table.replace('\\toprule', '')
table = table.replace('\\bottomrule', '')
table = table.replace('\\midrule', '')
table = table.replace('\n', " ")
table = table.replace(' 0 ', '')
table = table.replace("85.00", "85")


plt.rcParams.update({
    "text.usetex": True})
plt.rcParams.update({"font.size": 18})
plt.figure(figsize=[6.4, 6.0])
bins = np.arange(2.00, 2.5, 0.025)
plt.hist(z_circ,
         histtype='step', color='gray',
         bins=bins, label='circular stop',
         lw=2, ls='--', alpha=0.8)
plt.hist(z_ellip,
         histtype='step', color='black',
         bins=bins, label='elliptical stop',
         lw=2)
plt.xlabel('$f/\\#$')
plt.ylabel('Camera count [-]')
plt.title('Camera f-numbers')
plt.figtext(0.79, 0.53, table, fontsize=12)
plt.legend()

# plt.savefig('cam_fnumbers_hist.png', dpi=150)
plt.tight_layout()
plt.savefig('cam_fnumbers_hist.pdf', dpi=150)
plt.close()
