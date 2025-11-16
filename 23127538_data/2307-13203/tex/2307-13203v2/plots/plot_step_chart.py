import matplotlib.pyplot as plt
import numpy as np

def plotStepChartData2(step_graph, num_checks):

    start_time = step_graph[0][1]
    norm_stepping = [(data_pt[0], data_pt[1] - start_time) for data_pt in step_graph]

    for i in range(len(norm_stepping) - 1):
        curr_pt = norm_stepping[i]
        next_pt = norm_stepping[i+1]
        time_spent = next_pt[1] - curr_pt[1]
        level, discs_chk, conf_chk = num_checks[i]
        dist_disc = (discs_chk/(discs_chk+conf_chk))*time_spent
        line1, = plt.plot([curr_pt[1], curr_pt[1] + dist_disc], [curr_pt[0], curr_pt[0]], 'b', linewidth=4)
        line2, = plt.plot([curr_pt[1] + dist_disc, next_pt[1]], [curr_pt[0], curr_pt[0]], 'r', linewidth=4)
        plt.plot([next_pt[1], next_pt[1]], [curr_pt[0], next_pt[0]], 'k', linewidth=0.5, linestyle='dotted')

        plt.legend([line1, line2], ['discriminations', 'conflations'])
    
    # plt.show()

def annotate(text, xy, textpos):
    plt.annotate(text, xy, xytext=textpos, textcoords='axes fraction',
            arrowprops=dict(facecolor='black', arrowstyle='-', shrinkB=10, linestyle='-.', alpha=0.5),
            fontsize=12,
            horizontalalignment='right', verticalalignment='top')

plt.figure(figsize=(8,6))
plt.xlabel('time (s)')
plt.ylabel('sensor set size')

npzfile = np.load("step_chart_data_vanilla.npz")
stepping_gr, num_checks_1 = npzfile['arr_0'], npzfile['arr_1']
plotStepChartData2(stepping_gr, num_checks_1)
annotate('vanilla', ((stepping_gr[len(stepping_gr)//2][1] + stepping_gr[len(stepping_gr)//2+1][1])/2, stepping_gr[len(stepping_gr)//2][0]), (0.8, 0.7))

npzfile = np.load("step_chart_data_optimized.npz")
stepping_gr, num_checks_3 = npzfile['arr_0'], npzfile['arr_1']
plotStepChartData2(stepping_gr, num_checks_3)
annotate('optimized', (stepping_gr[-1][1], stepping_gr[-1][0]), (0.4, 0.1))

plt.show()