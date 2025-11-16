import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch 
from matplotlib.cm import ScalarMappable

def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        print("image   ", image)
        raise ValueError(f'Cannot read image {path}.')
    return image

def get_colors_dict(out_list, alpha_weight):
    linspace = np.linspace(0.55,1, len(alpha_weight))
    colors = plt.cm.Blues(linspace)
    rel_linspace = np.linspace(min(alpha_weight), max(alpha_weight), len(alpha_weight))
    out = {}
    for i in range(len(out_list)):
        value = alpha_weight[i]
        idx = (np.abs(rel_linspace-value)).argmin()
        out[out_list[i]] =  colors[idx]
    return out, colors

def get_list_of_connection(threshold, att_row):
    out_list = [] 
    alpha_weight = []
    for i in range(len(att_row)):
        if att_row[i] > threshold:
            out_list.append(i)
            alpha_weight.append(att_row[i])
    colors_dict, colors =  get_colors_dict(out_list, alpha_weight)
    return out_list, colors_dict, colors, [min(alpha_weight), max(alpha_weight)]


if __name__ == "__main__":
    aspect = 1.1
    head_num = 0
    alpha_threshold = 0.003
    linewidth = 0.8
    infor1 = [pd.read_csv("0_n1024.txt", header =None, sep =" "), "indoor6/scene1/21-frame000110.png","indoor6/scene1/0_n1024/Tensor3.pt", "indoor6/scene1/0_n1024/Tensor4.pt"]
    infor2 = [pd.read_csv("0_n1024.txt", header =None, sep =" "), "indoor6/scene1/21-frame000110.png","indoor6/scene1/0_n1024/Tensor3.pt", "indoor6/scene1/0_n1024/Tensor4.pt"]
    # infor2 = [pd.read_csv("99.txt", header =None, sep =" "), "KingsCollege/seq2/frame00040.png"]
    # infor3 = [pd.read_csv("32.txt", header =None, sep =" "), "BKC_westwing/seq5/frame0525.png"]
    # infor4 = [pd.read_csv("70.txt", header =None, sep =" "), "BKC_westwing/seq7/frame0900.png"]
    # infor5 = [pd.read_csv("20.txt", header =None, sep =" "), "indoor6/scene1/22-frame000500.png"]
    point_size = 0.5
    list_infor =[infor1, infor2]
    length = len(list_infor)
    # figure, plot = plt.subplots(3, 4, gridspec_kw={'hspace': 0, 'wspace': 0.02})
    figure, plot = plt.subplots(2, 2, gridspec_kw={'hspace': 0, 'wspace': 0.02})
    fig, ax = plt.subplots()
    figure.set_figheight(4.41)
    figure.set_figwidth(7)

    for i in range(length):
        file = list_infor[i][1]

        att_tensor1 = torch.squeeze(torch.load(list_infor[i][2]))[head_num,:,:].detach().cpu().numpy()
        att_tensor2 = torch.squeeze(torch.load(list_infor[i][3]))[head_num,:,:].detach().cpu().numpy()
        print(att_tensor1.shape)
        # print(sum(att_tensor[11,:]))
        # raise
        idx = 47
        if i> 0:
            idx = 11
        # uncertainty = list(list_infor[i][0].iloc[:,2])
        # uncer_print = {i:[uncertainty[i], att_tensor[idx, i]] for i in range(len(uncertainty))}

        list_of_connection1, colors_dict1, _, _  = get_list_of_connection(alpha_threshold, att_tensor1[idx,:])
        list_of_connection2, colors_dict2, _, _  = get_list_of_connection(alpha_threshold, att_tensor2[idx,:])
        
        # print(uncer_print)
        # print(uncertainty[idx])
        # print(sum(att_tensor[idx,:]))
        # plt.plot(att_tensor[idx,:])
        # plt.show()
        # x = np.arange(len(att_tensor))
        # # fig, ax = plt.subplots()
        # plot[0].scatter(x, att_tensor[idx,:], color=uncertainty)
        # plot[0].plot(x, att_tensor[idx,:])
        # plt.show()
        # raise
        plot[i,0].imshow(cv2.cvtColor(read_image(file), cv2.COLOR_BGR2RGB))
        plot[i,1].imshow(cv2.cvtColor(read_image(file), cv2.COLOR_BGR2RGB))
        # plot[2,i].imshow(cv2.cvtColor(read_image(file), cv2.COLOR_BGR2RGB))
        
        points2D_infor = list_infor[i][0].iloc[:,:2].to_numpy()
        print(points2D_infor.shape)
        uncertainty = list(list_infor[i][0].iloc[:,2])

        plot[i,0].scatter(points2D_infor[:,0], points2D_infor[:,1], c=uncertainty, marker='+', s = point_size)
        for cores_idx in list_of_connection1:
            x = [points2D_infor[idx,0], points2D_infor[cores_idx,0]]
            y = [points2D_infor[idx,1], points2D_infor[cores_idx,1]]
            plot[i,0].plot(x,y, color = colors_dict1[cores_idx], linewidth = linewidth)
        plot[i,0].plot(points2D_infor[idx,0], points2D_infor[idx,1], c='b', marker='o', markersize = point_size*8)


        plot[i,1].scatter(points2D_infor[:,0], points2D_infor[:,1], c=uncertainty, marker='+', s = point_size)

        # ax.scatter(points2D_infor[:,0], points2D_infor[:,1], c=uncertainty, marker='+', s = point_size)
        # ax.imshow(cv2.cvtColor(read_image(file), cv2.COLOR_BGR2RGB))
        for cores_idx in list_of_connection2:
            x = [points2D_infor[idx,0], points2D_infor[cores_idx,0]]
            y = [points2D_infor[idx,1], points2D_infor[cores_idx,1]]
            plot[i,1].plot(x,y, color = colors_dict2[cores_idx], linewidth = linewidth)
        plot[i,1].plot(points2D_infor[idx,0], points2D_infor[idx,1], c='b', marker='o', markersize = point_size*8)
        # cb.ax.text(0.45, 1.008, '1')
        # cb.ax.text(0.45, 0.17, '0')
        # ax.set_xticks([])
        # ax.set_yticks([])

        truefalse = [True if x=='lime' else False for x in uncertainty]
        truefalse = np.asarray(truefalse)
        list_of_visible_points = np.where(truefalse == True)
        # print(list_of_visible_points)
        # # raise
        # points2D_infor = points2D_infor[truefalse]
        # plot[2,i].scatter(points2D_infor[:,0], points2D_infor[:,1], c='lime', marker='+', s = point_size)

        scene_name = file.split("/")[0]
        file = file.replace(scene_name + "/", '')

        plot[i,0].set_xticks([])
        plot[i,0].set_yticks([])
        plot[i,1].set_xticks([])
        plot[i,1].set_yticks([])
        # plot[2,i].set_xticks([])
        # plot[2,i].set_yticks([])
        # plot[2,i].text(0.87, 0.011, file, size = 4, color = "w", horizontalalignment='center', verticalalignment='center', transform=plot[2,i].transAxes)


        plot[i,0].set(aspect=aspect)
        plot[i,1].set(aspect=aspect)


    cols = ['Layer 3', 'Layer 4']
    rows = ['SuperPoint', 'D2S(ours)']

    for ax, col in zip(plot[0], cols):
        ax.set_title(col, fontsize=10)

    # for ax, row in zip(plot[:,0], rows):
    #     ax.set_ylabel(row, rotation=90, size='large', labelpad=1)

    # sm = ScalarMappable(cmap=plt.cm.Blues)
    # cb = fig.colorbar(sm, shrink=0.5)
    # cb.set_ticks([])

    fig.tight_layout()
    figure.savefig('attention.pdf', dpi=150, bbox_inches='tight')
    # fig.savefig('attention.pdf', dpi=150)
