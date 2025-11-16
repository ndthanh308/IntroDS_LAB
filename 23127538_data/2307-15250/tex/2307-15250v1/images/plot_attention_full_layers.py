import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch 
from matplotlib.cm import ScalarMappable
import os.path as osp

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
    if alpha_weight  == []:
        rel_linspace = np.array([0])
    else:
        rel_linspace = np.linspace(0.0005, 0.163, len(alpha_weight))
    out = {}
    print(max(alpha_weight))
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
    if len(alpha_weight) == 0:
        minmax = [0, 0]
    else:
        minmax = [min(alpha_weight), max(alpha_weight)]
    return out_list, colors_dict, colors, minmax


if __name__ == "__main__":
    aspect = 1.1
    head_num = 0
    alpha_threshold = 0.0005
    linewidth = 0.8
    num_variants = 3
    path = "indoor6/scene1"
    o_path = "BKC_westwing/seq2"
    infor1 = [pd.read_csv("0_n1024.txt", header =None, sep =" "), osp.join(path, "21-frame000110.png"), osp.join(path, "0_n1024/Tensor1.pt"), osp.join(path, "0_n1024/Tensor3.pt"),osp.join(path, "0_n1024/Tensor5.pt")]
    infor2 = [pd.read_csv("20_n1024.txt", header =None, sep =" "), osp.join(path, "22-frame000500.png"), osp.join(path, "20_n1024/Tensor1.pt"), osp.join(path, "20_n1024/Tensor3.pt"),osp.join(path, "20_n1024/Tensor5.pt")]
    infor3 = [pd.read_csv("75_n1024.txt", header =None, sep =" "), osp.join(o_path, "frame1125.png"), osp.join(o_path, "75_n1024/Tensor1.pt"), osp.join(o_path, "75_n1024/Tensor3.pt"),osp.join(o_path, "75_n1024/Tensor5.pt")]
    infor4 = [pd.read_csv("84_n1024.txt", header =None, sep =" "), osp.join(o_path, "frame1275.png"), osp.join(o_path, "84_n1024/Tensor1.pt"), osp.join(o_path, "84_n1024/Tensor3.pt"), osp.join(o_path, "84_n1024/Tensor4.pt"),osp.join(o_path, "84_n1024/Tensor5.pt")]
    # infor2 = [pd.read_csv("99.txt", header =None, sep =" "), "KingsCollege/seq2/frame00040.png"]
    # infor3 = [pd.read_csv("32.txt", header =None, sep =" "), "BKC_westwing/seq5/frame0525.png"]
    # infor4 = [pd.read_csv("70.txt", header =None, sep =" "), "BKC_westwing/seq7/frame0900.png"]
    # infor5 = [pd.read_csv("20.txt", header =None, sep =" "), "indoor6/scene1/22-frame000500.png"]
    point_size = 0.5
    list_infor =[infor1, infor2, infor3]
    length = len(list_infor)
    # figure, plot = plt.subplots(3, 4, gridspec_kw={'hspace': 0, 'wspace': 0.02})
    figure, plot = plt.subplots(3, num_variants, gridspec_kw={'hspace': 0, 'wspace': 0.02})
    fig, ax = plt.subplots()
    figure.set_figheight(6.15)
    figure.set_figwidth(9.6)

    for i in range(length):
        # i = 2
        file = list_infor[i][1]
        att_tensor = []
        for ii in range(num_variants):
             
            att_tensor.append(torch.squeeze(torch.load(list_infor[i][ii+2]))[head_num,:,:].detach().cpu().numpy())
            

        print(len(att_tensor))
        # print(sum(att_tensor[11,:]))
        # raise
        idx = 47
        if i> 0:
            idx = 142
        if i > 1:
            idx = 138
            # alpha_threshold = 0.001
        # uncertainty = list(list_infor[i][0].iloc[:,2])
        # uncer_print = {i:[uncertainty[i], att_tensor[idx, i]] for i in range(len(uncertainty))}

        list_of_connection = []
        colors_dict = []
        for ii in range(num_variants):
            tmp1, tmp2, _, _ = get_list_of_connection(alpha_threshold, att_tensor[ii][idx,:])
            list_of_connection.append(tmp1)
            colors_dict.append(tmp2)
        # print(list_of_connection)
        # print(uncer_print)
        # print(uncertainty[idx])
        # print(sum(att_tensor[idx,:]))

        # plt.plot(att_tensor[0][idx,:])
        # plt.show()
        # raise
        # x = np.arange(len(att_tensor))
        # # fig, ax = plt.subplots()
        # plot[0].scatter(x, att_tensor[idx,:], color=uncertainty)
        # plot[0].plot(x, att_tensor[idx,:])
        # plt.show()
        # raise
        for ii in range(num_variants):
            plot[i,ii].imshow(cv2.cvtColor(read_image(file), cv2.COLOR_BGR2RGB))
        
        points2D_infor = list_infor[i][0].iloc[:,:2].to_numpy()
        print(points2D_infor.shape)
        uncertainty = list(list_infor[i][0].iloc[:,2])

        for ii in range(num_variants):
            plot[i,ii].scatter(points2D_infor[:,0], points2D_infor[:,1], c=uncertainty, marker='+', s = point_size)
            for cores_idx in list_of_connection[ii]:
                x = [points2D_infor[idx,0], points2D_infor[cores_idx,0]]
                y = [points2D_infor[idx,1], points2D_infor[cores_idx,1]]
                plot[i,ii].plot(x,y, color = colors_dict[ii][cores_idx], linewidth = linewidth)
            plot[i,ii].plot(points2D_infor[idx,0], points2D_infor[idx,1], c='b', marker='o', markersize = point_size*8)
        
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

        for ii in range(num_variants):
            plot[i,ii].set_xticks([])
            plot[i,ii].set_yticks([])
            plot[i,ii].set(aspect=aspect)

        # plot[2,i].text(0.87, 0.011, file, size = 4, color = "w", horizontalalignment='center', verticalalignment='center', transform=plot[2,i].transAxes)




    cols = ['Layer 1', 'Layer 3' , 'Layer 5']
    rows = ['SuperPoint', 'D2S(ours)']

    for ax, col in zip(plot[0], cols):
        ax.set_title(col, fontsize=15)

    # for ax, row in zip(plot[:,0], rows):
    #     ax.set_ylabel(row, rotation=90, size='large', labelpad=1)

    # sm = ScalarMappable(cmap=plt.cm.Blues)
    # cb = fig.colorbar(sm, shrink=0.5)
    # cb.set_ticks([])

    fig.tight_layout()
    figure.savefig('attention_full.pdf', dpi=150, bbox_inches='tight')
    # fig.savefig('attention.pdf', dpi=150)
