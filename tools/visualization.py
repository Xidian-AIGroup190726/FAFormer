"""
integrate model output patches into the original remote sensing size.
"""
import argparse
import os
import mmcv
import numpy as np
import gdal
import sys
from PIL import Image, ImageDraw
import cv2
from numba import jit
sys.path.append('visualization')

def transCannal(out_attn):
    out_attn = out_attn.transpose(2, 0, 1)    # (c, row, col)
    out_attn = out_attn.astype(int)
    img = linear(out_attn)
    if img.shape[0] in [4, 8]:  # (row, col, c)
        img = img[(2, 1, 0), :, :]
        img = img.transpose(1, 2, 0)
    elif img.shape[0] is 1:
        _, h, w = img.shape
        img = img.reshape(h, w)

    # 转换为整型
    img = img.astype(np.uint8)
    return img

def np2img(np_arry):
    img = np.array(np_arry, dtype=np.uint8)
    img = Image.fromarray(np_arry)
    return img

@jit(nopython=True)
def linear(data):
    img_new = np.zeros(data.shape)
    sum_ = data.shape[1] * data.shape[2]
    for i in range(0, data.shape[0]):
        num = np.zeros(5000)
        prob = np.zeros(5000)
        for j in range(0, data.shape[1]):
            for k in range(0, data.shape[2]):
                num[data[i, j, k]] = num[data[i, j, k]] + 1
        for tmp in range(0, 5000):
            prob[tmp] = num[tmp] / sum_
        min_val = 0
        max_val = 0
        min_prob = 0.0
        max_prob = 0.0
        while min_val < 5000 and min_prob < 0.2:
            min_prob += prob[min_val]
            min_val += 1
        while True:
            max_prob += prob[max_val]
            max_val += 1
            if max_val >= 5000 or max_prob >= 0.98:
                break
        for m in range(0, data.shape[1]):
            for n in range(0, data.shape[2]):
                if data[i, m, n] > max_val:
                    img_new[i, m, n] = 255
                elif data[i, m, n] < min_val:
                    img_new[i, m, n] = 0
                else:
                    img_new[i, m, n] = (data[i, m, n] - min_val) / (max_val - min_val) * 255
    return img_new

# GF-2:      col=22 row=13 attn_block=22   rect_width=4  scale=5.5
# GF-1:      col=21 row=21 attn_block=102   rect_width=4  scale=7
# QB:        col=8 row=4   attn_block=15   rect_width=8  scale=1
# GF1-to-WV2:col=6 row=19  attn_block=16    rect_width=16 scale=1

def parse_args():
    parser = argparse.ArgumentParser(description='integrate patches into one big image')
    parser.add_argument('-z', '--data_type', required=False, default=r'GF-1', help='data type')
    parser.add_argument('-d', '--dir', required=False, default='', help='directory of input patches')
    parser.add_argument('-t', '--dst', required=False, default=r'result_png', help='directory of save path')
    parser.add_argument('-c', '--col', required=False, default=21, type=int, help='how many columns')
    parser.add_argument('-r', '--row', required=False, default=21, type=int, help='how many rows')
    parser.add_argument('--ms_chan', default=4, type=int, help='how many channels of MS')
    parser.add_argument('-p', '--patch_size', default=400, type=int, help='patch size')
    parser.add_argument('-a', '--attn_block', default=102, type=int, help='the number of patch for show')
    parser.add_argument('-l', '--rect_width', default=4, type=float, help='the rate of image dowm-sample scale')
    parser.add_argument('-s', '--scale', default=7, type=float, help='the rate of image dowm-sample scale')

    return parser.parse_args()

def Draw(args, model, src_path, dst_path):
    patch_size = args.patch_size
    # 确定图像尺寸
    y_size = patch_size // 2 * (args.row - 1) + patch_size
    x_size = patch_size // 2 * (args.col - 1) + patch_size
    out = np.zeros(shape=[y_size, x_size, args.ms_chan], dtype=np.float32)
    cnt = np.zeros(shape=out.shape, dtype=np.float32)
    # print(out.shape)

    # 组成tif块
    i = 0
    y = 0
    x_attn = 0
    y_attn = 0
    img_attn = None
    for _ in range(args.row):
        x = 0
        for __ in range(args.col):
            ly = y
            ry = y + patch_size
            lx = x
            rx = x + patch_size
            cnt[ly:ry, lx:rx, :] = cnt[ly:ry, lx:rx, :] + 1
            # img = f'{args.dir}/{i}_mul_hat.tif'
            img = None
            if model == "GT":
                img = f'{src_path}/{i}_mul.tif'
            elif model == "PAN":
                img = f'{src_path}/{i}_pan.tif'
            elif model == "LRMS":
                img = f'{src_path}/{i}_lr_u.tif'
            else:
                img = f'{src_path}/{i}_mul_hat.tif'
            img = gdal.Open(img).ReadAsArray()

            if model == "PAN":
                img = np.expand_dims(img, axis=0)
                img = np.repeat(img, args.ms_chan, axis=0)

            img = img.transpose(1, 2, 0)
            img = np.array(img, dtype=np.float32)
            out[ly:ry, lx:rx, :] = out[ly:ry, lx:rx, :] + img
            if i == args.attn_block:
                x_attn = x
                y_attn = y
                img_attn = img  # (row, col, c)

            i = i + 1
            x = x + patch_size // 2
        y = y + patch_size // 2
    out = out / cnt # (row, col, c)

    # 图像与关键点放缩
    h, w = out.shape[:2]
    size = (int(w // args.scale), int(h // args.scale))
    out_attn = cv2.resize(out, size, interpolation=cv2.INTER_CUBIC)
    x_attn, y_attn, patch_size_attn = int(x_attn // args.scale), int(y_attn // args.scale), int(patch_size // args.scale),

    # 通道转换
    img_attn = transCannal(img_attn)
    out_attn = transCannal(out_attn)

    # 保存大图
    dst_big_path = f'{dst_path}/big'
    mmcv.mkdir_or_exist(dst_big_path)
    dst_file = dst_big_path + '/' + args.data_type + '-' + model + '.png'
    np2img(out_attn).save(dst_file)

    # 保存小图
    dst_small_path = f'{dst_path}/small'
    mmcv.mkdir_or_exist(dst_small_path)
    dst_file = dst_small_path + '/' + args.data_type + '-' + model + '.png'
    np2img(img_attn).save(dst_file)

    # 保存标出小图位置的图像
    img = np2img(out_attn)
    draw = ImageDraw.Draw(img)
    attn_points = [(x_attn, y_attn), (x_attn+patch_size_attn, y_attn+patch_size_attn)]
    h, w = size
    draw.rectangle(attn_points, outline='yellow', width=args.rect_width)
    dst_locate_path = f'{dst_path}/locate'
    mmcv.mkdir_or_exist(dst_locate_path)
    dst_file = dst_locate_path + '/' + args.data_type + '-' + model + '.png'
    img.save(dst_file)

    # 保存大图+小图(大图右下角显示小图)
    h_attn, w_attn = out_attn.shape[:2]
    out_attn[h_attn-patch_size:h_attn, w_attn-patch_size:w_attn, :] = img_attn
    img = np2img(out_attn)
    draw = ImageDraw.Draw(img)
    attn_points = [(x_attn, y_attn), (x_attn+patch_size_attn, y_attn+patch_size_attn)]
    h, w = size
    attn2_points = [(h-patch_size, w-patch_size), size]
    draw.rectangle(attn_points, outline='yellow', width=args.rect_width)
    draw.rectangle(attn2_points, outline='yellow', width=args.rect_width)
    dst_png_path = f'{dst_path}/png'
    mmcv.mkdir_or_exist(dst_png_path)
    dst_file = dst_png_path + '/' + args.data_type + '-' + model + '.png'
    img.save(dst_file)


    print(f"finish model:{model}")


if __name__ == '__main__':
    # 参数获取
    args = parse_args()
    mmcv.mkdir_or_exist(args.dst)
    dst_path = f'{args.dst}/{args.data_type}'
    mmcv.mkdir_or_exist(dst_path)

    file_path = f'{args.dir}/{args.data_type}'
    for idx, model in enumerate(os.listdir(file_path)):
        if model == 'LR':
            continue
        src_path = f'{file_path}/{model}'
        Draw(args, model, src_path, dst_path)

