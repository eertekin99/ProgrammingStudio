from PIL import *
from PIL import Image, ImageDraw
import numpy as np

def main():

    img = Image.open('numbers2.PNG')
    #img.show()
    img_gray = img.convert('L')  # converts the image to grayscale image
    ONE = 150
    a = np.asarray(img_gray)  # from PIL to np array
    a_bin = threshold(a, 100, ONE, 0)
    im = Image.fromarray(a_bin)  # from np array to PIL format
    #im.show()

    im_label, colour_label, table = blob_coloring_8_connected(a_bin, ONE)

    #new_img2 = np2PIL_color(colour_label)
    #new_img2.show()

    new_img3 = draw_rectangles(table, img)
    new_img3.show()


def binary_image(nrow,ncol,Value):
    x, y = np.indices((nrow, ncol))
    mask_lines = np.zeros(shape=(nrow,ncol))

    x0, y0, r0 = 30, 30, 10
    x1, y1, r1 = 70, 30, 10


    for i in range (50, 70):
        mask_lines[i][i] = 1
        mask_lines[i][i + 1] = 1
        mask_lines[i][i + 2] = 1
        mask_lines[i][i + 3] = 1
        mask_lines[i][i + 6] = 1
        mask_lines[i-20][90-i+1] = 1
        mask_lines[i-20][90-i+2] = 1
        mask_lines[i-20][90-i+3] = 1


    #mask_circle1 = np.abs((x - x0) ** 2 + (y - y0) ** 2 - r0 ** 2 ) <= 5
    mask_square1 = np.fmax(np.absolute( x - x1), np.absolute( y - y1)) <= r1
    #mask_square2 = np.fmax(np.absolute( x - x2), np.absolute( y - y2)) <= r2
    #mask_square3 = np.fmax(np.absolute( x - x3), np.absolute( y - y3)) <= r3
    #mask_square4 =  np.fmax(np.absolute( x - x4), np.absolute( y - y4)) <= r4
    #imge = np.logical_or ( np.logical_or(mask_lines, mask_circle1), mask_square1) * Value
    imge = np.logical_or(mask_lines, mask_square1) * Value
    #imge = np.logical_or(mask_lines, mask_circle1) * Value

    return imge

def np2PIL(im):
    print("size of arr: ", im.shape)
    img = Image.fromarray(im, 'RGB')
    return img

def np2PIL_color(im):
    print("size of arr: ", im.shape)
    img = Image.fromarray(np.uint8(im))
    return img

def threshold(im, T, LOW, HIGH):
    (nrows, ncols) = im.shape
    im_out = np.zeros(shape = im.shape)
    for i in range(nrows):
        for j in range(ncols):
            if abs(im[i][j]) <  T :
                im_out[i][j] = LOW
            else:
                im_out[i][j] = HIGH
    return im_out


def blob_coloring_8_connected(bim, ONE):
    max_label = int(10000)
    nrow = bim.shape[0]
    ncol = bim.shape[1]
    #print("nrow, ncol", nrow, ncol) Don't need for now.
    im = np.zeros(shape=(nrow, ncol), dtype=int)
    a = np.zeros(shape=max_label, dtype=int)
    a = np.arange(0, max_label, dtype=int)
    color_map = np.zeros(shape=(max_label, 3), dtype=np.uint8)
    color_im = np.zeros(shape=(nrow, ncol, 3), dtype=np.uint8)

    for i in range(max_label):
        np.random.seed(i)
        color_map[i][0] = np.random.randint(0, 255, 1, dtype=np.uint8)
        color_map[i][1] = np.random.randint(0, 255, 1, dtype=np.uint8)
        color_map[i][2] = np.random.randint(0, 255, 1, dtype=np.uint8)

    k = 0
    for i in range(nrow):
        for j in range(ncol):
            im[i][j] = max_label
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
                c = bim[i][j]
                l = bim[i][j - 1]
                u = bim[i - 1][j]
                label_u = im[i - 1][j]
                label_l = im[i][j - 1]
                label_ul = im[i-1][j-1]
                label_ur = im[i+1][j-1]

                im[i][j] = max_label
                if c == ONE:
                    min_label = min(label_u, label_l, label_ur, label_ul)
                    if min_label == max_label:
                        k += 1
                        im[i][j] = k
                    else:
                        im[i][j] = min_label
                        if min_label != label_u and label_u != max_label  :
                            update_array(a, min_label, label_u)
                        if min_label != label_l and label_l != max_label  :
                            update_array(a, min_label, label_l)
                        if min_label != label_ul and label_ul != max_label :
                            update_array(a, min_label, label_ul)
                        if min_label != label_ur and label_ur != max_label :
                            update_array(a, min_label, label_ur)
                else:
                    im[i][j] = max_label

    # final reduction in label array
    for i in range(k+1):
        index = i
        while a[index] != index:
            index = a[index]
        a[i] = a[index]

    #second pass to resolve labels and show label colors
    for i in range(nrow):
        for j in range(ncol):

            if bim[i][j] == ONE:
                im[i][j] = a[im[i][j]]
                if im[i][j] == max_label:
                    im[i][j] == 0
                    color_im[i][j][0] = 0
                    color_im[i][j][1] = 0
                    color_im[i][j][2] = 0
                color_im[i][j][0] = color_map[im[i][j], 0]
                color_im[i][j][1] = color_map[im[i][j], 1]
                color_im[i][j][2] = color_map[im[i][j], 2]

    ## labelling ##

    counter = -1     #Don't care background's label
    list = []
    for i in range(nrow):
        for j in range(ncol):
            if list.__contains__(im[i][j]):
                pass
            else:
                list.append(im[i][j])
                counter = counter + 1

    list.pop(0)
    print("The number of digits in this picture =", len(list))

    ## Table format is = "label-min_i-min_j-max_i-max_j" ##
    ## Filling table with labels' properties ##

    table = np.zeros((len(list), 5))    #label/min_i/min_j/max_i/max_j

    for a in range(len(list)):
        table[a][0] = list[a]

    for ind in range(len(list)):
        for i in range(nrow):
            for j in range(ncol):
                if im[i][j] == int(table[ind][0]):
                    if int(table[ind][1]) > i or int(table[ind][1]) == 0:
                        table[ind][1] = i
                    if int(table[ind][2]) > j or int(table[ind][2]) == 0:
                        table[ind][2] = j
                    if int(table[ind][3]) < i or int(table[ind][3]) == 0:
                        table[ind][3] = i
                    if int(table[ind][4]) < j or int(table[ind][4]) == 0:
                        table[ind][4] = j

    return im, color_im, table


def update_array(a, label1, label2):
    index = lab_small = lab_large = 0
    if label1 < label2:
        lab_small = label1
        lab_large = label2
    else:
        lab_small = label2
        lab_large = label1
    index = lab_large
    while index > 1 and a[index] != lab_small:
        if a[index] < lab_small:
            temp = index
            index = lab_small
            lab_small = a[temp]
        elif a[index] > lab_small:
            temp = a[index]
            a[index] = lab_small
            index = temp
        else: #a[index] == lab_small
            break

    return

def draw_rectangles (array, image):
    draw = ImageDraw.Draw(image)
    for a in range(len(array)):
        draw.rectangle([int(array[a][2])-2, int(array[a][1])-2, int(array[a][4])+2, int(array[a][3])+2], width=2, outline="#ff0000")
    return image

if __name__=='__main__':
    main()
