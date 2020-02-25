from PIL import *
from PIL import Image, ImageDraw
import numpy as np
import math
import cv2

samples = np.loadtxt("samples.txt")
samplesR = np.loadtxt("samplesR.txt")

def main():
    img = Image.open("ortalamazorluk.png")
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
    #new_img3.show()

    cropped_images = []
    for i in range(len(table)):
        cropped = img.crop((table[i][2], table[i][1], table[i][4], table[i][3]))
        cropped = cropped.resize((21, 21))
        cropped_images.append(cropped)
        #cropped.show()

    label_hu_nums = hu_moments(cropped_images)
    label_R_nums = r_moment(cropped_images)

    # dsize = (21, 21)
    # image = cv2.imread('1.png')
    # image = cv2.resize(image, dsize)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(cv2.moments(image))
    # print(cv2.HuMoments(cv2.moments(image)))

    picture_to_number_hu(samples, label_hu_nums, table, new_img3)
    new_img3.show()

    # picture_to_number_R(samplesR, label_R_nums, table, new_img3)
    # new_img3.show()

#label_hu doldurma kısmı
    # for i in range(len(label_hu_nums)):
    #     label_hu_nums[i][0] = i % 10
    # print(label_hu_nums)
    # data = np.asarray(label_hu_nums)
    # with open("samples.txt", "ab") as f:
    #     np.savetxt(f, data)


    # np.savetxt("samples6343.txt", data)
    #print(np.load("samples.txt"))

#label_R doldurma kısmı
    # for i in range(len(label_R_nums)):
    #     label_R_nums[i][0] = i % 10
    # print(label_R_nums)
    # data = np.asarray(label_R_nums)
    # with open("samplesR.txt", "ab") as f:
    #     np.savetxt(f, data)


    # np.savetxt("samplesR213214.txt", data)
    #print(np.load("samples.txt"))

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


    ## Table format is = "label-min_i-min_j-max_i-max_j" ##

    #new_array = []
    #elements = []
    #elements.append([])
    #elements.append([])

    #for a in range(len(list)):
    #    new_array.append(elements)
    #    elements = []
    #    elements.append([])
    #    elements.append([])
    #    for i in range(table[a][1], table[a][3]):
    #        for j in range(table[a][2], table[a][4]):
    #print(new_array)

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
        draw.rectangle([int(array[a][2])-2, int(array[a][1])-2, int(array[a][4])+2, int(array[a][3])+2], width=1, outline="#ff0000")
    return image

def hu_moments (img_array):
    label_hu_numbers = np.zeros((len(img_array), 8)) #number-Hu1-.....-Hu7

    for num in range(len(img_array)):
        img_gray = img_array[num].convert('L')  # converts the image to grayscale image
        ONE = 1
        a = np.asarray(img_gray)  # from PIL to np array
        a_bin = threshold(a, 100, ONE, 0)
        im = Image.fromarray(a_bin)
        # a_bin = np.asarray(im)
        nrow = a_bin.shape[0]
        ncol = a_bin.shape[1]
        #print(a_bin)
        #print("-----------------")

        # m_00, m_10, m_01 = 0, 0, 0
        #
        # for i in range(nrow):
        #     for j in range(ncol):
        #         m_00 = m_00 + a_bin[i][j]
        #         #print(m_00)
        #         m_10 = m_10 + i*(a_bin[i][j])
        #         m_01 = m_01 + j*(a_bin[i][j])
        #     #print(m_00)
        #
        # x_0 = m_10 / m_00
        # y_0 = m_01 / m_00
        #
        # mu_00, mu_01, mu_10, mu_11, mu_02, mu_20, mu_12, mu_21, mu_03, mu_30 = 0,0,0,0,0,0,0,0,0,0
        #
        # for i in range(nrow):
        #     for j in range(ncol):
        #         mu_00 = mu_00 + a_bin[i][j]
        #         mu_01 = mu_01 + (j - y_0) * a_bin[i][j]
        #         mu_10 = mu_10 + (i - x_0) * a_bin[i][j]
        #         mu_11 = mu_11 + (j - y_0) * (i - x_0) * a_bin[i][j]
        #         mu_02 = mu_02 + ((j - y_0) ** 2) * a_bin[i][j]
        #         mu_20 = mu_20 + ((i - x_0) ** 2) * a_bin[i][j]
        #         mu_12 = mu_12 + (i - x_0) * ((j - y_0) ** 2) * a_bin[i][j]
        #         mu_21 = mu_21 + (j - y_0) * ((i - x_0) ** 2) * a_bin[i][j]
        #         mu_03 = mu_03 + ((j - y_0) ** 3) * a_bin[i][j]
        #         mu_30 = mu_30 + ((i - x_0) ** 3) * a_bin[i][j]
        #     #print(mu_00)
        # n_00, n_01, n_10, n_11, n_02, n_20, n_12, n_21, n_03, n_30, yu = 0,0,0,0,0,0,0,0,0,0,0
        #
        # yu = 1
        # n_00 = mu_00/(mu_00)**yu
        # yu = 3/2.0
        # n_01 = mu_01/(mu_00)**yu
        # n_10 = mu_10/(mu_00)**yu
        # yu = 2
        # n_11 = mu_11/(mu_00)**yu
        # n_02 = mu_02/(mu_00)**yu
        # n_20 = mu_20/(mu_00)**yu
        # yu = 5/2.0
        # n_12 = mu_12/(mu_00)**yu
        # n_21 = mu_21/(mu_00)**yu
        # n_03 = mu_03/(mu_00)**yu
        # n_30 = mu_30/(mu_00)**yu
        #
        # H1, H2, H3, H4, H5, H6, H7 = 0,0,0,0,0,0,0
        #
        # H1 = (n_20 + n_02)
        # H2 = ((n_20 - n_02)**2 + 4 * (n_11)**2)
        # H3 = ((n_30 - 3*n_12)**2 + (3*n_21 - n_03)**2)
        # H4 = ((n_30 + n_12)**2 + (n_21 + n_03)**2)
        # H5 = ((n_30 - 3*n_12)*(n_30 + n_12)*((n_30 + n_12)**2 - 3*(n_21 + n_03)**2) + (3*n_21 - n_03)*(n_21 + n_03)*(3*(n_30 + n_12)**2 - (n_21 + n_03)**2))
        # H6 = ((n_20 - n_02)*((n_30 + n_12)**2 - (n_21 + n_03)**2) + 4*n_11*(n_30 + n_12)*(n_21 + n_03))
        # H7 = ((3*n_21 - n_03)*(n_30 + n_12)*((n_30 + n_12)**2 - 3*(n_21 + n_03)**2) + (3*n_12 - n_30)*(n_21 + n_03)*(3*(n_30 + n_12)**2 - (n_21 + n_03)**2))

        # label_hu_numbers[num][1] = -1 * math.copysign(1.0, H1) * math.log10(abs(H1))
        # label_hu_numbers[num][2] = -1 * math.copysign(1.0, H2) * math.log10(abs(H2))
        # label_hu_numbers[num][3] = -1 * math.copysign(1.0, H3) * math.log10(abs(H3))
        # label_hu_numbers[num][4] = -1 * math.copysign(1.0, H4) * math.log10(abs(H4))
        # label_hu_numbers[num][5] = -1 * math.copysign(1.0, H5) * math.log10(abs(H5))
        # label_hu_numbers[num][6] = -1 * math.copysign(1.0, H6) * math.log10(abs(H6))
        # label_hu_numbers[num][7] = -1 * math.copysign(1.0, H7) * math.log10(abs(H7))

        # label_hu_numbers[num][1] = H1
        # label_hu_numbers[num][2] = H2
        # label_hu_numbers[num][3] = H3
        # label_hu_numbers[num][4] = H4
        # label_hu_numbers[num][5] = H5
        # label_hu_numbers[num][6] = H6
        # label_hu_numbers[num][7] = H7


        m = [[0,0],[0,0]]
        for i in range(2) :
            for j in range(2) :
                for row in range(nrow) :
                    for col in range(ncol) :
                        m[i][j] = m[i][j] + (pow(row,i) * pow(col,j) * a_bin[row][col])

        x_0 = m[1][0] / m[0][0]
        y_0 = m[0][1] / m[0][0]

        # print(m[0][0])
        # print(m[0][1])
        # print(m[1][0])

        # print(x_0)
        # print(y_0)

        mu = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        for i in range(4):
            for j in range(4):
                for row in range(nrow):
                    for col in range(ncol):
                        mu[i][j] = mu[i][j] + (pow((row - x_0), i) * pow((col - y_0), j) * a_bin[row][col])

        # print("m00===",mu[0][0])
        # print("m01===",mu[0][1])
        # print("m02===",mu[0][2])
        # print("m03===",mu[0][3])

        n = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        for i in range(4):
            for j in range(4):
                for row in range(nrow):
                    for col in range(ncol):
                        n[i][j] = (mu[i][j] / pow(mu[0][0], ((i+j)/2)+1))

        # print(n[0][0])
        # print(n[0][1])
        # print(n[0][2])
        # print(n[0][3])

        H1 = n[2][0] + n[0][2]
        H2 = pow((n[2][0] - n[0][2]), 2) + 4 * pow((n[1][1]), 2)
        H3 = pow((n[3][0] - 3 * n[1][2]), 2) + pow((3 * n[2][1] - n[0][3]), 2)
        H4 = pow((n[3][0] + n[1][2]), 2) + pow((n[2][1] + n[0][3]), 2)
        H5 = (n[3][0] - 3 * n[1][2]) * (n[3][0] + n[1][2]) * (pow((n[3][0] + n[1][2]), 2) - 3 * pow((n[2][1] + n[0][3]), 2)) + \
        (3 * n[2][1] - n[0][3]) * (n[2][1] + n[0][3]) * (3 * pow((n[3][0] + n[1][2]), 2) - pow((n[2][1] + n[0][3]), 2))
        H6 = (n[2][0] - n[0][2]) * (pow((n[3][0] + n[1][2]), 2) - pow((n[2][1] + n[0][3]), 2)) + \
            4 * n[1][1] * (n[3][0] + n[1][2]) * (n[2][1] + n[0][3])
        H7 = -(((3 * n[2][1]) - n[0][3]) * (n[3][0] + n[1][2]) * (pow((n[3][0] + n[1][2]), 2) - (3 * pow(n[2][1] + n[0][3], 2))) - \
         (n[3][0] - (3 * n[1][2])) * (n[2][1] + n[0][3]) * ((3 * pow((n[3][0] + n[1][2]), 2)) - (pow((n[2][1] + n[0][3]), 2))))


        # print(H1)
        # print(H2)
        # print(H3)
        # print(H4)
        # print(H5)
        # print(H6)
        # print(H7)

        label_hu_numbers[num][1] = H1
        label_hu_numbers[num][2] = H2
        label_hu_numbers[num][3] = H3
        label_hu_numbers[num][4] = H4
        label_hu_numbers[num][5] = H5
        label_hu_numbers[num][6] = H6
        label_hu_numbers[num][7] = H7


        # moments = cv2.moments(a_bin)
        # huMoments = cv2.HuMoments(moments)
        # print("CV-MOMENTS (DOWN)")
        # print(huMoments)

    return label_hu_numbers

def picture_to_number_hu(sample_array, current_hu, pixels, img):
    #print("cur_hu =", len(current_hu))
    #print("sample_hu =", len(sample_array))
    #print(sample_array)

    draw = ImageDraw.Draw(img)

    for cur in range(len(current_hu)):
        current_number = 9999999999999
        for i in range(len(sample_array)):
            a = math.sqrt((current_hu[cur][1] - sample_array[i][1])**2+(current_hu[cur][2] - sample_array[i][2])**2 +
                          (current_hu[cur][3] - sample_array[i][3])**2+(current_hu[cur][4] - sample_array[i][4])**2 +
                          (current_hu[cur][5] - sample_array[i][5])**2+(current_hu[cur][6] - sample_array[i][6])**2 +
                          (current_hu[cur][7] - sample_array[i][7])**2)
            if current_number >= a:
                current_number = a
                current_hu[cur][0] = sample_array[i][0]
            else:
                pass

    # for loop in range(len(current_hu)):
    #     print(int(current_hu[loop][0]))

    for loop in range(len(pixels)):
        draw.text((((pixels[loop][2] + pixels[loop][4]) / 2), pixels[loop][1] - 12), str(int(current_hu[loop][0])), fill="black", font=None, anchor=None)

    return


def r_moment(image_array):

    label_R_numbers = np.zeros((len(image_array), 11))  # number-R1-.....-R10
    label_hu_numbers = hu_moments(image_array)

    for i in range(len(label_hu_numbers)):
        label_R_numbers[i][0] = label_hu_numbers[i][0]

    for i in range(len(label_hu_numbers)):
        label_R_numbers[i][1] = math.sqrt(label_hu_numbers[i][2]) / label_hu_numbers[i][1]
        label_R_numbers[i][2] = (label_hu_numbers[i][1] + math.sqrt(label_hu_numbers[i][2])) / (label_hu_numbers[i][1] - math.sqrt(label_hu_numbers[i][2]))
        label_R_numbers[i][3] = math.sqrt(label_hu_numbers[i][3]) / math.sqrt(label_hu_numbers[i][4])
        label_R_numbers[i][4] = math.sqrt(label_hu_numbers[i][3]) / math.sqrt(abs(label_hu_numbers[i][5]))
        label_R_numbers[i][5] = math.sqrt(label_hu_numbers[i][4]) / math.sqrt(abs(label_hu_numbers[i][5]))
        label_R_numbers[i][6] = abs(label_hu_numbers[i][6]) / label_hu_numbers[i][1] * label_hu_numbers[i][3]
        label_R_numbers[i][7] = abs(label_hu_numbers[i][6]) / label_hu_numbers[i][1] * math.sqrt(abs(label_hu_numbers[i][5]))
        label_R_numbers[i][8] = abs(label_hu_numbers[i][6]) / label_hu_numbers[i][3] * math.sqrt(label_hu_numbers[i][2])
        label_R_numbers[i][9] = abs(label_hu_numbers[i][6]) / math.sqrt(abs(label_hu_numbers[i][5])) * math.sqrt(label_hu_numbers[i][2])
        label_R_numbers[i][10] = abs(label_hu_numbers[i][5]) / label_hu_numbers[i][3] * label_hu_numbers[i][4]

    return label_R_numbers

def picture_to_number_R (sample_array, current_hu, pixels, img):

    draw = ImageDraw.Draw(img)

    for cur in range(len(current_hu)):
        current_number = 9999999999999
        for i in range(len(sample_array)):
            a = math.sqrt((current_hu[cur][1] - sample_array[i][1])**2 + (current_hu[cur][2] - sample_array[i][2])**2 +
                          (current_hu[cur][3] - sample_array[i][3])**2 + (current_hu[cur][4] - sample_array[i][4])**2 +
                          (current_hu[cur][5] - sample_array[i][5])**2 + (current_hu[cur][6] - sample_array[i][6])**2 +
                          (current_hu[cur][7] - sample_array[i][7])**2 + (current_hu[cur][8] - sample_array[i][8])**2 +
                          (current_hu[cur][9] - sample_array[i][9])**2 + (current_hu[cur][10] - sample_array[i][10])**2)
            if current_number >= a:
                current_number = a
                current_hu[cur][0] = sample_array[i][0]
            else:
                pass

    # for loop in range(len(current_hu)):
    #     print(int(current_hu[loop][0]))

    for loop in range(len(pixels)):
        draw.text((((pixels[loop][2] + pixels[loop][4]) / 2), pixels[loop][1] - 12), str(int(current_hu[loop][0])), fill="black", font=None, anchor=None)

    return

if __name__=='__main__':
    main()
