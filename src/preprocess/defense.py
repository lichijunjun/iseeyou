import random
from io import BytesIO

import cv2
import numpy as np
import skimage
from PIL import Image


def defend_rescale(img, size=56):
    ori_size = 224
    new_size = size
    img = np.array(img)
    rescaled = skimage.transform.resize(img, (new_size, new_size))
    rescaled = skimage.transform.resize(rescaled, (ori_size, ori_size))
    return (rescaled * 255).astype(np.uint8)

def defend_rescale_s(img):
    ori_size = 32
    new_size = 48
    img = np.array(img)
    rescaled = skimage.transform.resize(img, (new_size, new_size))
    rescaled = skimage.transform.resize(rescaled, (ori_size, ori_size))
    return rescaled

def defend_onlyrand(img):
    ori_size = 224 # 299
    max_size = 299 # 400 / 299 * 224
    img = np.array(img)
    rnd = np.random.randint(ori_size,max_size,(1,))[0]
    rescaled = skimage.transform.resize(img,(rnd,rnd))
    h_rem = max_size - rnd
    w_rem = max_size - rnd
    pad_left = np.random.randint(0,w_rem,(1,))[0]
    pad_right = w_rem - pad_left
    pad_top = np.random.randint(0,h_rem,(1,))[0]
    pad_bottom = h_rem - pad_top
    padded = np.pad(rescaled,((pad_top,pad_bottom),(pad_left,pad_right),(0,0)),'constant',constant_values = 0.5)
    padded = skimage.transform.resize(padded,(ori_size,ori_size))
    return (padded * 255).astype(np.uint8)


def defend_onlyrand_s(img):
    ori_size = 32 # 299
    max_size = 43 # 400 / 299 * 32
    img = np.array(img)
    rnd = np.random.randint(ori_size,max_size,(1,))[0]
    rescaled = skimage.transform.resize(img,(rnd,rnd))
    h_rem = max_size - rnd
    w_rem = max_size - rnd
    pad_left = np.random.randint(0,w_rem,(1,))[0]
    pad_right = w_rem - pad_left
    pad_top = np.random.randint(0,h_rem,(1,))[0]
    pad_bottom = h_rem - pad_top
    padded = np.pad(rescaled,((pad_top,pad_bottom),(pad_left,pad_right),(0,0)),'constant',constant_values = 0.5)
    padded = skimage.transform.resize(padded,(ori_size,ori_size))
    return padded


def defense_fd():
    pass


def defense_rdg():
    pass


from albumentations import augmentations
from scipy.fftpack import dct, idct, irfft, rfft

# FD algorithm
num = 8
q_table = np.ones((num,num))*30
q_table[0:4,0:4] = 25

def dct2 (block):
    return dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')
def idct2(block):
    return idct(idct(block.T, norm = 'ortho').T, norm = 'ortho')
def rfft2 (block):
    return rfft(rfft(block.T).T)
def irfft2(block):
    return irfft(irfft(block.T).T)




# Feature distillation for single imput
def FD_fuction_sig(input_matrix):
    output = []

    h = input_matrix.shape[0]
    w = input_matrix.shape[1]
    c = input_matrix.shape[2]
    horizontal_blocks_num = w / num
    vertical_blocks_num = h / num
    output2 = np.zeros((c, h, w))

    c_block = np.split(input_matrix, c, axis=2)
    j = 0
    for ch_block in c_block:
        vertical_blocks = np.split(ch_block, vertical_blocks_num, axis=0)
        k = 0
        for block_ver in vertical_blocks:
            hor_blocks = np.split(block_ver, horizontal_blocks_num, axis=1)
            m = 0
            for block in hor_blocks:
                block = np.reshape(block, (num, num))
                block = dct2(block)
                # quantization
                table_quantized = np.matrix.round(np.divide(block, q_table))
                table_quantized = np.squeeze(np.asarray(table_quantized))
                # de-quantization
                table_unquantized = table_quantized * q_table
                IDCT_table = idct2(table_unquantized)
                if m == 0:
                    output = IDCT_table
                else:
                    output = np.concatenate((output, IDCT_table), axis=1)
                m = m + 1
            if k == 0:
                output1 = output
            else:
                output1 = np.concatenate((output1, output), axis=0)
            k = k + 1
        output2[j] = output1
        j = j + 1

    output2 = np.transpose(output2, (1, 0, 2))
    output2 = np.transpose(output2, (0, 2, 1))

    return output2.astype(np.uint8)

def padresult_sig(cleandata):
    margin = 8
    ori_size = 224
    pad = augmentations.transforms.PadIfNeeded(min_height=ori_size + margin, min_width=ori_size + margin, border_mode=4)
    paddata = pad(image=cleandata)['image']
    return paddata

def cropresult_sig(paddata):
    ori_size = 224
    crop = augmentations.transforms.Crop(0, 0, ori_size, ori_size)
    resultdata = crop(image=paddata)['image']
    return resultdata

def defend_FD_sig(data):
    data = np.array(data)
    paddata = padresult_sig(data)
    defendresult = FD_fuction_sig(paddata)
    resultdata = cropresult_sig(defendresult)
    return resultdata

def padresult_sig_s(cleandata):
    margin = 8
    ori_size = 32
    pad = augmentations.transforms.PadIfNeeded(min_height=ori_size + margin, min_width=ori_size + margin, border_mode=4)
    paddata = pad(image=cleandata)['image']
    return paddata

def cropresult_sig_s(paddata):
    ori_size = 32
    crop = augmentations.transforms.Crop(0, 0, ori_size, ori_size)
    resultdata = crop(image=paddata)['image']
    return resultdata

def defend_FD_sig_s(data):
    paddata = padresult_sig_s(data)
    defendresult = FD_fuction_sig(paddata)
    resultdata = cropresult_sig_s(defendresult)
    return resultdata

def defend_gd(img,distort_limit = 0.25):
    img = np.array(img)
    ori_size = 224
    num_steps = 10

    xsteps = [1 + random.uniform(-distort_limit, distort_limit) for i in range(num_steps + 1)]
    ysteps = [1 + random.uniform(-distort_limit, distort_limit) for i in range(num_steps + 1)]

    height, width = img.shape[:2]

    x_step = width // num_steps
    xx = np.zeros(width, np.float32)
    prev = 0
    for idx, x in enumerate(range(0, width, x_step)):
        start = x
        end = x + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * xsteps[idx]

        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    y_step = height // num_steps
    yy = np.zeros(height, np.float32)
    prev = 0
    for idx, y in enumerate(range(0, height, y_step)):
        start = y
        end = y + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * ysteps[idx]

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur
    xx = np.round(xx).astype(int)
    yy = np.round(yy).astype(int)
    xx[xx >= ori_size] = ori_size - 1
    yy[yy >= ori_size] = ori_size - 1

    map_x, map_y = np.meshgrid(xx, yy)

    #     index=np.dstack((map_y,map_x))
    #     outimg = remap(img,index)
    #     if np.ndim(img)>2:
    #         outimg = outimg.transpose(1,0,2)
    #     else:
    #         outimg = outimg.T

    # to speed up the mapping procedure, OpenCV 2 is adopted
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    outimg = cv2.remap(img, map1=map_x, map2=map_y, interpolation=1, borderMode=4, borderValue=None)
    return outimg

def defend_gd_s(img,distort_limit = 0.25):
    ori_size = 32
    num_steps = 10

    xsteps = [1 + random.uniform(-distort_limit, distort_limit) for i in range(num_steps + 1)]
    ysteps = [1 + random.uniform(-distort_limit, distort_limit) for i in range(num_steps + 1)]

    height, width = img.shape[:2]

    x_step = width // num_steps
    xx = np.zeros(width, np.float32)
    prev = 0
    for idx, x in enumerate(range(0, width, x_step)):
        start = x
        end = x + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * xsteps[idx]

        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    y_step = height // num_steps
    yy = np.zeros(height, np.float32)
    prev = 0
    for idx, y in enumerate(range(0, height, y_step)):
        start = y
        end = y + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * ysteps[idx]

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur
    xx = np.round(xx).astype(int)
    yy = np.round(yy).astype(int)
    xx[xx >= ori_size] = ori_size - 1
    yy[yy >= ori_size] = ori_size - 1

    map_x, map_y = np.meshgrid(xx, yy)

    #     index=np.dstack((map_y,map_x))
    #     outimg = remap(img,index)
    #     if np.ndim(img)>2:
    #         outimg = outimg.transpose(1,0,2)
    #     else:
    #         outimg = outimg.T

    # to speed up the mapping procedure, OpenCV 2 is adopted
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    outimg = cv2.remap(img, map1=map_x, map2=map_y, interpolation=1, borderMode=4, borderValue=None)
    return outimg


def defend_pd(img, deflections=600, window=10):
    img = np.copy(img)
    H, W, C = img.shape
    while deflections > 0:
        #for consistency, when we deflect the given pixel from all the three channels.
        for c in range(C):
            x,y = random.randint(0,H-1), random.randint(0,W-1)
            while True: #this is to ensure that PD pixel lies inside the image
                a,b = random.randint(-1*window,window), random.randint(-1*window,window)
                if x+a < H and x+a > 0 and y+b < W and y+b > 0: break
            # calling pixel deflection as pixel swap would be a misnomer,
            # as we can see below, it is one way copy
            img[x,y,c] = img[x+a,y+b,c]
        deflections -= 1
    return img


def nearest_neighbour_scaling(label, new_h, new_w, patch_size, patch_n, patch_m):
    if len(label.shape) == 2:
        label_new = np.zeros([new_h, new_w])
    else:
        label_new = np.zeros([new_h, new_w, label.shape[2]])
    n_pos = np.arange(patch_n)
    m_pos = np.arange(patch_m)
    n_pos = n_pos.repeat(patch_size)[:299]
    m_pos = m_pos.repeat(patch_size)[:299]
    n_pos = n_pos.reshape(n_pos.shape[0], 1)
    n_pos = np.tile(n_pos, (1, new_w))
    m_pos = np.tile(m_pos, (new_h, 1))
    assert n_pos.shape == m_pos.shape
    label_new[:, :] = label[n_pos[:, :], m_pos[:, :]]
    return label_new

def jpeg(input_array, quali):
    pil_image = Image.fromarray((input_array * 255.0).astype(np.uint8))
    f = BytesIO()
    pil_image.save(f, format='jpeg', quality=quali)  # quality level specified in paper
    jpeg_image = np.asarray(Image.open(f)).astype(np.float32) / 255.0
    return jpeg_image

def defend_shield(x, qualities=(20, 40, 60, 80), patch_size=8):
    n = x.shape[0]
    m = x.shape[1]
    patch_n = int(n / patch_size)
    patch_m = int(m / patch_size)
    num_qualities = len(qualities)
    if n % patch_size > 0:
        patch_n = np.int(n / patch_size) + 1
        delete_n = 1
    if m % patch_size > 0:
        patch_m = np.int(m / patch_size) + 1
        delet_m = 1

    R = np.tile(np.reshape(np.arange(n), (n, 1)), [1, m])
    C = np.reshape(np.tile(np.arange(m), [n]), (n, m))
    mini_Z = (np.random.rand(patch_n, patch_m) * num_qualities).astype(int)
    Z = (nearest_neighbour_scaling(mini_Z, n, m, patch_size, patch_n, patch_m)).astype(int)
    indices = np.transpose(np.stack((Z, R, C)), (1, 2, 0))
    # x = img_as_ubyte(x)
    x_compressed_stack = []

    for quali in qualities:
        processed = jpeg(x, quali)
        x_compressed_stack.append(processed)

    x_compressed_stack = np.asarray(x_compressed_stack)
    x_slq = np.zeros((n, m, 3))
    for i in range(n):
        for j in range(m):
            x_slq[i, j] = x_compressed_stack[tuple(indices[i][j])]
    return x_slq


def defend_BitReduct(arr, depth=3):
    arr = (np.array(arr)).astype(np.uint8)
    shift = 8 - depth
    arr = (arr >> shift) << shift
    arr = arr.astype(np.uint8)
    return arr

def defend_FixedJpeg(input_array, quality=75):
    pil_image = input_array
    f = BytesIO()
    pil_image.save(f, format='jpeg', quality=quality) # quality level specified in paper
    jpeg_image = np.asarray(Image.open(f)).astype(np.uint8)
    return jpeg_image

def bregman(image, mask, weight, eps=1e-3, max_iter=100):
    rows, cols, dims = image.shape
    rows2 = rows + 2
    cols2 = cols + 2
    total = rows * cols * dims
    shape_ext = (rows2, cols2, dims)

    u = np.zeros(shape_ext)
    dx = np.zeros(shape_ext)
    dy = np.zeros(shape_ext)
    bx = np.zeros(shape_ext)
    by = np.zeros(shape_ext)

    u[1:-1, 1:-1] = image
    # reflect image
    u[0, 1:-1] = image[1, :]
    u[1:-1, 0] = image[:, 1]
    u[-1, 1:-1] = image[-2, :]
    u[1:-1, -1] = image[:, -2]
    
    i = 0
    rmse = np.inf
    lam = 2 * weight
    norm = (weight + 4 * lam)

    while i < max_iter and rmse > eps:
        rmse = 0

        for k in range(dims):
            for r in range(1, rows + 1):
                for c in range(1, cols + 1):
                    uprev = u[r, c, k]

                    # forward derivatives
                    ux = u[r, c + 1, k] - uprev
                    uy = u[r + 1, c, k] - uprev

                    # Gauss-Seidel method
                    if mask[r - 1, c - 1]:
                        unew = (lam * (u[r + 1, c, k] +
                                       u[r - 1, c, k] +
                                       u[r, c + 1, k] +
                                       u[r, c - 1, k] +
                                       dx[r, c - 1, k] -
                                       dx[r, c, k] +
                                       dy[r - 1, c, k] -
                                       dy[r, c, k] -
                                       bx[r, c - 1, k] +
                                       bx[r, c, k] -
                                       by[r - 1, c, k] +
                                       by[r, c, k]
                                       ) + weight * image[r - 1, c - 1, k]
                                ) / norm
                    else:
                        # similar to the update step above, except we take
                        # lim_{weight->0} of the update step, effectively
                        # ignoring the l2 loss
                        unew = (u[r + 1, c, k] +
                                u[r - 1, c, k] +
                                u[r, c + 1, k] +
                                u[r, c - 1, k] +
                                dx[r, c - 1, k] -
                                dx[r, c, k] +
                                dy[r - 1, c, k] -
                                dy[r, c, k] -
                                bx[r, c - 1, k] +
                                bx[r, c, k] -
                                by[r - 1, c, k] +
                                by[r, c, k]
                                ) / 4.0
                    u[r, c, k] = unew

                    # update rms error
                    rmse += (unew - uprev) ** 2

                    bxx = bx[r, c, k]
                    byy = by[r, c, k]

                    # d_subproblem
                    s = ux + bxx
                    if s > 1 / lam:
                        dxx = s - 1 / lam
                    elif s < -1 / lam:
                        dxx = s + 1 / lam
                    else:
                        dxx = 0
                    s = uy + byy
                    if s > 1 / lam:
                        dyy = s - 1 / lam
                    elif s < -1 / lam:
                        dyy = s + 1 / lam
                    else:
                        dyy = 0

                    dx[r, c, k] = dxx
                    dy[r, c, k] = dyy

                    bx[r, c, k] += ux - dxx
                    by[r, c, k] += uy - dyy

        rmse = np.sqrt(rmse / total)
        i += 1
        
    return np.squeeze(np.asarray(u[1:-1, 1:-1]))
def defend_TotalVarience(input_array, keep_prob=0.5, lambda_tv=0.03):
    input_array = np.array(input_array)
    mask = np.random.uniform(size=input_array.shape[:2])
    mask = mask < keep_prob
    return bregman(input_array, mask, weight=2.0 / lambda_tv)