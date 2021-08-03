import glob
import os
from shutil import copyfile
import json
from shapely.geometry import mapping, shape
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from matplotlib import cm as colormaps
import cv2
from scipy import ndimage
import copy
import time
import collections
import yaml
import csv

Image.MAX_IMAGE_PIXELS = None

global_counters = []
global_lists = []

global_reshape_w = 0

# proj_name='IgAN-PAS'
proj_name = 'IgAN-regions'
main_folder = 'Bignefro'


def perform_on_whole_dataset(funct, ids=None):
    if ids is None:
        ids = list(range(1000, 2220))
    for i in ids:
        if i % 250 == 0:
            print(f'now processing image #{i}')
        try:
            funct(i)
        except FileNotFoundError:
            # print(f"Error for id {i}:")
            # print(e)
            # print()
            continue
        except Exception as e:
            print(f"Error for id {i}:")
            print(e)
            print()


def debug_plot(img, cmap=None):
    img = np.array(img)
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.show(block=False)


def copy_files():
    path = f'E:\{main_folder}\IgAN_annotated (QuPath)\QuPath projects\{proj_name}\data\\'
    new_path = f'E:\{main_folder}\\porcherie\\'
    files = glob.glob(path + '*\\data.qpdata')
    for f in files:
        new_fname = f'{os.path.basename(os.path.dirname(f))}.qpdata'
        print(f'copying {f} into {new_path}\\{new_fname}')
        copyfile(f, new_path + "\\" + new_fname)


def parse_geojsn(file_path, big_size):
    labels_dict = {'Glomerulus': 1,
                   'Artery': 2,
                   'hilus': 3,
                   'crescent': 4,
                   'Vein': 5,
                   'Medulla': 6,
                   'Capsule': 7,
                   'unknown': 8}
    img = Image.new('L', big_size, 0)
    with open(file_path) as json_file:
        data = json.load(json_file)
        for polygon in data:
            try:
                name = polygon["properties"]["name"]
            except:
                try:
                    name = polygon["properties"]["classification"]["name"]
                except KeyError as e:
                    print(f'could not get the name of a polygon in {os.path.basename(file_path).split(".")[0]}')
                    print()
                    name = 'unknown'
            if name in ['discarded', 'mesangial']:
                continue
            try:
                class_name = polygon['properties']['classification']['name']
            except KeyError:
                class_name = 'unknown'
            # POLLO DO WE WANT TO ONLY PARSE SOME CLASSES??
            # if class_name not in ['Glomerulus']:
            #     continue
            fill_value = labels_dict.get(class_name)
            polygon_shape = shape(polygon['geometry'])
            for closed_figure in mapping(polygon_shape)['coordinates']:

                # POL: the case len == 2 if handcrafted after seeing 2 single failure cases on arteries in images 3623_pas_Regione 1 and 2891_pas_Regione 0.
                if len(closed_figure) in [1, 2]:
                    closed_figure = closed_figure[0]
                try:
                    ImageDraw.Draw(img).polygon(xy=closed_figure, outline=fill_value, fill=fill_value)
                except TypeError:
                    print(f'could not draw polygon {name} in {os.path.basename(file_path).split(".")[0]}')

        return img


def get_region_from_id(data_id):
    if data_id > 1000:
        data_id -= 1000
        proj_name = 'IgAN-regions'
    else:
        proj_name = 'IgAN-PAS'
    path = f'E:/{main_folder}/IgAN_annotated (QuPath)/QuPath projects/{proj_name}/data/{data_id}/'
    with open(path + 'server.json') as json_file:
        return json.load(json_file)['metadata']['name'].split('.')[0]


def segment_region(data_id):
    read_data_id = data_id
    if data_id > 1000:
        read_data_id = data_id - 1000
    path = f'E:/{main_folder}/IgAN_annotated (QuPath)/QuPath projects/{proj_name}/data/{read_data_id}/'
    region_name = get_region_from_id(data_id)
    imgname_root = f'E:/{main_folder}/thumbnails_processing/id{data_id}_{region_name}'
    img = cv2.imread(path + 'thumbnail.jpg', 0)
    cv2.imwrite(imgname_root + '_0thumb.png', img)
    thresh, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(imgname_root + '_1Otsu.png', bin_img)
    try:
        bin_img = remove_futile(data_id, region_name, bin_img)
    except FileNotFoundError:
        get_small_annotation_npy(data_id)
        bin_img = remove_futile(data_id, region_name, bin_img)
    cv2.imwrite(imgname_root + '_2RemoveFutiles.png', bin_img)
    bin_img = cv2.dilate(bin_img, np.ones((11, 11)))  # this initial closing step was added later
    bin_img = cv2.erode(bin_img, np.ones((11, 11)))  # this initial closing step was added later
    bin_img = cv2.dilate(bin_img, np.ones((11, 11)))  # this initial closing step was added later
    bin_img = cv2.erode(bin_img, np.ones((11, 11)))  # this initial closing step was added later
    cv2.imwrite(imgname_root + '_3TwoRoundsOfClosing.png', bin_img)
    bin_img = cv2.erode(bin_img, np.ones((11, 11)))
    bin_img = cv2.dilate(bin_img, np.ones((11, 11)))
    bin_img = cv2.erode(bin_img, np.ones((11, 11)))
    bin_img = cv2.dilate(bin_img, np.ones((11, 11)))
    cv2.imwrite(imgname_root + '_4TwoRoundsOfOpening.png', bin_img)
    bin_img = cv2.dilate(bin_img, np.ones((11, 11)))
    bin_img = cv2.dilate(bin_img, np.ones((11, 11)))
    bin_img = cv2.erode(bin_img, np.ones((11, 11)))
    cv2.imwrite(imgname_root + '_5DilationPlusClosing.png', bin_img)
    bin_img, final_img = merge_otsu_and_annotations(data_id, region_name, bin_img)
    cv2.imwrite(imgname_root + '_6MergeGlomeruliPlusFillHoles.png', bin_img)
    cv2.imwrite(imgname_root + '_7CCL.png', final_img)


def processandresize_big_image(data_id):
    img_name = get_region_from_id(data_id)
    nbio = img_name.split('_')[0]
    bigimg_path = f'E:/{main_folder}/IgAN_acquired_images_(master)/{nbio}/{img_name}.tif'
    big_img = Image.open(bigimg_path)
    thumb_path = f'E:/{main_folder}/thumbnails_processing/id{data_id}_{img_name}_0thumb.png'
    thumb_img = Image.open(thumb_path)
    small_img = big_img.resize(thumb_img.size, resample=Image.BILINEAR)

    json_path = f'E:/{main_folder}/json_annotations/{img_name}.txt'
    annotations = parse_geojsn(json_path, big_img.size)
    small_annotations = annotations.resize(thumb_img.size, resample=Image.NEAREST)
    cm_bin = colormaps.get_cmap('viridis')
    printable_annotations = np.array(small_annotations)
    printable_annotations = printable_annotations * 51
    printable_annotations = cm_bin(printable_annotations)[:, :, :-1]
    printable_annotations = np.uint8(255 * printable_annotations)
    printable_annotations = Image.fromarray(printable_annotations)

    # debug_plot(printable_annotations)
    # debug_plot(small_img)
    printable_annotations.save(f'E:/{main_folder}/tmp/id{data_id}_{img_name}_annotations.png')
    small_img.save(f'E:/{main_folder}/tmp/id{data_id}_{img_name}_resized.png')


def fix_json_file(data_id):
    img_name = get_region_from_id(data_id)
    json_path = f'E:/{main_folder}/json_annotations/{img_name}.txt'
    try:
        with open(json_path, 'r') as json_file:
            json.load(json_file)
    except json.decoder.JSONDecodeError:
        print(data_id)
        with open(json_path, 'r') as json_file:
            temp = json_file.read().replace("[,{", "[{")

        with open(json_path, 'w') as json_file:
            json_file.write(temp)
    with open(json_path, 'r') as json_file:
        json.load(json_file)


def get_small_annotation_npy(data_id):
    img_name = get_region_from_id(data_id)
    nbio = img_name.split('_')[0]
    bigimg_path = f'E:/{main_folder}/IgAN_acquired_images_(master)/{nbio}/{img_name}.tif'
    big_img = Image.open(bigimg_path)
    thumb_path = f'E:/{main_folder}/thumbnails_processing/id{data_id}_{img_name}_0thumb.png'
    thumb_img = Image.open(thumb_path)
    # small_img = big_img.resize(thumb_img.size, resample=Image.BILINEAR)

    json_path = f'E:/{main_folder}/json_annotations/{img_name}.txt'
    annotations = parse_geojsn(json_path, big_img.size)
    small_annotations = annotations.resize(thumb_img.size, resample=Image.NEAREST)
    small_annotation_npy = np.asarray(small_annotations)
    np_path = f'E:/{main_folder}/small_npys/id{data_id}_{img_name}_small.npy'
    # debug_plot(small_annotation_npy)
    np.save(np_path, small_annotation_npy)


def merge_otsu_and_annotations(data_id, img_name, img):
    global global_counters
    np_path = f'E:/{main_folder}/small_npys/id{data_id}_{img_name}_small.npy'
    small_annotation_npy = np.load(np_path)
    glomeruli = (small_annotation_npy == 1).astype(np.uint8)
    img //= 255
    glomeruli_check = np.sum(glomeruli) - np.sum(img * glomeruli)
    # debug_plot(small_otsu, cmap='gray')
    # debug_plot(mask, cmap='gray')
    if glomeruli_check > 0:
        global_counters[-2] += 1
        global_counters[-1] += glomeruli_check
        print(f'{glomeruli_check} pixels from glomeruli were dropped in image {data_id}_{img_name}')
    # debug_plot(mask, cmap='gray')
    # debug_plot(glomeruli, cmap='gray')
    # debug_plot(glomeruli - mask * glomeruli, cmap='gray')

    bin_img = (np.logical_or(img, glomeruli) * 255).astype(np.uint8)
    bin_img[ndimage.binary_fill_holes(bin_img)] = 255
    final_img = bin_img.copy()
    final_img = remove_small_CCL(data_id, img_name, final_img, glomeruli)
    # debug_plot(bin_img, 'gray')
    # debug_plot(final_img, 'gray')
    return bin_img, final_img


def remove_futile(data_id, img_name, img):
    np_path = f'E:/{main_folder}/small_npys/id{data_id}_{img_name}_small.npy'
    small_annotation_npy = np.load(np_path)
    small_otsu = img // 255
    futile = (small_annotation_npy == 6).astype(np.uint8) + (small_annotation_npy == 7).astype(np.uint8)
    return (small_otsu * np.logical_not(futile) * 255).astype(np.uint8)


def remove_files(folder, str):
    global global_counters
    global_counters.append(0)
    files_2remove = glob.glob(folder + f'*{str}*')
    for f in files_2remove:
        if str in f:
            global_counters[-1] += 1
            os.remove(f)


def rename_files(folder, str, replace_str):
    global global_counters
    global_counters.append(0)
    files_2remove = glob.glob(folder + f'*{str}*')
    for f in files_2remove:
        if str in f:
            global_counters[0] += 1
            os.rename(f, f.replace(str, replace_str))
            # print(f, f.replace(str, replace_str))


def remove_small_CCL(*args, **kwargs):
    if proj_name == 'IgAN-PAS':
        return remove_small_CCL_w_annotations(*args, **kwargs)
    else:
        return remove_small_CCL_NO_annotations(*args, **kwargs)


def remove_small_CCL_w_annotations(data_id, img_name, img, glomeruli):
    global global_counters
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    # just taking out the background
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = sizes.max() // 5
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        single_CC = np.zeros((output.shape))
        single_CC[output == i + 1] = 1
        if not np.sum(single_CC * glomeruli):
            img[output == i + 1] = 0
            global_counters[-2] += 1
            if sizes[i] >= min_size:
                print(f'image id{data_id}_{img_name} contains a big glomerulusless CC')
        # else:
        #     print(f'image id{data_id}_{img_name} contains a detatched glomerulus')

    return img


def remove_small_CCL_NO_annotations(data_id, img_name, img, glomeruli):
    global global_counters
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    # just taking out the background
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    for i in range(0, nb_components):
        single_CC = np.zeros((output.shape))
        single_CC[output == i + 1] = 1
        if np.sum(single_CC) < (single_CC.size // 20):
            img[output == i + 1] = 0
            global_counters[-2] += 1
        # else:
        #     print(f'image id{data_id}_{img_name} contains a detatched glomerulus')

    return img


def get_polygon_from_WTK(file_lines, glname):
    for i in range(len(file_lines)):
        stuff = file_lines[i].split(' ')
        if stuff[0] == glname:
            type = stuff[1].replace('(', '').replace(')', '')
            obj_name = stuff[2].replace('(', '').replace(')\n', '')
            if type != 'Polygon' or obj_name != 'Glomerulus':
                raise KeyError("SOMETHING WENT WRONG")
            line = file_lines[i + 1]
            break
    line_start = 'POLYGON (('
    line_end = '))\n'
    if not line.startswith(line_start) or not line.endswith(line_end):
        raise KeyError("SOMETHING WENT WRONG")
    line = line.replace(line_start, '').replace(line_end, '')
    points = line.split(', ')
    json_points = [[int(el) for el in couple.split(' ')] for couple in points]
    d = {"type": type,
         "coordinates": [json_points]}
    return d


def add_single_glomerulus(filename, glname):
    print(f'filename: {filename} | glname: {glname}')
    data_root = f'E:/{main_folder}/'
    jsons_root = data_root + 'json_annotations/'
    new_jsons_root = data_root + 'new_json_annotations/'
    WTK_root = data_root + 'tmp_annotations/'

    with open(WTK_root + filename) as WTK_file:
        WTK_lines = WTK_file.readlines()

    with open(jsons_root + filename) as json_file:
        data = json.load(json_file)
        for el in data:
            try:
                obj_name = el['properties']['classification']['name']
            except Exception:
                continue
            if obj_name == 'Glomerulus':
                new_el = el
                break
        new_el['properties']['name'] = glname
        new_el['geometry'] = get_polygon_from_WTK(WTK_lines, glname)
        data.append(new_el)
    with open(new_jsons_root + filename, 'w') as new_json_file:
        json.dump(data, new_json_file)


def WTK2Json():
    data_root = f'E:/{main_folder}/'
    with open(data_root + 'valid_jsons_logs.txt', 'r') as logsfile:
        lines = logsfile.readlines()

    lines = [line for line in lines if line != '\n']
    line_start = "INFO: ERROR in "
    i = 0
    while i < len(lines):
        if lines[i].startswith(line_start):
            filename = lines[i].replace(line_start, '')[:-1]
            i += 1
            glname = lines[i].split(' ')[4]
            add_single_glomerulus(filename, glname)
        i += 1

    return


def check_jsons(path):
    files = glob.glob(path + '*.txt')
    for f in files:
        with open(f) as jsonfile:
            data = json.load(jsonfile)
        print(f'file {os.path.basename(f)} contains {len(data)} shapes')


def get_rotation_angle(img):
    gray_thresh = img.astype(np.double)
    col_mask = np.arange(gray_thresh.shape[1])
    col_mask = np.expand_dims(col_mask, 0)
    col_mask = np.repeat(col_mask, repeats=gray_thresh.shape[0], axis=0)
    col_mask = col_mask.astype(np.double)
    row_mask = np.arange(gray_thresh.shape[0])
    row_mask = np.expand_dims(row_mask, 1)
    row_mask = np.repeat(row_mask, repeats=gray_thresh.shape[1], axis=1)
    row_mask = row_mask.astype(np.double)
    M00 = np.sum(gray_thresh)
    M01 = np.sum((col_mask ** 0) * (row_mask ** 1) * gray_thresh)
    M10 = np.sum((col_mask ** 1) * (row_mask ** 0) * gray_thresh)
    M11 = np.sum((col_mask ** 1) * (row_mask ** 1) * gray_thresh)
    M02 = np.sum((col_mask ** 0) * (row_mask ** 2) * gray_thresh)
    M20 = np.sum((col_mask ** 2) * (row_mask ** 0) * gray_thresh)
    xm = M10 / M00
    ym = M01 / M00
    u20 = M20 / M00 - (xm ** 2)
    u02 = M02 / M00 - (ym ** 2)
    u11 = M11 / M00 - (xm * ym)
    lambda1 = (u20 + u02) / 2 + np.sqrt(4 * u11 ** 2 + (u20 - u02) ** 2) / 2
    lambda2 = (u20 + u02) / 2 - np.sqrt(4 * u11 ** 2 + (u20 - u02) ** 2) / 2
    d = np.sqrt(M00 * np.sqrt(lambda1 * lambda2) / np.pi)
    dmin = d / np.sqrt(lambda1)
    dmag = d / np.sqrt(lambda2)
    angle = 0.5 * np.arctan2(2 * u11, (u20 - u02))
    angle = angle * 180 / np.pi
    return angle, xm, ym


def custom_rotate(img, angle, cx, cy):
    # image_center = cx, cy
    cx = int(cx)
    cy = int(cy)
    h, w = img.shape[:2]
    if len(img.shape) == 3:
        assert img.shape[2] == 3
        fill_value = (255, 255, 255)
        resample_mode = Image.BILINEAR
    else:
        fill_value = 0
        resample_mode = Image.NEAREST

    # pad to center image
    pad_dist = int(np.abs(cx - w / 2) * 2)
    if cx - w / 2 > 0:
        right = pad_dist
        left = 0
    else:
        left = pad_dist
        right = 0
    pad_dist = int(np.abs(cy - h / 2) * 2)
    if cy - h / 2 > 0:
        bot = pad_dist
        top = 0
    else:
        bot = pad_dist
        top = 0
    result = cv2.copyMakeBorder(img, top, bot, left, right, cv2.BORDER_CONSTANT, value=fill_value)

    # pad to not crop during rotate
    diag = np.sqrt(h ** 2 + w ** 2)
    if diag > result.shape[1]:
        left = right = int(diag - img.shape[1]) // 2
        result = cv2.copyMakeBorder(result, 0, 0, left, right, cv2.BORDER_CONSTANT, value=fill_value)

    # rotate with opencv fails for very big images
    # rot_mat = cv2.getRotationMatrix2D((result.shape[1] / 2, result.shape[0] / 2), angle, 1.0)
    # result = cv2.warpAffine(result, rot_mat, result.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=fill_value)

    # therefore, rotate image with PIL
    try:
        result = Image.fromarray(result)
    except ValueError:
        result = create_big_PIL_image(result, 4)

    result = result.rotate(angle, resample=resample_mode, fillcolor=fill_value)
    # result = np.array(result)
    return result


def invert_custom_rotate(img, angle, cx, cy, patch_annotation):
    # image_center = cx, cy
    cx = int(cx)
    cy = int(cy)
    h, w = img.shape[:2]
    if len(patch_annotation.shape) == 3:
        assert patch_annotation.shape[2] == 3
        fill_value = (255, 255, 255)
        resample_mode = Image.BILINEAR
    else:
        fill_value = 0
        resample_mode = Image.NEAREST

    # pad to center image
    pad_dist = int(np.abs(cx - w / 2) * 2)
    if cx - w / 2 > 0:
        right = pad_dist
        left = 0
    else:
        left = pad_dist
        right = 0
    pad_dist = int(np.abs(cy - h / 2) * 2)
    if cy - h / 2 > 0:
        bot = pad_dist
        top = 0
    else:
        bot = pad_dist
        top = 0
    result = cv2.copyMakeBorder(img, top, bot, left, right, cv2.BORDER_CONSTANT, value=fill_value)

    # pad to not crop during rotate
    diag = np.sqrt(h ** 2 + w ** 2)
    if diag > result.shape[1]:
        left += int(diag - img.shape[1]) // 2
        right += int(diag - img.shape[1]) // 2

    # rotate with opencv fails for very big images
    # rot_mat = cv2.getRotationMatrix2D((result.shape[1] / 2, result.shape[0] / 2), angle, 1.0)
    # result = cv2.warpAffine(result, rot_mat, result.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=fill_value)

    # therefore, rotate image with PIL
    try:
        patch_annotation = Image.fromarray(patch_annotation)
    except ValueError:
        patch_annotation = create_big_PIL_image(patch_annotation, 4)
    patch_annotation = patch_annotation.rotate(-angle, resample=resample_mode, fillcolor=fill_value)
    w, h = patch_annotation.size
    patch_annotation = patch_annotation.crop((left, top, w - right, h - bot))
    # result = np.array(result)
    return patch_annotation


def merge_annotations_frompatches(data_id):
    global_lists = [[], [], [], []]
    region_name = get_region_from_id(data_id)
    nbio = region_name.split('_')[0]
    mask_path = f'E:/{main_folder}/thumbnails_processing/id{data_id}_{region_name}_7CCL.png'
    np_path = f'E:/{main_folder}/small_npys/id{data_id}_{region_name}_small.npy'
    img_path = f'E:/{main_folder}/tmp/id{data_id}_{region_name}_resized.png'
    bigimg_path = f'E:/{main_folder}/IgAN_acquired_images_(master)/{nbio}/{region_name}.tif'
    big_img_size = Image.open(bigimg_path).size
    mask = cv2.imread(mask_path, 0)
    # pick current CC only
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # just taking out the background
    nb_components = nb_components - 1
    orig_img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    inverted_gt = np.zeros(shape=orig_img.shape[:-1])

    for i in range(0, nb_components):
        img = orig_img.copy()
        glomeruli = np.where(np.load(np_path) == 1, 1, 0)

        ratio_h, ratio_w = np.array([big_img_size[1], big_img_size[0]]) / np.array(img.shape[:2])

        patches = []  # patches (bounding boxes) are saved as [x0, y0, w, h]
        single_CC_mask = np.zeros((output.shape))
        single_CC_mask[output == i + 1] = 1

        # rotate over momentum axis
        angle, center_x, center_y = get_rotation_angle(single_CC_mask)
        single_CC_mask = np.array(custom_rotate(single_CC_mask, angle, center_x, center_y)).astype(np.uint8)
        img = np.array(custom_rotate(img, angle, center_x, center_y))
        glomeruli = np.array(custom_rotate(glomeruli, angle, center_x, center_y))
        single_CC_glomeruli = (glomeruli * single_CC_mask).astype(np.uint8)

        # get bounding box of current ROTATED connected component
        _, rotated_output, rotated_stats, rotated_centroids = cv2.connectedComponentsWithStats(single_CC_mask, connectivity=8)

        cc_stats = rotated_stats[1]
        bb_l, bb_t, bb_w, bb_h = cc_stats[:4]

        if single_CC_glomeruli.max() == 0:
            gl_mean_diameter = 14
            gl_big_diameter = 28
        elif proj_name != 'IgAN-PAS':
            print("oooops")
        else:
            # get biggest glomerulus
            _, _, gl_stats, _ = cv2.connectedComponentsWithStats(single_CC_glomeruli, connectivity=8)
            gl_big_diameter = 2 * np.sqrt(gl_stats[1:, -1].max() / np.pi).astype(int)
            gl_mean_diameter = 2 * np.sqrt(gl_stats[1:, -1].mean() / np.pi).astype(int)

        # check if CC is too small to make a patch for it
        if 10 * gl_big_diameter ** 2 > bb_h * bb_w:
            continue

        w = int(12.5 * gl_mean_diameter)
        h = min(w, bb_h)
        if bb_h < 1.5 * w and bb_h > w:
            w = h = bb_h
        global_lists[0].append(h * ratio_h)
        global_lists[1].append(w * ratio_w)
        global_lists[2].append(h * ratio_h * w * ratio_w)

        # print(f'one bounding box would be {w * h / bb_w / bb_h} of the whole foreground')
        # TODO add check on bb size given the original image size
        if bb_w * bb_h < 1.75 * w * h:
            patches.append([bb_l, bb_t, bb_w, bb_h])
        else:
            # make grid
            # find cool d
            reps_w = bb_w // w
            d_w = 0
            while d_w < gl_big_diameter:
                reps_w += 1
                d_w = int((w * reps_w - bb_w) / (reps_w - 1))

            reps_h = bb_h // h
            d_h = 0
            while d_h < gl_big_diameter and bb_h != h:
                reps_h += 1
                d_h = int((h * reps_h - bb_h) / (reps_h - 1))

            c_x, c_y = bb_l, bb_t  # find current x and y
            # patches.append([c_x, c_y, w, h])
            w_step_counter = 0
            while w_step_counter < reps_w:  # move along x
                c_y = cc_stats[1]
                h_step_counter = 0
                while h_step_counter < reps_h:  # move along y
                    current_patch = single_CC_mask[c_y:c_y + h - d_h, c_x:c_x + w - d_w]
                    # debug_plot(current_patch, cmap='gray')
                    fullness = np.mean(current_patch)
                    if fullness > 0.3:
                        patches.append([c_x, c_y, w, h])

                    c_y = c_y + h - d_h
                    h_step_counter += 1
                    # if c_y > bb_t + bb_h or bb_h == h:
                    #     break

                c_x = c_x + w - d_w
                w_step_counter += 1

        # create whole_slide annotation
        glomeruli_root_path = f'E:/{main_folder}/patches_dataset/detected_glomeruli/id{data_id}_{region_name}_CC{i}'
        full_glomeruli_annotation = np.zeros(shape=img.shape[:-1])
        for j, bbox in enumerate(patches):
            tmp_glomeruli_annotation = np.zeros_like(full_glomeruli_annotation)
            # cv2.imwrite(patch_root_path + f'_patch_{j}.png', img_patch)
            gt_patch = np.load(glomeruli_root_path + f'_patch_{j}.npy')
            x, y, w, h = bbox
            gt_patch = cv2.resize(gt_patch, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
            try:
                tmp_glomeruli_annotation[y:y + h, x: x + w] = gt_patch
            except ValueError:
                lim_h, lim_w = tmp_glomeruli_annotation[y:y + h, x: x + w].shape
                tmp_glomeruli_annotation[y:y + h, x: x + w] = gt_patch[:lim_h, :lim_w]
            full_glomeruli_annotation += tmp_glomeruli_annotation

        cc_annotation = invert_custom_rotate(orig_img, angle, center_x, center_y, full_glomeruli_annotation)
        inverted_gt += cc_annotation

    inverted_gt = np.clip(inverted_gt, 0, 1)
    # debug_plot(inverted_gt)
    # debug_plot(orig_img)
    return inverted_gt


def draw_patches(bboxes, img):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        color = colors[i % len(colors)]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    return img


def process_patches(data_id):
    global_lists = [[], [], [], []]
    region_name = get_region_from_id(data_id)
    nbio = region_name.split('_')[0]
    mask_path = f'E:/{main_folder}/thumbnails_processing/id{data_id}_{region_name}_7CCL.png'
    np_path = f'E:/{main_folder}/small_npys/id{data_id}_{region_name}_small.npy'
    img_path = f'E:/{main_folder}/tmp/id{data_id}_{region_name}_resized.png'
    bigimg_path = f'E:/{main_folder}/IgAN_acquired_images_(master)/{nbio}/{region_name}.tif'
    big_img_size = Image.open(bigimg_path).size
    mask = cv2.imread(mask_path, 0)
    # pick current CC only
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # just taking out the background
    nb_components = nb_components - 1

    for i in range(0, nb_components):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        glomeruli = np.where(np.load(np_path) == 1, 1, 0)

        ratio_h, ratio_w = np.array([big_img_size[1], big_img_size[0]]) / np.array(img.shape[:2])

        patches = []  # patches (bounding boxes) are saved as [x0, y0, w, h]
        single_CC_mask = np.zeros((output.shape))
        single_CC_mask[output == i + 1] = 1

        # rotate over momentum axis
        angle, center_x, center_y = get_rotation_angle(single_CC_mask)
        single_CC_mask = np.array(custom_rotate(single_CC_mask, angle, center_x, center_y)).astype(np.uint8)
        img = np.array(custom_rotate(img, angle, center_x, center_y))
        glomeruli = np.array(custom_rotate(glomeruli, angle, center_x, center_y))
        single_CC_glomeruli = (glomeruli * single_CC_mask).astype(np.uint8)

        # get bounding box of current ROTATED connected component
        _, rotated_output, rotated_stats, rotated_centroids = cv2.connectedComponentsWithStats(single_CC_mask, connectivity=8)

        cc_stats = rotated_stats[1]
        bb_l, bb_t, bb_w, bb_h = cc_stats[:4]

        if single_CC_glomeruli.max() == 0:
            gl_mean_diameter = 14
            gl_big_diameter = 28
        elif proj_name != 'IgAN-PAS':
            print("oooops")
        else:
            # get biggest glomerulus
            _, _, gl_stats, _ = cv2.connectedComponentsWithStats(single_CC_glomeruli, connectivity=8)
            gl_big_diameter = 2 * np.sqrt(gl_stats[1:, -1].max() / np.pi).astype(int)
            gl_mean_diameter = 2 * np.sqrt(gl_stats[1:, -1].mean() / np.pi).astype(int)

        # check if CC is too small to make a patch for it
        if 10 * gl_big_diameter ** 2 > bb_h * bb_w:
            continue

        w = int(12.5 * gl_mean_diameter)
        h = min(w, bb_h)
        if bb_h < 1.5 * w and bb_h > w:
            w = h = bb_h
        global_lists[0].append(h * ratio_h)
        global_lists[1].append(w * ratio_w)
        global_lists[2].append(h * ratio_h * w * ratio_w)

        # print(f'one bounding box would be {w * h / bb_w / bb_h} of the whole foreground')
        # TODO add check on bb size given the original image size
        if bb_w * bb_h < 1.75 * w * h:
            patches.append([bb_l, bb_t, bb_w, bb_h])
        else:
            # make grid
            # find cool d
            reps_w = bb_w // w
            d_w = 0
            while d_w < gl_big_diameter:
                reps_w += 1
                d_w = int((w * reps_w - bb_w) / (reps_w - 1))

            reps_h = bb_h // h
            d_h = 0
            while d_h < gl_big_diameter and bb_h != h:
                reps_h += 1
                d_h = int((h * reps_h - bb_h) / (reps_h - 1))

            c_x, c_y = bb_l, bb_t  # find current x and y
            # patches.append([c_x, c_y, w, h])
            w_step_counter = 0
            while w_step_counter < reps_w:  # move along x
                c_y = cc_stats[1]
                h_step_counter = 0
                while h_step_counter < reps_h:  # move along y
                    current_patch = single_CC_mask[c_y:c_y + h - d_h, c_x:c_x + w - d_w]
                    # debug_plot(current_patch, cmap='gray')
                    fullness = np.mean(current_patch)
                    if fullness > 0.3:
                        patches.append([c_x, c_y, w, h])

                    c_y = c_y + h - d_h
                    h_step_counter += 1
                    # if c_y > bb_t + bb_h or bb_h == h:
                    #     break

                c_x = c_x + w - d_w
                w_step_counter += 1

        img = draw_patches(patches, img)
        angle = -angle
        # single_CC_mask = np.array(custom_rotate(single_CC_mask, angle, img.shape[1] / 2, img.shape[0] / 2))
        img = np.array(custom_rotate(img, angle, img.shape[1] / 2, img.shape[0] / 2))
        # glomeruli = np.array(custom_rotate(glomeruli, angle, img.shape[1] / 2, img.shape[0] / 2))
        writable_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'E:/{main_folder}/tmp/id{data_id}_{region_name}_8BBS_{i}.png', writable_img)


def create_patches_dataset(data_id):
    global_lists = [[], [], []]
    global reshape_w
    region_name = get_region_from_id(data_id)
    nbio = region_name.split('_')[0]
    mask_path = f'E:/{main_folder}/thumbnails_processing/id{data_id}_{region_name}_7CCL.png'
    np_path = f'E:/{main_folder}/small_npys/id{data_id}_{region_name}_small.npy'
    json_path = f'E:/{main_folder}/json_annotations/{region_name}.txt'
    img_path = f'E:/{main_folder}/tmp/id{data_id}_{region_name}_resized.png'
    bigimg_path = f'E:/{main_folder}/IgAN_acquired_images_(master)/{nbio}/{region_name}.tif'

    original_big_img = Image.open(bigimg_path)
    big_img_size = original_big_img.size

    mask = cv2.imread(mask_path, 0)
    # pick current CC only
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # just taking out the background
    nb_components = nb_components - 1

    # original_big_glomeruli = parse_geojsn(json_path, big_img_size)
    #
    # original_big_img = np.array(original_big_img)
    # original_big_img = cv2.cvtColor(original_big_img, cv2.COLOR_RGB2BGR)
    # original_big_glomeruli = np.array(original_big_glomeruli)
    for i in range(0, nb_components):
        # if i == nb_components - 1:
        #     big_img = original_big_img
        #     big_glomeruli = original_big_glomeruli
        # else:
        #     big_img = copy.deepcopy(original_big_img)
        #     big_glomeruli = copy.deepcopy(original_big_glomeruli)

        big_img = Image.open(bigimg_path)
        big_glomeruli = parse_geojsn(json_path, big_img_size)

        big_img = np.array(big_img)
        big_img = cv2.cvtColor(big_img, cv2.COLOR_RGB2BGR)
        big_glomeruli = np.where(np.array(big_glomeruli) == 1, 1, 0)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        glomeruli = np.where(np.load(np_path) == 1, 1, 0)

        ratio_h, ratio_w = np.array([big_img_size[1], big_img_size[0]]) / np.array(img.shape[:2])

        patches = []  # patches (bounding boxes) are saved as [x0, y0, w, h]
        single_CC_mask = np.zeros((output.shape))
        single_CC_mask[output == i + 1] = 1

        # crop single CC, added to handle very small CCs
        _, _, small_cc_stats, _ = cv2.connectedComponentsWithStats(single_CC_mask.astype(np.uint8), connectivity=8)
        cc_l, cc_t, cc_w, cc_h, _ = small_cc_stats[1]
        # if cc_w * cc_h * 5 < single_CC_mask.size:
        single_CC_mask = single_CC_mask[cc_t:cc_t + cc_h, cc_l:cc_l + cc_w]
        img = img[cc_t:cc_t + cc_h, cc_l:cc_l + cc_w]
        glomeruli = glomeruli[cc_t:cc_t + cc_h, cc_l:cc_l + cc_w]

        cc_l, cc_w = int(cc_l * ratio_w), int(cc_w * ratio_w)
        cc_t, cc_h = int(cc_t * ratio_h), int(cc_h * ratio_h)
        big_glomeruli = big_glomeruli[cc_t:cc_t + cc_h, cc_l:cc_l + cc_w]
        big_img = big_img[cc_t:cc_t + cc_h, cc_l:cc_l + cc_w]

        # rotate over momentum axis
        angle, center_x, center_y = get_rotation_angle(single_CC_mask)
        single_CC_mask = np.array(custom_rotate(single_CC_mask, angle, center_x, center_y)).astype(np.uint8)
        img = np.array(custom_rotate(img, angle, center_x, center_y))
        glomeruli = np.array(custom_rotate(glomeruli, angle, center_x, center_y))
        single_CC_glomeruli = (glomeruli * single_CC_mask).astype(np.uint8)

        big_glomeruli = custom_rotate(big_glomeruli, angle, center_x * ratio_w, center_y * ratio_h)
        big_img = custom_rotate(big_img, angle, center_x * ratio_w, center_y * ratio_h)

        # get bounding box of current ROTATED connected component
        _, rotated_output, rotated_stats, rotated_centroids = cv2.connectedComponentsWithStats(single_CC_mask, connectivity=8)

        cc_stats = rotated_stats[1]
        bb_l, bb_t, bb_w, bb_h = cc_stats[:4]

        if single_CC_glomeruli.max() == 0:
            gl_mean_diameter = 14
            gl_big_diameter = 28
        elif proj_name != 'IgAN-PAS':
            print("oooops")
        else:
            # get biggest glomerulus
            _, _, gl_stats, _ = cv2.connectedComponentsWithStats(single_CC_glomeruli, connectivity=8)
            gl_big_diameter = 2 * np.sqrt(gl_stats[1:, -1].max() / np.pi).astype(int)
            gl_mean_diameter = 2 * np.sqrt(gl_stats[1:, -1].mean() / np.pi).astype(int)

        # check if CC is too small to make a patch for it
        if 10 * gl_big_diameter ** 2 > bb_h * bb_w:
            continue

        w = int(12.5 * gl_mean_diameter)
        h = min(w, bb_h)
        if bb_h < 1.5 * w and bb_h > w:
            w = h = bb_h
        # global_lists[0].append(h * ratio_h)
        # global_lists[1].append(w * ratio_w)
        # global_lists[2].append(h * ratio_h * w * ratio_w)

        # print(f'one bounding box would be {w * h / bb_w / bb_h} of the whole foreground')
        # TODO add check on bb size given the original image size
        if bb_w * bb_h < 1.75 * w * h:
            patches.append([bb_l, bb_t, bb_w, bb_h])
        else:
            # make grid
            # find cool d
            reps_w = bb_w // w
            d_w = 0
            while d_w < gl_big_diameter:
                reps_w += 1
                d_w = int((w * reps_w - bb_w) / (reps_w - 1))

            reps_h = bb_h // h
            d_h = 0
            while d_h < gl_big_diameter and bb_h != h:
                reps_h += 1
                d_h = int((h * reps_h - bb_h) / (reps_h - 1))

            c_x, c_y = bb_l, bb_t  # find current x and y
            # patches.append([c_x, c_y, w, h])
            w_step_counter = 0
            while w_step_counter < reps_w:  # move along x
                c_y = cc_stats[1]
                h_step_counter = 0
                while h_step_counter < reps_h:  # move along y
                    current_patch = single_CC_mask[c_y:c_y + h - d_h, c_x:c_x + w - d_w]
                    # debug_plot(current_patch, cmap='gray')
                    fullness = np.mean(current_patch)
                    if fullness > 0.3:
                        patches.append([c_x, c_y, w, h])

                    c_y = c_y + h - d_h
                    h_step_counter += 1
                    # if c_y > bb_t + bb_h or bb_h == h:
                    #     break

                c_x = c_x + w - d_w
                w_step_counter += 1

        patch_root_path = f'E:/{main_folder}/patches_dataset/images/id{data_id}_{region_name}_CC{i}'
        gt_root_path = f'E:/{main_folder}/patches_dataset/gts/id{data_id}_{region_name}_CC{i}'
        for j, bbox in enumerate(patches):
            x, y, w, h = bbox
            crop_rectangle = [x * ratio_w, y * ratio_h, x * ratio_w + w * ratio_w, y * ratio_h + h * ratio_h]
            crop_rectangle = tuple([int(f) for f in crop_rectangle])
            img_patch = np.array(big_img.crop(crop_rectangle), dtype=np.uint8)
            # img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB)
            gt_patch = np.array(big_glomeruli.crop(crop_rectangle), dtype=np.uint8)
            reshape_ratio = global_reshape_w / w / ratio_w
            if reshape_ratio < 1:
                img_patch = cv2.resize(img_patch, (0, 0), fx=reshape_ratio, fy=reshape_ratio)
                gt_patch = cv2.resize(gt_patch, (0, 0), fx=reshape_ratio, fy=reshape_ratio)

            # img_patch = cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR)
            # debug_plot(img_patch)
            # debug_plot(cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR))
            # cv2.imwrite('RGB.png', img_patch)
            # cv2.imwrite('BGR.png', cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR))
            # return
            # debug_plot(gt_patch)
            cv2.imwrite(patch_root_path + f'_patch_{j}.png', img_patch)
            np.save(gt_root_path + f'_patch_{j}_gt.npy', gt_patch)

        # debug_plot(draw_patches(patches, img))


def add_glomeruli(data_id, glomeruli_gt):
    img_name = get_region_from_id(data_id)
    np_path = f'E:/{main_folder}/small_npys/id{data_id}_{img_name}_small.npy'
    new_np_path = f'E:/{main_folder}/small_npys_wdetectedglom/id{data_id}_{img_name}_small_wgl.npy'
    manual_gt = np.load(np_path)
    full_annotation = np.where(glomeruli_gt == 1, 1, manual_gt)
    np.save(new_np_path, full_annotation.astype(np.uint8))


def create_big_PIL_image(big_arr, n):
    big_img = Image.new('RGB', (big_arr.shape[1], big_arr.shape[0]))
    x_offset = 0
    for im in np.array_split(big_arr, n, axis=1):
        pimg = Image.fromarray(im)
        temp_box = (x_offset, 0)
        big_img.paste(pimg, temp_box)
        x_offset += pimg.size[0]
    return big_img


class Tree(collections.defaultdict):
    def __init__(self, value=None):
        super(Tree, self).__init__(Tree)
        self.value = value

    def _get_el(self, i):
        return list(self.keys())[i], self[list(self.keys())[i]]

    def get_node(self, i):
        return self[list(self.keys())[i]]

    def is_leaf(self, el):
        if 'images' in el:
            return True
        return False

    def get_first_leaf(self):
        new_el = self.get_node(0)
        if self.is_leaf(new_el):
            return new_el
        else:
            return new_el.get_first_leaf()

    def get_subtree(self, idxs):
        subtree = Tree()
        for idx in idxs:
            k, v = self._get_el(idx)
            subtree[k] = v
        return subtree

    def images_len(self):
        cnt = 0
        for el in self:
            if type(self[el]) is str:
                cnt += 1
            else:
                cnt += len(self[el])
        return cnt

    def __len__(self):
        return len(self.keys())

    def __contains__(self, item):
        for el in self:
            if self.is_leaf(self[el]):
                if item == self[el]:
                    return True
            else:
                return item in self[el]
        return False


def get_bios_dataset(dir):
    bios = {}
    filenames = glob.glob(dir + '*.png')
    labels = get_labels(dir)
    for fullpath in filenames:
        f = os.path.basename(fullpath)
        bio = f.split('_')[1]
        id = f.split('_')[0][2:]
        if bio in bios.keys():
            continue
        try:
            label = labels[bio]
        except KeyError:
            continue

        if label['fup'] == '':
            label['fup'] = '0'

        bios[bio] = {'bio': bio,
                     'id': id,
                     'ESRD': label['ESRD'],
                     'fup': label['fup']}
    bios = sorted(bios.values(), key=lambda item: int(item['id']))
    return bios


def get_patches_dataset(dir):
    main_tree = Tree()
    imgs = {}
    filenames = glob.glob(dir + '*.png')
    labels = get_labels(dir)
    for fullpath in filenames:
        f = os.path.basename(fullpath)
        bio = f.split('_')[1]
        id = f.split('_')[0][2:]
        region = f.split('Regione')[1][1]
        cc = f.split('_CC')[1].split('_')[0]
        patch = f.split('patch_')[1].split('.')[0]
        try:
            label = labels[bio]
        except KeyError:
            continue
        main_tree[bio][id][cc][patch] = 'images/' + f

        if label['fup'] == '':
            label['fup'] = '0'

        imgs['images/' + f] = {'bio': bio,
                               'id': id,
                               'region': region,
                               'cc': cc,
                               'patch': patch,
                               'ESRD': label['ESRD'],
                               'fup': label['fup']}
        imgs = dict(sorted(imgs.items(), key=lambda item: int(item[1]['id'])))
    return main_tree, imgs


def get_glomeruli_dataset(dir):
    main_tree = Tree()
    imgs = {}
    filenames = glob.glob(dir + '*.png')
    labels = get_labels(dir)
    for fullpath in filenames:
        f = os.path.basename(fullpath)
        bio = f.split('_')[1]
        id = f.split('_')[0][2:]
        region = f.split('Regione')[1][1]
        gl = f.split('gl_')[1].split('.')[0]
        try:
            label = labels[bio]
        except KeyError:
            continue
        main_tree[bio][id][gl] = 'images/' + f

        if label['fup'] == '':
            label['fup'] = '0'

        imgs['images/' + f] = {'bio': bio,
                               'id': id,
                               'region': region,
                               'gl': gl,
                               'ESRD': label['ESRD'],
                               'fup': label['fup']}
        imgs = dict(sorted(imgs.items(), key=lambda item: int(item[1]['id'])))
    return main_tree, imgs


def get_patches_dataset_splits(dir):
    s_dim = (10, 90)
    main_tree, imgs = get_patches_dataset(dir)

    np.random.seed(10)
    idxs = np.arange(len(main_tree.keys()))
    np.random.shuffle(idxs)
    val = main_tree.get_subtree(idxs[:s_dim[0]])
    test = main_tree.get_subtree(idxs[s_dim[0]:s_dim[0] + s_dim[1]])
    train = main_tree.get_subtree(idxs[s_dim[0] + s_dim[1]:])
    return (train, val, test), imgs


def split_by_label(main_tree, imgs, indexes, time_thresh=10):
    true_indexes = []
    false_indexes = []
    censored_indexes = []
    for id in indexes:
        el = imgs[main_tree.get_node(id).get_first_leaf()]
        if el['fup'] in ['']:
            print("removing dataset row with no fup")
            continue
        if el['ESRD'] == 'FALSE':
            if float(el['fup']) >= time_thresh:
                false_indexes.append(id)
            else:
                censored_indexes.append(id)
        elif float(el['fup']) <= time_thresh:
            true_indexes.append(id)
        elif float(el['fup']) <= time_thresh * 2:  # ADDING this since we use soft label for 10<fup<20
            censored_indexes.append(id)
        else:
            false_indexes.append(id)

    return true_indexes, false_indexes, censored_indexes


def split_bio_list_by_label(bios, indexes, time_thresh=10):
    true_indexes = []
    false_indexes = []
    censored_indexes = []
    for id in indexes:
        el = bios[id]
        if el['fup'] in ['']:
            print("removing dataset row with no fup")
            continue
        if el['ESRD'] == 'FALSE':
            if float(el['fup']) >= time_thresh:
                false_indexes.append(id)
            else:
                censored_indexes.append(id)
        elif float(el['fup']) <= time_thresh:
            true_indexes.append(id)
        elif float(el['fup']) <= time_thresh * 2:  # ADDING this since we use soft label for 10<fup<20
            censored_indexes.append(id)
        else:
            false_indexes.append(id)

    return true_indexes, false_indexes, censored_indexes


def list_pop(l, n):
    ret = []
    for i in range(n):
        ret.append(l.pop(0))
    return ret


def get_patches_dataset_splits_fup(dir):
    s_dim = {'val': 10, 'test': 90}
    trues_dim = {'val': 2, 'test': 9}
    false_dim = {'val': s_dim['val'] - trues_dim['val'], 'test': s_dim['test'] - trues_dim['test']}
    splits_idxs = {'val': [], 'test': []}
    main_tree, imgs = get_patches_dataset(dir)
    seed = 10
    np.random.seed(seed)
    idxs = np.arange(len(main_tree.keys()))
    np.random.shuffle(idxs)
    true_idxs, false_idxs, censored_idxs = split_by_label(main_tree, imgs, idxs)
    for s in splits_idxs.keys():
        splits_idxs[s] = main_tree.get_subtree(list_pop(true_idxs, trues_dim[s]) + list_pop(false_idxs, false_dim[s]))
    splits_idxs['train'] = main_tree.get_subtree(true_idxs + false_idxs + censored_idxs)

    return splits_idxs, imgs


def get_bios_dataset_splits_fup(dir):
    s_dim = {'test': 100}
    trues_dim = {'test': 20}
    false_dim = {'test': s_dim['test'] - trues_dim['test']}
    splits_idxs = {'test': []}
    bios = get_bios_dataset(dir)
    seed = 10
    np.random.seed(seed)
    idxs = np.arange(len(bios))
    np.random.shuffle(idxs)
    true_idxs, false_idxs, censored_idxs = split_bio_list_by_label(bios, idxs)
    for s in splits_idxs.keys():
        splits_idxs[s] = list_pop(true_idxs, trues_dim[s]) + list_pop(false_idxs, false_dim[s])
    splits_idxs['train'] = true_idxs + false_idxs + censored_idxs

    return splits_idxs, bios


def explore_patches_dataset(dir=f'E:/{main_folder}/patches_dataset/images/'):
    splits, imgs = get_patches_dataset_splits(dir)
    global_lists = [[], [], [], []]
    for i, tree in enumerate(splits):
        print(len(tree))
        bios = dict(tree)
        temp_list = list(bios.values())
        global_lists[0].append([])
        global_lists[1].append([])
        global_lists[2].append([])
        for region in temp_list:
            global_lists[0][i].append(len(region.keys()))
            for cc in list(region.values()):
                global_lists[1][i].append(len(cc.keys()))
                for patch in list(cc.values()):
                    global_lists[2][i].append(len(patch.keys()))
        global_lists[3].append([len(bio) for bio in tree.values()])

    for i, global_list in enumerate(global_lists):
        plt.hist(global_list, bins=20, histtype='bar', stacked=False)
        plt.show()


def explore_patches_dataset_fup(dir=f'E:/{main_folder}/patches_dataset/images/'):
    main_tree, imgs = get_patches_dataset(dir)
    labels_d = {'': 0, '0': 0, '5': 0, '10': 0, 'False': 0, '5_censored': 0, '10_censored': 0}
    for bio in main_tree.keys():
        el = main_tree[bio].get_first_leaf()
        if imgs[el]['fup'] in ['', '0']:
            labels_d[imgs[el]['fup']] += 1
        elif imgs[el]['ESRD'] == 'FALSE':
            if float(imgs[el]['fup']) >= 10:
                labels_d['False'] += 1
            elif float(imgs[el]['fup']) >= 5:
                labels_d['10_censored'] += 1
            else:
                labels_d['5_censored'] += 1
        else:
            if float(imgs[el]['fup']) > 10:
                print(float(imgs[el]['fup']))
                labels_d['False'] += 1
            elif float(imgs[el]['fup']) > 5:
                labels_d['10'] += 1
            else:
                labels_d['5'] += 1

    for k, v in zip(labels_d.keys(), labels_d.values()):
        print(f'{k}: {v}')


def explore_glomeruli_dataset_fup(dir=f'E:/{main_folder}/glomeruli_dataset/images/'):
    main_tree, imgs = get_glomeruli_dataset(dir)
    labels_d = {'': 0, '0': 0, '5': 0, '10': 0, 'False': 0, '5_censored': 0, '10_censored': 0}
    for bio in main_tree.keys():
        el = main_tree[bio].get_first_leaf()
        if imgs[el]['fup'] in ['', '0']:
            labels_d[imgs[el]['fup']] += 1
        elif imgs[el]['ESRD'] == 'FALSE':
            if float(imgs[el]['fup']) >= 10:
                labels_d['False'] += 1
            elif float(imgs[el]['fup']) >= 5:
                labels_d['10_censored'] += 1
            else:
                labels_d['5_censored'] += 1
        else:
            if float(imgs[el]['fup']) > 10:
                print(float(imgs[el]['fup']))
                labels_d['False'] += 1
            elif float(imgs[el]['fup']) > 5:
                labels_d['10'] += 1
            else:
                labels_d['5'] += 1

    for k, v in zip(labels_d.keys(), labels_d.values()):
        print(f'{k}: {v}')

    gl_histo = [bio.images_len() for bio in main_tree.values()]
    plt.hist(gl_histo, bins=100)
    plt.show()


def get_labels(dir):
    fname = f'E:/Bignefro/patches_dataset/' + 'patients_final_output.csv'
    d = {}
    with open(fname) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=';')
        for i, row in enumerate(readCSV):
            if i == 0:
                col_names = {}
                n = 0
                for el in row:
                    col_names[el] = n
                    n += 1
                continue
            label_d = {}
            label_d['ESRD'] = row[col_names['ESRD']]
            label_d['fup'] = row[col_names['fup_ESRD']]
            d[row[col_names['numbio']].zfill(4)] = label_d

    return d


def create_yaml_dataset_file(dir=f'E:/{main_folder}/patches_dataset/images/'):
    yaml_d = {'name': 'big_nephro'}
    yaml_imgs = []
    splits, imgs = get_patches_dataset_splits(dir)
    yaml_splits = [[], [], []]
    for i, img in enumerate(imgs.keys()):
        new_img = {'location': img,
                   'label': img.replace('images', 'gts').replace('.png', '_gt.npy'),
                   'values': imgs[img]}
        yaml_imgs.append(new_img)
        for j, s in enumerate(splits):
            if imgs[img]['bio'] in s.keys():
                yaml_splits[j].append(i)

    yaml_d['images'] = yaml_imgs
    yaml_d['split'] = {'training': yaml_splits[0],
                       'validation': yaml_splits[1],
                       'test': yaml_splits[2]}
    with open(f'E:/{main_folder}/patches_dataset/big_nephro_dataset.yml', 'w') as file:
        yaml.dump(yaml_d, file)


def create_yaml_dataset_file_fup(dir=f'E:/{main_folder}/patches_dataset/images/'):
    yaml_d = {'name': 'big_nephro'}
    yaml_imgs = []
    splits, imgs = get_patches_dataset_splits_fup(dir)
    splits = (splits['train'], splits['val'], splits['test'])  # THIS IS DONE FOR RETROCOMPATIBILITY
    yaml_splits = [[], [], []]
    for i, img in enumerate(imgs.keys()):
        new_img = {'location': img,
                   'label': img.replace('images', 'gts').replace('.png', '_gt.npy'),
                   'values': imgs[img]}
        yaml_imgs.append(new_img)
        for j, s in enumerate(splits):
            if imgs[img]['bio'] in s.keys():
                yaml_splits[j].append(i)

    yaml_d['images'] = yaml_imgs
    yaml_d['split'] = {'training': yaml_splits[0],
                       'validation': yaml_splits[1],
                       'test': yaml_splits[2]}
    with open(f'E:/{main_folder}/patches_dataset/big_nephro_dataset.yml', 'w') as file:
        yaml.dump(yaml_d, file)


def create_yaml_dataset_file_fup_per_bio(dir=f'E:/{main_folder}/patches_dataset/images/'):
    yaml_d = {'name': 'big_nephro'}
    yaml_bios = []
    splits, bios = get_bios_dataset_splits_fup(dir)
    for k, v in splits.items():
        splits[k] = [int(el) for el in v]
    for i, bio in enumerate(bios):
        new_img = {'values': bio}
        yaml_bios.append(new_img)


    yaml_d['bios'] = bios
    yaml_d['split'] = {'training': splits['train'],
                       'test': splits['test']}
    with open(f'E:/{main_folder}/patches_dataset/big_nephro_bios_dataset.yml', 'w') as file:
        yaml.dump(yaml_d, file)


def BGR2RGB(full_dir):
    global global_counters
    global_counters.append(0)
    fnames = glob.glob(full_dir + '/*.png')
    for f in fnames:
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f.replace('/images', '/rgb'), img)
        global_counters[0] += 1


def create_glomeruli_dataset(data_id):
    global_lists = [[], [], []]
    global reshape_w
    region_name = get_region_from_id(data_id)
    nbio = region_name.split('_')[0]
    np_path = f'E:/{main_folder}/small_npys_wdetectedglom/id{data_id}_{region_name}_small_wgl.npy'
    bigimg_path = f'E:/{main_folder}/IgAN_acquired_images_(master)/{nbio}/{region_name}.tif'
    glomerulus_root_path = f'E:/{main_folder}/glomeruli_dataset/images/id{data_id}_{region_name}'

    original_big_img = Image.open(bigimg_path)
    big_img = Image.open(bigimg_path)

    big_img = np.array(big_img)
    big_img = cv2.cvtColor(big_img, cv2.COLOR_RGB2BGR)
    big_img_size = original_big_img.size

    glomeruli_mask = np.where(np.load(np_path) == 1, 1, 0).astype(np.uint8)
    ratio_h, ratio_w = np.array([big_img_size[1], big_img_size[0]]) / np.array(glomeruli_mask.shape[:2])

    # iterate over single glomeruli
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(glomeruli_mask, connectivity=8)
    for i in range(1, nb_components):
        bb_l, bb_t, bb_w, bb_h = stats[i][:4]
        bb_l *= ratio_w
        bb_w *= ratio_w
        bb_t *= ratio_h
        bb_h *= ratio_h
        bb_dim = int(np.round(max(bb_w, bb_h)) * 1.2)
        bb_l = int(bb_l - np.round(max(0, bb_dim - bb_w) / 2))
        bb_t = int(bb_t - np.round(max(0, bb_dim - bb_h) / 2))
        bb_limits = [bb_t, bb_t + bb_dim, bb_l, bb_l + bb_dim]
        # bb_limits = [int(np.round(l)) for l in bb_limits]
        gl_bb = big_img[bb_limits[0]:bb_limits[1], bb_limits[2]:bb_limits[3]]
        if gl_bb.size < 75 * 75 * 3:
            print("not writing" + glomerulus_root_path + f'_gl_{i}.png')
            continue
        if gl_bb.size > 512 * 512 * 3:
            gl_bb = cv2.resize(gl_bb, dsize=(512, 512), interpolation=cv2.INTER_LANCZOS4)
        # debug_plot(gl_bb)
        cv2.imwrite(glomerulus_root_path + f'_gl_{i}.png', gl_bb)


def processing_pipeline(data_id):
    segment_region(data_id)
    processandresize_big_image(data_id)
    process_patches(data_id)
    # create_patches_dataset(data_id)  # this takes a lot of time!!!


def merge_annotations_pipeline(data_id):
    glomeruli_annotation = merge_annotations_frompatches(data_id)
    add_glomeruli(data_id, glomeruli_annotation)


def glomeruli_processing_pipeline(data_id):
    merge_annotations_pipeline(data_id)
    create_glomeruli_dataset(data_id)


if __name__ == '__main__':
    global_reshape_w = 2000


    global_counters.append(0)
    global_counters.append(0)
    global_counters.append(0)

    perform_on_whole_dataset(processing_pipeline)

    create_yaml_dataset_file_fup_per_bio()
    for i, global_counter in enumerate(global_counters):
        print(f'global counter {i}: {global_counter}')
    for i, global_list in enumerate(global_lists):
        plt.hist(global_list, bins=100)
        plt.show()
