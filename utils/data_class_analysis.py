import pathlib
import cv2
import pandas as pd
import numpy as np
from utils import DATA_SPLITS, CLASS_INFO, CLASS_NAMES, heatmap, annotate_heatmap, remap_experiment, mask_to_colormap
import matplotlib.pyplot as plt


PATH = pathlib.Path('C:\\Users\\RVIM_Claudio\\Documents\\cadis\\data\\data_info')


def get_class_numbers(data_path: pathlib.Path = None):
    data_path = PATH if data_path is None else data_path
    label_list = []
    subfolders = [[f.name, f] for f in data_path.iterdir() if f.is_dir()]
    for folder_name, folder_path in subfolders:
        for file in (folder_path / 'Labels').iterdir():
            label = cv2.imread(str(file), 0)
            record_class_list = [int(folder_name[-2:]), folder_name, file.name]
            for i in range(36):
                record_class_list.extend([np.sum(label == i)])
            label_list.append(record_class_list)
            if np.sum(record_class_list[3:]) == label.size:
                print(str(file) + '   Test: passed')
            else:
                print("Unknown class found - aborting.")
                exit()
    df = pd.DataFrame(label_list, columns=['vid_num', 'folder_name', 'file_name']+CLASS_NAMES[0])
    df.to_pickle(data_path / 'label_table.pkl', compression='gzip')
    df.to_csv(data_path / 'label_table.csv')


def analyse_class_numbers(data_path: pathlib.Path = None):
    data_path = PATH if data_path is None else data_path
    df = pd.read_pickle(data_path / 'label_table.pkl', compression='gzip')
    videos = df.folder_name.unique()
    num_vids, num_frames = len(videos), df.shape[0]
    class_nums = [len(CLASS_INFO[1][0]), len(CLASS_INFO[2][0]), len(CLASS_INFO[3][0]), 36]
    vid_lengths = []
    for video in videos:
        vid_lengths.append(df[df['folder_name'] == video].shape[0])
    text_scale = 1
    pad_text_w, pad_text_h = 100 * text_scale, 30 * text_scale
    pad_vid, pad_in, pad_out, pix_h, pix_w = 10, 30, 50, 5, 3
    # Resulting image consists of:
    # height: pad_out * 2, (exp1 + exp2 + exp3 + actual classes) * pix_size, pad_in * 3
    # width: pad_out * 2, (num_vids - 1) * pad_vid, num_all_frames
    h = 2 * pad_out + pad_text_h + np.sum(class_nums) * pix_h + 3 * pad_in
    w = 2 * pad_out + pad_text_w + (num_vids - 1) * pad_vid + num_frames * pix_w
    img = np.zeros((h, w, 3), 'uint8')  # Will become the HSV image
    h_val = 120
    s_min = 0
    # H chosen as 120 (blue), S is [55, 255] for [0, 100%] with 55 set to 0 afterwards to show 0 values, V is 255
    for i in range(num_frames):
        values = df.iloc[i].loc['Pupil':'Iris Hooks'].to_numpy()
        for j in range(4):
            if j in [0, 1, 2]:  # If exp 1 to 3, remap values
                vals = np.zeros(class_nums[j], 'i')
                for k in range(class_nums[j]):
                    if k == 17 or k == 25:  # Correct for annoying class numbering in exp 2 and 3: the last class is 255
                        vals[k] = np.sum(values[CLASS_INFO[j + 1][0][255]])
                    else:
                        vals[k] = np.sum(values[CLASS_INFO[j + 1][0][k]])
            else:
                vals = values
            perc_vals = vals / np.sum(vals)
            sat_vals = perc_vals * (255 - s_min) + s_min
            curr_pos_h = pad_out + pad_text_h + int(np.sum(class_nums[:j])) * pix_h + len(class_nums[:j]) * pad_in
            curr_pos_w = pad_out + pad_text_w + i * pix_w + (int(df.iloc[i].loc['folder_name'][-2:]) - 1) * pad_vid
            img[curr_pos_h:curr_pos_h + pix_h * class_nums[j], curr_pos_w:curr_pos_w + pix_w, 0] = h_val
            img[curr_pos_h:curr_pos_h + pix_h * class_nums[j], curr_pos_w:curr_pos_w + pix_w, 1] =\
                np.tile(np.repeat(sat_vals, pix_h)[:, np.newaxis], (1, pix_w))
            img[curr_pos_h:curr_pos_h + pix_h * class_nums[j], curr_pos_w:curr_pos_w + pix_w, 2] = 255
    img[img == s_min] = 0
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    for i, text in enumerate(['exp1', 'exp2', 'exp3', 'orig']):
        h_pos = pad_out + pad_text_h + i * pad_in + int(np.sum(class_nums[:i])) * pix_h +\
                pix_h * class_nums[i] // 2 + 11 * text_scale
        img = cv2.putText(img, text, (pad_out, h_pos),
                          fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 255, 255))
    for i, video in enumerate(videos):
        w_pos = pad_out + pad_text_w + i * pad_vid + int(np.sum(vid_lengths[:i])) * pix_w +\
                pix_w * vid_lengths[i] // 2 - 70 * text_scale
        img = cv2.putText(img, video, (w_pos, pad_out),
                          fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 255, 255))

    cv2.imwrite(str(data_path / 'class_image.png'), img)


def analyse_set_splits(data_path: pathlib.Path = None):
    data_path = PATH if data_path is None else data_path
    df = pd.read_pickle(data_path / 'label_table.pkl', compression='gzip')
    for i, split in enumerate(DATA_SPLITS):
        with pd.ExcelWriter(data_path / 'split_{}_analysis.xlsx'.format(i)) as writer:
            for j in range(4):
                train_num_frames, train_classes, valid_num_frames, valid_classes =\
                    get_train_valid_classes_from_split(df, *split, j, 1)
                split_df = pd.DataFrame([[train_num_frames, *train_classes], [valid_num_frames, *valid_classes]],
                                        columns=['num_frames'] + CLASS_NAMES[j])
                split_df.to_excel(writer, sheet_name='experiment_{}'.format(j))


def get_train_valid_classes_from_split(df: pd.DataFrame, train_list: list, valid_list: list, exp: int, normalise: bool)\
        -> list:
    """returns number of records in the training set and the class distribution, and same for validation set"""
    train = df.loc[df['vid_num'].isin(train_list)].iloc[:, 3:].to_numpy()
    train = remap_classes(train, exp)
    valid = df.loc[df['vid_num'].isin(valid_list)].iloc[:, 3:].to_numpy()
    valid = remap_classes(valid, exp)
    return_list = [train.shape[0], np.sum(train, axis=0), valid.shape[0], np.sum(valid, axis=0)]
    if normalise:
        return_list[1] = return_list[1] / np.sum(return_list[1])
        return_list[3] = return_list[3] / np.sum(return_list[3])
    return return_list


def remap_classes(input_arr: np.ndarray, experiment: int) -> np.ndarray:
    if input_arr.ndim == 1:
        input_arr = input_arr[np.newaxis, :]
    output_class_num = len(CLASS_INFO[experiment][0])
    output_arr = np.zeros((input_arr.shape[0], output_class_num), 'i')
    for i, key in enumerate(sorted(CLASS_INFO[experiment][0].keys())):
        output_arr[:, i] = np.sum(input_arr[:, CLASS_INFO[experiment][0][key]], axis=1)
    return output_arr


def get_class_figure(input_arr: np.ndarray, row_labels: list, col_labels: list, bar_label: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(.7*len(col_labels), .7*len(row_labels)))
    im, cbar = heatmap(input_arr, row_labels, col_labels,
                       ax=ax, cbar_kw={}, cmap="YlGn", cbarlabel=bar_label)
    annotate_heatmap(im, valfmt='{x:.2f}', threshold=.6)
    fig.tight_layout()
    return fig


def analyse_class_distribution_per_video(data_path: pathlib.Path = None):
    data_path = PATH if data_path is None else data_path
    df = pd.read_pickle(data_path / 'label_table.pkl', compression='gzip')
    videos = df.folder_name.unique()
    video_classes = [[], [], [], []]
    video_classes_normalised = [[], [], [], []]
    for video in videos:
        video_frames = df.loc[df['folder_name'] == video].iloc[:, 3:].to_numpy()
        video_class_sums = np.sum(video_frames, 0)
        video_classes[0].append([video, video_frames.shape[0], *video_class_sums])
        video_classes_normalised[0].append([video, video_frames.shape[0],
                                            *(video_class_sums / np.sum(video_class_sums))])

        for j in [1, 2, 3]:  # Experiment number
            video_class_sums_tmp = remap_classes(video_class_sums, experiment=j)[0]
            video_classes[j].append([video, video_frames.shape[0], *video_class_sums_tmp])
            video_classes_normalised[j].append([video, video_frames.shape[0],
                                                *(video_class_sums_tmp / np.sum(video_class_sums_tmp))])
    sheet_names = ['original_classes', 'experiment_1', 'experiment_2', 'experiment_3']

    # Write to Excel .xlsx file (same file, several sheets) and .pkl files
    with pd.ExcelWriter(data_path / 'classes_per_video.xlsx') as writer:
        for j in range(4):
            df = pd.DataFrame(video_classes[j], columns=['video_name', 'num_frames'] + CLASS_NAMES[j])
            df.to_pickle(data_path / 'classes_per_video_{}.pkl'.format(j), compression='gzip')
            df.to_excel(writer, sheet_name=sheet_names[j])
    with pd.ExcelWriter(data_path / 'classes_per_video_normalised.xlsx') as writer:
        for j in range(4):
            df = pd.DataFrame(video_classes_normalised[j], columns=['video_name', 'num_frames'] + CLASS_NAMES[j])
            df.to_pickle(data_path / 'classes_per_video_normalised_{}.pkl'.format(j), compression='gzip')
            df.to_excel(writer, sheet_name=sheet_names[j])

    # Save as .png images
    for j in range(4):
        fig = get_class_figure(np.array(video_classes_normalised[j])[:, 3:].astype('f'),
                               videos, CLASS_NAMES[j], 'Probability')
        fig.savefig(PATH / 'exp{}.png'.format(j))


def get_permutation_candidate() -> list:
    # For exp 2, 'ignore' class (17 0-indexed) is relevant, next is 16.
    # For exp 3, it's classes (decreasing relevance) 25 ('ignore'), 24, 22, (21, 18, 20)
    # video_nums = {
    #     2: {
    #         17: [7, 9, 11, 13, 18, 20, 23, 24],
    #         16: [0, 4, 7, 9, 10, 11, 13, 15, 18, 20, 23, 24]
    #     },
    #     3: {
    #         25: [0, 7, 9, 11, 13, 18, 20, 21, 23, 24],
    #         24: [0, 11, 13, 15],
    #         22: [0, 1, 2, 4, 10, 11, 12, 20, 21, 22, 24],
    #         21: [0, 1, 2, 6, 9, 12, 14, 16, 18, 20],
    #         18: [0, 1, 2, 4, 6, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 23],
    #         20: [0, 1, 2, 3, 4, 6, 14, 15, 16, 17, 20, 21, 23]
    #     }
    # }
    video_nums_strict = {  # Thresholded at >1e-4
        0: {
            0: list(range(25))  # To fill in all other videos
        },
        2: {
            17: [7, 9, 13, 18, 23, 24],
            16: [4, 7, 9, 10, 11, 13, 15, 18, 20, 23, 24]
        },
        3: {
            25: [0, 7, 9, 11, 13, 18, 23, 24],
            24: [0, 11, 15],
            22: [0, 1, 2, 4, 11, 20, 24],
            21: [0, 1, 2, 6, 9, 12, 14, 16, 18, 20],
            18: [0, 1, 2, 6, 11, 12, 13, 14, 15, 17, 20, 21, 23],
            20: [0, 1, 3, 4, 15, 17, 20, 21, 23]
        }
    }
    # Goal: allocate an even amount of these classes into 5 groups.
    # Challenge: combining all the constraints
    priority_exp_key_list = [
        [3, 25],
        [2, 17],
        [3, 24],
        [2, 16],
        # [3, 22],
        # [3, 21],
        # [3, 18],
        # [3, 20],
        [0, 0]  # All videos
    ]
    distribution = [[], [], [], [], []]
    np.random.shuffle(priority_exp_key_list)
    for exp, key in priority_exp_key_list:
        vid_list = np.array(video_nums_strict[exp][key])
        vids_to_allocate = np.setdiff1d(vid_list, [item for sublist in distribution for item in sublist])
        np.random.shuffle(vids_to_allocate)
        for vid_num in vids_to_allocate:
            fill_num = np.zeros(5, 'i')
            for i in range(5):
                fill_num[i] = len(set(distribution[i]) & set(vid_list))
            i_to_fill = np.argmin(fill_num)
            distribution[i_to_fill].extend([vid_num])
    permutation = []
    for d in distribution:
        permutation.extend(d)
    assert np.unique(permutation).size == 25, "Permutation not valid"
    return permutation


def split_permutator(data_path: pathlib.Path = None):
    data_path = PATH if data_path is None else data_path
    df = pd.read_pickle(data_path / 'label_table.pkl', compression='gzip')
    valid_permutations = []
    failed_classes = [[], [], []]
    i = 0
    thresholds = [.75, .95, 1.9, .35]
    num_tries = 1e6
    print("{} permutations, thresholds: {}".format(num_tries, thresholds))
    while i < num_tries:
        perm = get_permutation_candidate()
        split_percentages, closeness, passing, failed_classes =\
            evaluate_permutation(df, perm, thresholds, failed_classes)
        if passing:
            valid_permutations.append([perm, split_percentages, closeness])
            print("\r> Valid permutation: {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n"
                  "             Splits: {}\n"
                  "      Avg closeness: {} / {} / {}"
                  .format(*perm, split_percentages, *[np.mean(c) for c in closeness[1:]]))
        else:
            print("\rTesting permutation {}: {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}"
                  .format(i, *perm), end='', flush=True)
            # print("\r      Avg closeness: {:.3f} (max {:.3f}) / {:.3f} (max {:.3f}) / {:.3f} (max {:.3f}"
            #       .format(np.mean(closeness[1]), np.max(closeness[1]),
            #               np.mean(closeness[2]), np.max(closeness[2]),
            #               np.mean(closeness[3]), np.max(closeness[3])), end='', flush=True)
        i += 1
    for i in range(3):
        print(failed_classes[i])
        plt.hist(failed_classes[i], range=(0, len(CLASS_NAMES[i + 1])),
                 bins=(np.arange(len(CLASS_NAMES[i + 1]) + 1) - .5).tolist())
        plt.xticks(np.arange(len(CLASS_NAMES[i + 1])))
        plt.savefig(PATH / 'failed_classes_exp{}_thresh_{}_{}.png'.format(i + 1, *thresholds[2:]))
        plt.clf()


def evaluate_permutation(df: pd.DataFrame, permutation: list,
                         thresholds: list = None, failed_classes: list = None):
    thresholds = [.75, .9, .1, .05] if thresholds is None else thresholds
    split_percentages = np.zeros(5, 'f')
    closeness = [
        np.zeros((5, len(CLASS_NAMES[0])), 'f'),
        np.zeros((5, len(CLASS_NAMES[1])), 'f'),
        np.zeros((5, len(CLASS_NAMES[2])), 'f'),
        np.zeros((5, len(CLASS_NAMES[3])), 'f')
    ]

    # Based on observations - classes where the conditions are impossible to satisfy
    impossible_classes = [
        [],  # Placeholder
        [],  # For exp 1
        [17],  # For exp 2
        [24, 25],  # For exp 3
    ]

    passing = True
    for i in range(5):
        valid_list = permutation[i * 5:(i + 1) * 5]
        train_list = list(set(permutation) - set(valid_list))
        for exp in range(4):
            split_list = get_train_valid_classes_from_split(df, train_list, valid_list, exp, 1)
            split_percentages[i] = split_list[0] / (split_list[0] + split_list[2])
            divisor = np.array(split_list[1])
            divisor[divisor == 0] = 1e-5  # To avoid division by 0 errors
            c = np.abs(np.array(split_list[1]) - np.array(split_list[3])) / divisor
            poss = list(set(range(len(CLASS_NAMES[exp]))) - set(impossible_classes[exp]))  # Classes to be tested
            if exp > 0:
                if thresholds[0] <= split_percentages[i] <= thresholds[1]\
                        and np.all(c[poss] < thresholds[2]) and np.mean(c[poss]) < thresholds[3]:
                    passing = passing and True
                else:
                    passing = False
                if failed_classes is not None:
                    if not thresholds[0] <= split_percentages[i] <= thresholds[1]\
                            or not np.all(c < thresholds[2]) or not np.mean(c) < thresholds[3]:
                        failed_classes[exp - 1].extend(np.where(c > thresholds[2])[0].tolist())
            closeness[exp][i] = c
    return split_percentages, closeness, passing, failed_classes


def get_split_and_closeness_from_order(order: list):
    df = pd.read_pickle(PATH / 'label_table.pkl', compression='gzip')
    split_percentages = np.zeros(5, 'f')
    closeness = [
        np.zeros((5, len(CLASS_NAMES[0])), 'f'),
        np.zeros((5, len(CLASS_NAMES[1])), 'f'),
        np.zeros((5, len(CLASS_NAMES[2])), 'f'),
        np.zeros((5, len(CLASS_NAMES[3])), 'f')
    ]
    distribution_train = [
        np.zeros((5, len(CLASS_NAMES[0])), 'f'),
        np.zeros((5, len(CLASS_NAMES[1])), 'f'),
        np.zeros((5, len(CLASS_NAMES[2])), 'f'),
        np.zeros((5, len(CLASS_NAMES[3])), 'f')
    ]
    distribution_valid = [
        np.zeros((5, len(CLASS_NAMES[0])), 'f'),
        np.zeros((5, len(CLASS_NAMES[1])), 'f'),
        np.zeros((5, len(CLASS_NAMES[2])), 'f'),
        np.zeros((5, len(CLASS_NAMES[3])), 'f')
    ]

    for i in range(5):
        valid_list = order[i * 5:(i + 1) * 5]
        train_list = list(set(order) - set(valid_list))
        for exp in range(4):
            split_list = get_train_valid_classes_from_split(df, train_list, valid_list, exp, 1)
            split_percentages[i] = split_list[0] / (split_list[0] + split_list[2])
            divisor = np.array(split_list[1])
            divisor[divisor == 0] = 1e-5  # To avoid division by 0 errors
            closeness[exp][i] = np.abs(np.array(split_list[1]) - np.array(split_list[3])) / divisor
            distribution_train[exp][i] = split_list[1]
            distribution_valid[exp][i] = split_list[3]
    df_list = [[], [], [], []]
    for exp in range(4):
        df0 = pd.DataFrame(distribution_train[exp], columns=CLASS_NAMES[exp])
        df0.insert(0, 'type', 'training')
        df0.insert(1, 'split', list(range(5)))
        df1 = pd.DataFrame(distribution_valid[exp], columns=CLASS_NAMES[exp])
        df1.insert(0, 'type', 'validation')
        df1.insert(1, 'split', list(range(5)))
        df2 = pd.DataFrame(closeness[exp], columns=CLASS_NAMES[exp])
        df2.insert(0, 'type', 'difference')
        df2.insert(1, 'split', list(range(5)))
        df_list[exp] = pd.concat([df0, df1, df2]).reset_index(drop=True)
    return split_percentages, df_list


def data_checker(experiment: int):
    df = pd.read_pickle('data/data.pkl')
    data_path = "C:\\Users\\RVIM_Claudio\\Documents\\cadis\\data\\segmentation"
    for item in range(len(df.index)):
        img = cv2.imread(str(pathlib.Path(data_path) / df.iloc[item].loc['img_path']))
        lbl = cv2.imread(str(pathlib.Path(data_path) / df.iloc[item].loc['lbl_path']), 0)
        remapped_mask, classes_exp, colormap = remap_experiment(lbl, experiment)
        lbl_img = mask_to_colormap(remapped_mask, colormap=colormap)[..., ::-1]
        g = []
        for i in range(3):
            g.append(np.linalg.norm(np.gradient(lbl_img[..., i]), axis=0))
        grad = np.sum(g, axis=0)
        res_img = np.round(lbl_img * .25 + img * .75)
        res_img[grad > 0] = 0
        file_name = pathlib.PurePath(df.iloc[item].loc['img_path']).parts[-1]
        if not pathlib.Path.is_dir(pathlib.Path(data_path) / 'comb_images'):
            pathlib.Path.mkdir(pathlib.Path(data_path) / 'comb_images')
        cv2.imwrite(str(pathlib.Path(data_path) / 'comb_images' / file_name), res_img)
        print("Img {}".format(file_name))
