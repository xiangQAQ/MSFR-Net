from batchgenerators.dataloading import SlimDataLoaderBase
import numpy as np



class DataLoader3D(SlimDataLoaderBase):
    def __init__(self, datadirlist, patch_size, batch_size, oversample_foreground_percent=0.0):
        super(DataLoader3D, self).__init__(datadirlist, batch_size, None)
        self.oversample_foreground_percent = oversample_foreground_percent
        self.patch_size = patch_size
        self.list_of_crop_dir = datadirlist
        self.batch_size = batch_size

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_crop_dir, self.batch_size, False, None)
        data = []
        seg = []
        for j, i in enumerate(selected_keys):
            case_all_data = np.load(i)['data']
            shape_differ = case_all_data[0].shape - self.patch_size
            if self.get_do_oversample(j):
                force_fg = True
            else:
                force_fg = False

            if not force_fg:
                bbox_x_lb = np.random.randint(0, shape_differ[0])
                bbox_y_lb = np.random.randint(0, shape_differ[1])
                bbox_z_lb = np.random.randint(0, shape_differ[2])
            else:
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                foreground_classes = np.unique(case_all_data[-1]).astype(np.int)
                foreground_classes = foreground_classes[foreground_classes > 0]
                if len(foreground_classes) == 0:
                    selected_class = 0
                else:
                    selected_class = np.random.choice(foreground_classes)

                selected_class_flag = np.argwhere(case_all_data[-1] == selected_class)

                if len(selected_class_flag) != 0:
                    selected_flag = selected_class_flag[np.random.choice(len(selected_class_flag))]
                    if (selected_flag[0] - self.patch_size[0] / 2) < 0:
                        bbox_x_lb = 0
                    elif (selected_flag[0] - self.patch_size[0] / 2) > shape_differ[0]:
                        bbox_x_lb = shape_differ[0]
                    else:
                        bbox_x_lb = (selected_flag[0] - self.patch_size[0] / 2).astype(np.int)

                    if (selected_flag[1] - self.patch_size[1] / 2) < 0:
                        bbox_y_lb = 0
                    elif (selected_flag[1] - self.patch_size[1] / 2) > shape_differ[1]:
                        bbox_y_lb = shape_differ[1]
                    else:
                        bbox_y_lb = (selected_flag[1] - self.patch_size[1] / 2).astype(np.int)

                    if (selected_flag[2] - self.patch_size[2] / 2) < 0:
                        bbox_z_lb = 0
                    elif (selected_flag[2] - self.patch_size[2] / 2) > shape_differ[2]:
                        bbox_z_lb = shape_differ[2]
                    else:
                        bbox_z_lb = (selected_flag[2] - self.patch_size[2] / 2).astype(np.int)
                else:

                    bbox_x_lb = np.random.randint(0, shape_differ[0])
                    bbox_y_lb = np.random.randint(0, shape_differ[1])
                    bbox_z_lb = np.random.randint(0, shape_differ[2])
            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]
            bbox_z_ub = bbox_z_lb + self.patch_size[2]

            case_all_data = case_all_data[:, bbox_x_lb:bbox_x_ub,
                            bbox_y_lb:bbox_y_ub,
                            bbox_z_lb:bbox_z_ub]

            data.append(case_all_data[:-1][None])
            seg.append(case_all_data[-1:][None])
        data = np.vstack(data)
        seg = np.vstack(seg)
        seg[np.where(seg==4)]=3
        seg_all = seg.copy()
        seg_t1c = seg.copy()
        seg_flair = seg.copy()
        seg_t1c[np.where(seg_t1c == 2)] = 0
        seg_flair[np.where(seg_flair == 1)] = 3
        return {'data': data, 'seg_all': seg_all, 'seg_t1c': seg_t1c, 'seg_flair': seg_flair}