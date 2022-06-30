# 切分验证集-0.2和训练集-0.8
import os
import random
import shutil

from glob import glob

data_path = r'D:\计算机视觉\garbage_classify_v2\garbage_classify_v2\train_data_v2'


# img_path_list包括了图片路径和对应标签
# 得到图片的信息
def get_image_txt_info(path):
    data_path_txt = glob(os.path.join(path, '*.txt'))  # all txt files

    img_list = []
    img2label_dic = {}
    # 记录每个类别的数目
    label_count_dic = {}
    for txt_path in data_path_txt:
        with open(txt_path, 'r') as f:
            line = f.readline()  # read a line

        line = line.strip()  # delete pre ' ' and  last '
        img_name = line.split(',')[0]  # img_2778.jpg
        img_label = int(line.split(',')[1])  # 7
        img_name_path = os.path.join(data_path, img_name)  # image

        img_list.append({'img_name_path': img_name_path, 'img_label': img_label})
        # image_name:img_label
        img2label_dic[img_name] = img_label

        img_label_count = label_count_dic.get(img_label, 0)  # 不存在则初始化为0
        if img_label_count:
            label_count_dic[img_label] += 1
        else:
            label_count_dic[img_label] = 1
    return img_list, img2label_dic, label_count_dic


# 将数据集整理成对应的形式
def main():
    img_list, img2label_dic, label_count_dic = get_image_txt_info(data_path)

    # 格式： label : [img_path]
    imglabel_imgpath_dict = {index: [] for index in range(40)}
    for img_data in img_list:
        imglabel_imgpath_dict[img_data['img_label']].append(img_data['img_name_path'])

    train_list = []
    val_list = []
    # 使得每个类别的样本分布均匀
    for label, label_num in label_count_dic.items():
        train_size = int(label_num * 0.8)
        # 对样本进行打乱
        temp_data = imglabel_imgpath_dict[label]
        random.shuffle(temp_data)
        train_list.extend(
            [{'image_name_path': img_path, 'image_label': label} for img_path in temp_data[:train_size]])
        val_list.extend(
            [{'image_name_path': img_path, 'image_label': label} for img_path in temp_data[train_size:]])

    # 生成train.txt文件  val.txt文件
    path = '../GarbageData/'
    train_val = {'train': train_list, 'val': val_list}
    for key in train_val:
        with open(os.path.join(path, key + '.txt'), 'w') as f:
            for img_dict in train_val[key]:
                img_name_path = img_dict['image_name_path']
                img_label = img_dict['image_label']
                f.write('{}\t{}\n'.format(img_name_path, img_label))

                # 生成train or val
                sub_path = os.path.join(path, key, str(img_label))
                if not os.path.exists(sub_path):
                    os.makedirs(sub_path)
                # 图片数据copy
                shutil.copy(img_name_path, sub_path)


if __name__ == '__main__':
    main()
