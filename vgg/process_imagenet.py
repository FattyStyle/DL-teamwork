import os
import shutil


def make_test_data(num_per_type, output_root, output_label_root, move=False):
    if not os.path.isdir(output_root):
        os.mkdir(output_root)
    train_root, dirs, _ = next(os.walk(r'/home/disk/zyr/dataset/ILSVRC2012_img_train'))
    img_num = 0
    f = open(output_label_root, 'w')
    for img_type in dirs:
        type_root, _, images = next(os.walk(os.path.join(train_root, img_type)))
        for image in images:
            f.write(str(img_num//num_per_type) + '\n')
            img_num += 1
            img_name = '%08d' % img_num + '.JPEG'
            if move:
                # 会从train中删掉
                shutil.move(os.path.join(train_root, img_type, image), os.path.join(output_root, img_name))
            else:
                shutil.copyfile(os.path.join(train_root, img_type, image), os.path.join(output_root, img_name))
            if img_num % num_per_type == 0:
                break
    f.close()


if __name__ == '__main__':
    # make test dataset from train set
    make_test_data(10, '/home/disk/zyr/dataset/test', '/home/disk/zyr/dataset/self_defined_test_label.txt', move=True)
