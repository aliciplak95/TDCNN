import cv2
import argparse
import os, time, datetime
# import PIL.Image as Image
import numpy as np
from keras.models import load_model, model_from_json
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave
from skimage import color
from skimage import io


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='data/Test', type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['Set12'], type=list, help='Modelde denenecek görüntü')
    parser.add_argument('--sigma', default=25, type=int, help='noise level')
    parser.add_argument('--model_dir', default=os.path.join('models', 'TDCNN'), type=str, help='Model klasör yolu')
    parser.add_argument('--model_name', default='model_190.hdf5', type=str, help='Çalıştıralcak model adı.')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of results')
    parser.add_argument('--save_result', default=1, type=int, help='Temizlenen görüntüyü kaydet')
    return parser.parse_args()


def to_tensor(img):
    if img.ndim == 2:
        return img[np.newaxis, ..., np.newaxis]
    elif img.ndim == 3:
        return np.moveaxis(img, 2, 0)[..., np.newaxis]


def from_tensor(img):
    return np.squeeze(np.moveaxis(img[..., 0], 0, -1))


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def save_result(result, path):
    path = path if path.find('.') != -1 else path + '.jpg'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1))


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def noise(x_, x):

    import matplotlib.pyplot as plt

    noise = x - x_

    print(x_.dtype, x_.shape)
    print(x.dtype, x.shape)
    print(noise.dtype, noise.shape)

    plt.imsave('B:/TDCNN/results/noise.jpg', noise)

    show(noise)


if __name__ == '__main__':

    args = parse_args()


    if not os.path.exists(os.path.join(args.model_dir, args.model_name)):
        # load json and create model
        json_file = open(os.path.join(args.model_dir, 'model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(os.path.join(args.model_dir, 'model_190.hdf5'))
        log('load trained model')
    else:
        model = load_model(os.path.join(args.model_dir, args.model_name), compile=False)
        log('load trained model')

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    for set_cur in args.set_names:

        if not os.path.exists(os.path.join(args.result_dir, set_cur)):
            os.mkdir(os.path.join(args.result_dir, set_cur))
        psnrs = []
        ssims = []

        for im in os.listdir(os.path.join(args.set_dir, set_cur)):
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                # x = np.array(Image.open(os.path.join(args.set_dir,set_cur,im)), dtype='float32') / 255.0
                x = np.array(imread(os.path.join(args.set_dir, set_cur, im)), dtype=np.float32) / 255.0
                show(x)
                np.random.seed(seed=0)  # for reproducibility
                y = x  # Add Gaussian noise without clipping
                y = y.astype(np.float32)
                y_ = to_tensor(y)
                start_time = time.time()
                x_ = model.predict(y_)  # inference
                elapsed_time = time.time() - start_time
                print('%10s : %10s : %2.4f second' % (set_cur, im, elapsed_time))
                x_ = from_tensor(x_)

                noise(x_, x)
                # z=x_
                # c=x-z
                # show(c)
                psnr_x_ = compare_psnr(x, x_)
                ssim_x_ = compare_ssim(x, x_)
                if args.save_result:
                    name, ext = os.path.splitext(im)
                    show(np.hstack((y, x_)))  # show the image
                    save_result(x_, path=os.path.join(args.result_dir, set_cur,
                                                      name + '_dncnn' + ext))  # save the denoised image
                psnrs.append(psnr_x_)
                ssims.append(ssim_x_)

        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)
        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)

        if args.save_result:
            save_result(np.hstack((psnrs, ssims)), path=os.path.join(args.result_dir, set_cur, 'results.txt'))

        log('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))




