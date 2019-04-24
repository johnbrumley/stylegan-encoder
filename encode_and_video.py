import os
import pickle
import operator
import numpy as np

import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
import PIL.Image

# align img
import sys
import bz2
from keras.utils import get_file
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector

# encode img
import argparse
from tqdm import tqdm
from encoder.perceptual_model import PerceptualModel

# dnn / generator

# video write
from random import randint
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc

#align
LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
RAW_IMAGES_DIR = "raw_images/" # sys.argv[1]
ALIGNED_IMAGES_DIR = "aligned_images/" # sys.argv[2]


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                           LANDMARKS_MODEL_URL, cache_subdir='temp'))

landmarks_detector = LandmarksDetector(landmarks_model_path)

# encode

# !python encode_images.py aligned_images generated_images latent_representations --iterations=150 --lr=0.05 --wd=0.005

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'

def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

src_dir = "aligned_images"
generated_images_dir = "generated_images"
dlatent_dir = "latent_representations"
batch_size = 1 # maybe try larger batch if this helps efficiency?
image_size = 256
lr = 0.05
iterations = 150
wd = 0.005
randomize_noise = False # also test with true to see if any better

os.makedirs(generated_images_dir, exist_ok=True)
os.makedirs(dlatent_dir, exist_ok=True)

# Initialize generator and perceptual model
tflib.init_tf()
with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f)

generator = Generator(Gs_network, batch_size, randomize_noise=randomize_noise)
perceptual_model = PerceptualModel(image_size, layer=9, batch_size=batch_size)
perceptual_model.build_perceptual_model(generator.generated_image)

# generate
  
# URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'

# tflib.init_tf()
# with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
#     generator_network, discriminator_network, Gs_network = pickle.load(f)

# generator = Generator(Gs_network, batch_size=1, randomize_noise=True)
  
def generate_image_array(latent_vector):
    latent_vector = latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    return img_array

def get_animation_frames(pA, pB, count=24):
    frames = []
    for c in np.linspace(0, 1, count):
        frames.append(generate_image_array(c*pA + (1-c)*pB))
    return frames

# Video
def resize(arr, res, ratio=1.):
    shape = (int(res*ratio),res)
    return np.array(PIL.Image.fromarray(arr).resize(shape, resample=PIL.Image.ANTIALIAS))

def crop(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

def hold_pattern(imgs,holding_length, direction=1):
    pat = [randint(1, 20) for p in range(0, holding_length)] if direction == 1 else [len(imgs) - randint(1, 20) for p in range(0, holding_length)]
    return [imgs[i] for i in pat]

def make_loop(imgs, fps=24, gap=5):
    return hold_pattern(imgs,gap*fps) + imgs + hold_pattern(imgs,2*gap*fps,-1) + imgs[::-1] + hold_pattern(imgs,gap*fps)

def make_video(name, imgs, fps=24, res=1024, lib='cv', width=1024, height=1024):
    imgs = make_loop(imgs)
    write_cv(imgs, name + '.mp4', fps)
    return
  
def write_cv(imgs, name, fps, width=768, height=1366):
    fourcc = VideoWriter_fourcc(*'PIM1') # other codecs MJPG (yes) MP4V (no) H264 (no) X264 (no) PIM1 (yes)
    video = VideoWriter(name, fourcc, float(fps), (width, height), True)
    [video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) for img in imgs]
    video.release()    

# pre-align since these images should already be aligned

# align

"""
Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
python align_images.py /raw_images /aligned_images
"""

RAW_IMAGES_DIR = "raw_images/" # /gdrive/My Drive/stylegan_data/raw_images/ raw_images/

for img_name in os.listdir(RAW_IMAGES_DIR):
    raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
    for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
        face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
        aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)

        image_align(raw_img_path, aligned_face_path, face_landmarks)

# run through pipes

# encode

os.makedirs("latents", exist_ok=True)

ref_images = [os.path.join(src_dir, x) for x in os.listdir(src_dir)]
ref_images = list(filter(os.path.isfile, ref_images))

if len(ref_images) == 0:
    raise Exception('%s is empty' % src_dir)

# Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
for images_batch in tqdm(split_to_batches(ref_images, batch_size), total=len(ref_images)//batch_size):
    names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]

    perceptual_model.set_reference_images(images_batch)
    op = perceptual_model.optimize(generator.dlatent_variable, iterations=iterations, learning_rate=lr, weight_decay=wd)
    pbar = tqdm(op, leave=False, total=iterations)
    for loss in pbar:
        pbar.set_description(' '.join(names)+' Loss: %.2f' % loss)
    print(' '.join(names), ' loss:', loss)

    # Generate images from found dlatents and save them
    generated_images = generator.generate_images()
    generated_dlatents = generator.get_dlatents()
    for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
        img = PIL.Image.fromarray(img_array, 'RGB')
        img.save(os.path.join(generated_images_dir, f'{img_name}.png'), 'PNG')
        np.save(os.path.join("latents", f'{img_name}.npy'), dlatent)

    generator.reset_dlatents()

# load latents (instead should hold in mem)

real = np.load('latent_representations/john.npy') # existing latent
# fake = np.load('latent_representations/target1_01.npy')
# chair = np.load('latent_representations/chair.npy')

fakes = [os.path.join("latents", x) for x in os.listdir("latents")]
fakes = list(filter(os.path.isfile, fakes))
fakes = [np.load(f) for f in fakes]


height = 1366
width = 768
fps = 24
seconds = 10

frames = fps*seconds

for i, f in enumerate(fakes, start=1):
  print("video",i)
  print("  generating",frames,"animation frames....")
  anim_frames = get_animation_frames(f, real, frames)

  # resize and crop
  print("  resize crop frames....")
  anim_frames_sized = [crop(resize(frame, height), (height,width)) for frame in anim_frames]

  print("  making video....")
  make_video("vid_"+str(i), anim_frames_sized, fps, lib='cv')
  
print("done.")