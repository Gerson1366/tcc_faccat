import numpy as np
import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt
import h5py

ary = np.load("/content/drive/My Drive/Colab Notebooks/kanji_01.npz",mmap_mode="r")['arr_0'].reshape([-1, 127, 128]).astype(np.float32) / 15
np.save("/content/drive/My Drive/Colab Notebooks/ary01",ary)