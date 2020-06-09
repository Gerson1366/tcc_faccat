import struct
import numpy as np
from PIL import Image

sz_record = 8199


def read_record_ETL9G(f):
    s = f.read(8199)
    r = struct.unpack('>2H8sI4B4H2B34x8128s7x', s)
    iF = Image.frombytes('F', (128, 127), r[14], 'bit', 4)
    iL = iF.convert('P')
    return r + (iL,)


def read_kanji():
    # Character type = 72, person = 160, y = 127, x = 128
    ary = np.zeros([1000, 200, 127, 128], dtype=np.uint8)
    ary2 = np.zeros([1000, 200, 127, 128], dtype=np.uint8)
    ary3 = np.zeros([1000, 200, 127, 128], dtype=np.uint8)
    ary4 = np.zeros([36, 200, 127, 128], dtype=np.uint8)
    for j in range(1, 51):
        print('Datasets/ETL9G/ETL9G_{:02d}'.format(j))
        filename = 'Datasets/ETL9G/ETL9G_{:02d}'.format(j)
        with open(filename, 'rb') as f:
            for id_dataset in range(4):
                moji = 0
                moji2 = 0
                moji3 = 0
                moji4 = 0
                for i in range(3036):
                    r = read_record_ETL9G(f)
                    #print(str(j)+":"+str(id_dataset)+":"+str(i))
                    #if b'.HIRA' in r[2]:
                    #if(r[1]==12326):
                     #   print("Code: ",r[1])
                      #  print("Character: ",r[2])
                       # print("Class: ",i)
                    if(i==869):
                        print("Character: ",r[2])
                        print("Code: ",r[1])
                    if(i<1000):
                        ary[moji, (j - 1) * 4 + id_dataset] = np.array(r[-1])
                        moji += 1
                    if(i>=1000 and i<2000):
                        ary2[moji2, (j - 1) * 4 + id_dataset] = np.array(r[-1])
                        moji2 += 1
                    if(i>=2000 and i<3000):
                        ary3[moji3, (j - 1) * 4 + id_dataset] = np.array(r[-1])
                        moji3 += 1
                    if(i>=3000):
                        ary4[moji4, (j - 1) * 4 + id_dataset] = np.array(r[-1])
                        moji4 += 1
    np.savez_compressed("kanji_01.npz", ary)
    np.savez_compressed("kanji_02.npz", ary2)
    np.savez_compressed("kanji_03.npz", ary3)
    np.savez_compressed("kanji_04.npz", ary4)

read_kanji()