from Crypto.Cipher import AES
import os
import pandas as pd
import numpy as np
import binascii
from base64 import b64encode, b64decode
from Crypto import Random
import pdb
from Crypto.Util import Padding
from keras.models import Sequential
from keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt

data1 = pd.read_csv('Train.csv')
data2 = pd.read_csv('Test.csv')
data3 = pd.read_csv('Valid.csv')

df = pd.DataFrame()
df['Plaintext'] = data1['text']
df2 = pd.DataFrame()
df2['Plaintext'] = data2['text']
df3 = pd.DataFrame()
df3['Plaintext'] = data3['text']
df = df.append(df2, ignore_index=True)
df = df.append(df3, ignore_index=True)
df['Not padded Plaintext'] = ''
df['Key16'] = ''
df['Ciphertext16'] = ''
df['Key24'] = ''
df['Ciphertext24'] = ''
# df['Key32'] = ''
# df['Ciphertext32'] = ''

# creating random keys 16,24,32
for i in df.index:
    random_key = os.urandom(16)
    df.iat[i, 2] = random_key

for i in df.index:
    random_key = os.urandom(24)
    df.iat[i, 4] = random_key

# for i in df.index:
#     random_key = os.urandom(32)
#     df.iat[i, 6] = random_key

print('Generating keys done!')


def encryption(plain_text, key):
    # text = Padding.pad(plain_text, AES.block_size, style = 'pkcs7')
    cipher = AES.new(key, mode=1)
    encrypted_text = cipher.encrypt(plain_text)
    return encrypted_text


# removing chars from texts with len !%16==0
for i in df.index:
    text = str.encode(df.iloc[i]['Plaintext'])
    if len(text) % 16 != 0:
        (df.at[i, 'Not padded Plaintext']) = text[:len(text) - (len(text) % 16)]
    else:
        (df.at[i, 'Not padded Plaintext']) = text
print(df['Not padded Plaintext'])
print("removing chars from texts with len !%16==0 done")

# encrypting
for i in df.index:
    text = df.iloc[i]['Not padded Plaintext']
    key = df.iloc[i]['Key16']
    e = encryption(text, key)
    df.at[i, 'Ciphertext16'] = e

for i in df.index:
    text = df.iloc[i]['Not padded Plaintext']
    key = df.iloc[i]['Key24']
    e = encryption(text, key)
    df.at[i, 'Ciphertext24'] = e

# for i in df.index:
#     text = df.iloc[i]['Not padded Plaintext']
#     key = df.iloc[i]['Key32']
#     e = encryption(text, key)
#     df.at[i, 'Ciphertext32'] = e

print(df['Ciphertext16'])
print(df['Ciphertext24'])
# print(df['Ciphertext32'])
print("Encrypting done")

# byte to bit of ciphertext16
df['BitCiphertext16'] = ''
for i in df.index:
    df.at[i, 'BitCiphertext16'] = ''.join(format(j, 'b') for j in df.iloc[i]['Ciphertext16'])

print(df['BitCiphertext16'])
print("byte to bit of ciphertext16 done")

# byte to bit of ciphertext24
df['BitCiphertext24'] = ''
for i in df.index:
    df.at[i, 'BitCiphertext24'] = ''.join(format(j, 'b') for j in df.iloc[i]['Ciphertext24'])

print(df['BitCiphertext24'])
print("byte to bit of ciphertext24 done")

# # byte to bit of ciphertext32
# df['BitCiphertext32'] = ''
# for i in df.index:
#     df.at[i, 'BitCiphertext32'] = ''.join(format(j, 'b') for j in df.iloc[i]['Ciphertext32'])
#
# print(df['BitCiphertext32'])
# print("byte to bit of ciphertext32 done")

# byte to bit plaintext
df['BitPlaintext'] = ''
for i in df.index:
    df.at[i, 'BitPlaintext'] = ''.join(format(j, 'b') for j in df.iloc[i]['Not padded Plaintext'])

print(df['BitPlaintext'])
print("byte to bit of plaintext done")

# len(bits) for training up to %128==0
df['TrainPlain'] = ''
df['Train16'] = ''
df['Train24'] = ''
# df['Train32'] = ''

for i in df.index:
    text = df.iloc[i]['BitPlaintext']
    if len(text) % 128 != 0:
        (df.at[i, 'TrainPlain']) = text[:len(text) - (len(text) % 128)]
    else:
        (df.at[i, 'TrainPlain']) = text

for i in df.index:
    cipher = df.iloc[i]['BitCiphertext16']
    if len(cipher) % 128 != 0:
        (df.at[i, 'Train16']) = cipher[:len(cipher) - (len(cipher) % 128)]
    else:
        (df.at[i, 'Train16']) = cipher

for i in df.index:
    cipher = df.iloc[i]['BitCiphertext24']
    if len(cipher) % 128 != 0:
        (df.at[i, 'Train24']) = cipher[:len(cipher) - (len(cipher) % 128)]
    else:
        (df.at[i, 'Train24']) = cipher

# for i in df.index:
#     cipher = df.iloc[i]['BitCiphertext32']
#     if len(cipher) % 128 != 0:
#         (df.at[i, 'Train32']) = cipher[:len(cipher) - (len(cipher) % 128)]
#     else:
#         (df.at[i, 'Train32']) = cipher
print("len(bits) for training up to %128==0 done")

# len(bits plain) == len(bits cipher)
for i in df.index:
    plain = df.iloc[i]['TrainPlain']
    cipher = df.iloc[i]['Train16']
    if (len(plain) > len(cipher)):
        df.at[i, 'TrainPlain'] = plain[:len(cipher)]
    else:
        df.at[i, 'Train16'] = plain[:len(plain)]

number_columns = 128
# a1
for i in range(0, 10000):
    sample_string = df.iloc[i]['TrainPlain']
    l = [list(sample_string[i:i + number_columns]) for i in range(0, len(sample_string), number_columns)]
    if (i == 0):
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                          np.dtype(int))
        a1 = matrix
    else:
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                           np.dtype(int))
        a1 = np.concatenate((a1, matrix))
    print('Done', i)

print(a1)

for i in range(10000, 20000):
    sample_string = df.iloc[i]['TrainPlain']
    l = [list(sample_string[i:i + number_columns]) for i in range(0, len(sample_string), number_columns)]
    if (i == 10000):
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                          np.dtype(int))
        a2 = matrix
    else:
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                           np.dtype(int))
        a2 = np.concatenate((a2, matrix))
    print('Done', i)

print(a2)

for i in range(20000, 30000):
    sample_string = df.iloc[i]['TrainPlain']
    l = [list(sample_string[i:i + number_columns]) for i in range(0, len(sample_string), number_columns)]
    if (i == 20000):
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                          np.dtype(int))
        a3 = matrix
    else:
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                           np.dtype(int))
        a3 = np.concatenate((a3, matrix))
    print('Done', i)

print(a3)

for i in range(30000, 40000):
    sample_string = df.iloc[i]['TrainPlain']
    l = [list(sample_string[i:i + number_columns]) for i in range(0, len(sample_string), number_columns)]
    if (i == 30000):
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                          np.dtype(int))
        a4 = matrix
    else:
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                           np.dtype(int))
        a4 = np.concatenate((a4, matrix))
    print('Done', i)

print(a4)

for i in range(40000, 50000):
    sample_string = df.iloc[i]['TrainPlain']
    l = [list(sample_string[i:i + number_columns]) for i in range(0, len(sample_string), number_columns)]
    if (i == 40000):
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                          np.dtype(int))
        a5 = matrix
    else:
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                           np.dtype(int))
        a5 = np.concatenate((a5, matrix))
    print('Done', i)

print(a5)

final_plain = np.concatenate((a1, a2))
final_plain = np.concatenate((final_plain, a3))
final_plain = np.concatenate((final_plain, a4))
final_plain = np.concatenate((final_plain, a5))

np.save("FinalPlain16.npy", final_plain)
print("Saving FinalPlain done!")

for i in range(0, 10000):
    sample_string = df.iloc[i]['Train16']
    l = [list(sample_string[i:i + number_columns]) for i in range(0, len(sample_string), number_columns)]
    if (i == 0):
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                          np.dtype(int))
        b1 = matrix
    else:
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                           np.dtype(int))
        b1 = np.concatenate((b1, matrix))
    print('Done', i)

print(b1)

for i in range(10000, 20000):
    sample_string = df.iloc[i]['Train16']
    l = [list(sample_string[i:i + number_columns]) for i in range(0, len(sample_string), number_columns)]
    if (i == 10000):
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                          np.dtype(int))
        b2 = matrix
    else:
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                           np.dtype(int))
        b2 = np.concatenate((b2, matrix))
    print('Done', i)

print(b2)

for i in range(20000, 30000):
    sample_string = df.iloc[i]['Train16']
    l = [list(sample_string[i:i + number_columns]) for i in range(0, len(sample_string), number_columns)]
    if (i == 20000):
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                          np.dtype(int))

        b3 = matrix
    else:
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                           np.dtype(int))
        b3 = np.concatenate((b3, matrix))
    print('Done', i)

print(b3)

for i in range(30000, 40000):
    sample_string = df.iloc[i]['Train16']
    l = [list(sample_string[i:i + number_columns]) for i in range(0, len(sample_string), number_columns)]
    if (i == 30000):
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                          np.dtype(int))
        b4 = matrix
    else:
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                           np.dtype(int))
        b4 = np.concatenate((b4, matrix))
    print('Done', i)

print(b4)

for i in range(40000, 50000):
    sample_string = df.iloc[i]['Train16']
    l = [list(sample_string[i:i + number_columns]) for i in range(0, len(sample_string), number_columns)]
    if (i == 40000):
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                          np.dtype(int))
        b5 =matrix
    else:
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                           np.dtype(int))
        b5 = np.concatenate((b5, matrix))
    print('Done', i)

print(b5)


final_cipher = np.concatenate((b1, b2))
final_cipher = np.concatenate((final_cipher, b3))
final_cipher = np.concatenate((final_cipher, b4))
final_cipher = np.concatenate((final_cipher, b5))

np.save("FinalTrain16.npy", final_cipher)


for i in df.index:
    plain = df.iloc[i]['TrainPlain']
    cipher = df.iloc[i]['Train24']
    if (len(plain) > len(cipher)):
        df.at[i, 'TrainPlain'] = plain[:len(cipher)]
    else:
        df.at[i, 'Train24'] = plain[:len(plain)]



# for i in df.index:
#     plain = df.iloc[i]['TrainPlain']
#     cipher = df.iloc[i]['Train32']
#     if (len(plain) > len(cipher)):
#         df.at[i, 'TrainPlain'] = plain[:len(cipher)]
#     else:
#         df.at[i, 'Train32'] = plain[:len(plain)]
# print("len(bits plain) == len(bits cipher) done")

for i in range(0, 10000):
    sample_string = df.iloc[i]['TrainPlain']
    l = [list(sample_string[i:i + number_columns]) for i in range(0, len(sample_string), number_columns)]
    if (i == 0):
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                          np.dtype(int))
        a1 = matrix
    else:
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                           np.dtype(int))
        a1 = np.concatenate((a1, matrix))
    print('Done', i)

print(a1)

for i in range(10000, 20000):
    sample_string = df.iloc[i]['TrainPlain']
    l = [list(sample_string[i:i + number_columns]) for i in range(0, len(sample_string), number_columns)]
    if (i == 10000):
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                          np.dtype(int))
        a2 = matrix
    else:
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                           np.dtype(int))
        a2 = np.concatenate((a2, matrix))
    print('Done', i)

print(a2)

for i in range(20000, 30000):
    sample_string = df.iloc[i]['TrainPlain']
    l = [list(sample_string[i:i + number_columns]) for i in range(0, len(sample_string), number_columns)]
    if (i == 20000):
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                          np.dtype(int))
        a3 = matrix
    else:
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                           np.dtype(int))
        a3 = np.concatenate((a3, matrix))
    print('Done', i)

print(a3)

for i in range(30000, 40000):
    sample_string = df.iloc[i]['TrainPlain']
    l = [list(sample_string[i:i + number_columns]) for i in range(0, len(sample_string), number_columns)]
    if (i == 30000):
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                          np.dtype(int))
        a4 = matrix
    else:
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                           np.dtype(int))
        a4 = np.concatenate((a4, matrix))
    print('Done', i)

print(a4)

for i in range(40000, 50000):
    sample_string = df.iloc[i]['TrainPlain']
    l = [list(sample_string[i:i + number_columns]) for i in range(0, len(sample_string), number_columns)]
    if (i == 40000):
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                          np.dtype(int))
        a5 = matrix
    else:
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                           np.dtype(int))
        a5 = np.concatenate((a5, matrix))
    print('Done', i)

print(a5)

final_plain = np.concatenate((a1, a2))
final_plain = np.concatenate((final_plain, a3))
final_plain = np.concatenate((final_plain, a4))
final_plain = np.concatenate((final_plain, a5))

np.save("FinalPlain24.npy", final_plain)
print("Saving FinalPlain done!")



for i in range(0, 10000):
    sample_string = df.iloc[i]['Train24']
    l = [list(sample_string[i:i + number_columns]) for i in range(0, len(sample_string), number_columns)]
    if (i == 0):
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                          np.dtype(int))
        b1 = matrix
    else:
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                           np.dtype(int))
        b1 = np.concatenate((b1, matrix))
    print('Done', i)

print(b1)

for i in range(10000, 20000):
    sample_string = df.iloc[i]['Train24']
    l = [list(sample_string[i:i + number_columns]) for i in range(0, len(sample_string), number_columns)]
    if (i == 10000):
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                          np.dtype(int))
        b2 = matrix
    else:
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                           np.dtype(int))
        b2 = np.concatenate((b2, matrix))
    print('Done', i)

print(b2)

for i in range(20000, 30000):
    sample_string = df.iloc[i]['Train24']
    l = [list(sample_string[i:i + number_columns]) for i in range(0, len(sample_string), number_columns)]
    if (i == 20000):
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                          np.dtype(int))

        b3 = matrix
    else:
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                           np.dtype(int))
        b3 = np.concatenate((b3, matrix))
    print('Done', i)

print(b3)

for i in range(30000, 40000):
    sample_string = df.iloc[i]['Train24']
    l = [list(sample_string[i:i + number_columns]) for i in range(0, len(sample_string), number_columns)]
    if (i == 30000):
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                          np.dtype(int))
        b4 = matrix
    else:
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                           np.dtype(int))
        b4 = np.concatenate((b4, matrix))
    print('Done', i)

print(b4)

for i in range(40000, 50000):
    sample_string = df.iloc[i]['Train24']
    l = [list(sample_string[i:i + number_columns]) for i in range(0, len(sample_string), number_columns)]
    if (i == 40000):
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                          np.dtype(int))
        b5 =matrix
    else:
        matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
                           np.dtype(int))
        b5 = np.concatenate((b5, matrix))
    print('Done', i)

print(b5)

final_cipher = np.concatenate((b1, b2))
final_cipher = np.concatenate((final_cipher, b3))
final_cipher = np.concatenate((final_cipher, b4))
final_cipher = np.concatenate((final_cipher, b5))

np.save("FinalTrain24.npy", final_cipher)
#
# for i in range(0, 10000):
#     sample_string = df.iloc[i]['Train32']
#     l = [list(sample_string[i:i + number_columns]) for i in range(0, len(sample_string), number_columns)]
#     if (i == 0):
#         matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
#                           np.dtype(int))
#         b1 = matrix
#     else:
#         matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
#                            np.dtype(int))
#         b1 = np.concatenate((b1, matrix))
#     print('Done', i)
#
# print(b1)
#
# for i in range(10000, 20000):
#     sample_string = df.iloc[i]['Train32']
#     l = [list(sample_string[i:i + number_columns]) for i in range(0, len(sample_string), number_columns)]
#     if (i == 10000):
#         matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
#                           np.dtype(int))
#         b2 = matrix
#     else:
#         matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
#                            np.dtype(int))
#         b2 = np.concatenate((b2, matrix))
#     print('Done', i)
#
# print(b2)
#
# for i in range(20000, 30000):
#     sample_string = df.iloc[i]['Train32']
#     l = [list(sample_string[i:i + number_columns]) for i in range(0, len(sample_string), number_columns)]
#     if (i == 20000):
#         matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
#                           np.dtype(int))
#
#         b3 = matrix
#     else:
#         matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
#                            np.dtype(int))
#         b3 = np.concatenate((b3, matrix))
#     print('Done', i)
#
# print(b3)
#
# for i in range(30000, 40000):
#     sample_string = df.iloc[i]['Train32']
#     l = [list(sample_string[i:i + number_columns]) for i in range(0, len(sample_string), number_columns)]
#     if (i == 30000):
#         matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
#                           np.dtype(int))
#         b4 = matrix
#     else:
#         matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
#                            np.dtype(int))
#         b4 = np.concatenate((b4, matrix))
#     print('Done', i)
#
# print(b4)
#
# for i in range(40000, 50000):
#     sample_string = df.iloc[i]['Train32']
#     l = [list(sample_string[i:i + number_columns]) for i in range(0, len(sample_string), number_columns)]
#     if (i == 40000):
#         matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
#                           np.dtype(int))
#         b5 =matrix
#     else:
#         matrix = np.array([s if len(s) == number_columns else s + [None] * (number_columns - len(s)) for s in l],
#                            np.dtype(int))
#         b5 = np.concatenate((b5, matrix))
#     print('Done', i)
#
# print(b5)
#
# final_cipher = np.concatenate((b1, b2))
# final_cipher = np.concatenate((final_cipher, b3))
# final_cipher = np.concatenate((final_cipher, b4))
# final_cipher = np.concatenate((final_cipher, b5))
#
# np.save("FinalTrain32.npy", final_cipher)