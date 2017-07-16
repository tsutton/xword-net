""" This module is for processing the word2vec embdding from https://code.google.com/archive/p/word2vec/ (GoogleNews-vectors-negative300.bin.gz) and the crossword clue database from http://www.otsys.com/clue/.

(format of the data, preprocessing, other options)
"""
import struct
import numpy as np
import random
import pickle

embedding_filename = "GoogleNews-vectors-negative300.bin"

clues_filename = "clues-5-stripped.txt"
pickled_index_filename = "picked-index.bin"
max_clue_length = 40

def read_until(fd, b):
    """Read bytes from open file object fd until byte b is found"""
    s = b''
    while True:
        c = fd.read(1)
        if not c: break #no more bytes to read
        if c == b: break #found our byte
        s += c
    return s

def build_index(fname):
    """Given the path to the word2vec model, build an index of {bytestring: file offset}

This assumes that the file consists of a header of the form:
<num words> <another number>\\n
and then lines of the form:
<word><space><1200 bytes of embedding>\\n
which is the format of the GoogleNews 300 dimensional embedding.
"""
    d = {}
    with open(fname, "rb") as f:
        # read dimension
        count = int(read_until(f, b' '))
        print("count: ",count)
        read_until(f, b'\n')

        #read words
        for i in range(count):
            if i % 100000 == 0: print(i)
            # attempt to read next word
            idx  = f.tell()
            word = read_until(f, b' ')
            d[word]=idx

            if not word:
                break
            else:
                f.read(1200)

        return d

def lookup(words, index, filename, length=None, postpad=False):
    """Return a numpy array of the embeddings of each word in words, using the pre-made index for the given filename.

If length is not none and is bigger than the number of words, the result will be padded with 0s.
The default is to prepend 0s, but appending can be set by giving postpad=True.
The returned shape is (n, 300) where n is either the number of words, or length.
"""

    l = len(words)
    pad = 0
    if length and length > l:
        pad  = length - l
        l = length

    a = np.zeros( (l, 300) , dtype=np.float32)

    with open(filename, "rb") as f:
        for w,i in zip(words, range(l)):

            b = bytes(w, encoding="latin-1")
            if b not in index:
                continue

            f.seek(index[b])
            check = read_until(f, b' ')
            if not check == b:
                raise RuntimeError("index incorrect for word {}".format(i))

            v = struct.unpack('f'*300, f.read(1200))
            if postpad:
                a[i] = np.array(v, dtype=np.float32)
            else:
                a[i+pad] = np.array(v, dtype=np.float32)

    return a

def guess_to_letters(guess):
    """Given an (n,m,26) array, return a list of n words, m letters each

Picks the highest number from each"""

    z = guess.argmax(axis=2).tolist()
    #z has shape (n, m)

    abc = 'abcdefghijklmnopqrstuvwxyz'
    return list(map(lambda l: ''.join(abc[x] for x in l), z))

def letters_to_one_hot(answer):
    """Turns a string of length l into an (l,26) one-hot numpy array

Anything not in [a-zA-Z] goes to all 0s"""
    l = len(answer)
    a = np.zeros( (l, 26) )
    abc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i in range(l):
        try:
            j = abc.index(answer[i].upper())
            a[i,j]=1.0
        except ValueError:
            pass

    return a


class Clues5Generator:

    def __init__(self, batch_size, clue_path=clues_filename, embedding_path=embedding_filename, length=max_clue_length, pickled_index=None, read_pickle=False, write_pickle=False, pad="post"):

        if (write_pickle or read_pickle) and not pickled_index:
            raise RuntimeError("Told to read or write pickle, but not given file name")
        elif read_pickle:
            with open(pickled_index, "rb") as f:
                print("Loading pickled index {}".format(pickled_index))
                self.index = pickle.load(f)
        elif not embedding_path:
            raise RuntimeError("No embedding_path and (not write_pickle)")
        else:
            print("Building index from binary embedding file {}".format(embedding_path))
            self.index = build_index(embedding_path)
            if write_pickle:
                print("Writing to pickle index {}".format(pickled_index))
                with open(pickled_index, "rb") as f:
                    pickle.dump(self.index, f)

        self.embedding_path = embedding_path
        self.batch_size = batch_size

        with open(clue_path, "r") as f:
            self.clues = f.readlines()

        self.num_clues = len(self.clues)
        print("num clues: ", self.num_clues)

        self.left_in_epoch = list(range(self.num_clues))
        random.shuffle(self.left_in_epoch)
        self.num_left_in_epoch = self.num_clues

        self.vector_length = length

        self.pad = pad

    def __iter__(self):
        return self

    def __next__(self):

        if self.num_left_in_epoch < self.batch_size:
            tmp = list(range(self.num_clues))
            random.shuffle(tmp)
            self.left_in_epoch.append(tmp)
            self.num_left_in_epoch += self.num_clues
            
        x = np.zeros( (self.batch_size, self.vector_length, 300), dtype=np.float32)
        y = np.zeros( (self.batch_size, 5, 26), dtype=np.float32)
        l = np.zeros( (self.batch_size,), dtype=np.int32)

        for i in range(self.batch_size):
            answer, *clue = self.clues[self.left_in_epoch[i]].strip().split(' ')

            x[i] = lookup(clue, self.index, self.embedding_path, self.vector_length,
                          True if self.pad=="post" else False)
            y[i] = letters_to_one_hot(answer)
            l[i] = len(clue)

        del self.left_in_epoch[:self.batch_size]
        self.num_left_in_epoch -= self.batch_size
        return (x,y,l)

    def next_with_english(self):
        if self.num_left_in_epoch < self.batch_size:
            tmp = list(range(self.num_clues))
            random.shuffle(tmp)
            self.left_in_epoch.append(tmp)
            self.num_left_in_epoch += self.num_clues
            
        x = np.zeros( (self.batch_size, self.vector_length, 300), dtype=np.float32)
        y = np.zeros( (self.batch_size, 5, 26), dtype=np.float32)
        l = np.zeros( (self.batch_size,), dtype=np.int32)
        xe = []
        ye = []
        for i in range(self.batch_size):
            answer, *clue = self.clues[self.left_in_epoch[i]].strip().split(' ')

            x[i] = lookup(clue, self.index, self.embedding_path, self.vector_length)
            y[i] = letters_to_one_hot(answer)
            l[i] = len(clue)
            xe.append(clue)
            ye.append(answer)

        del self.left_in_epoch[:self.batch_size]
        self.num_left_in_epoch -= self.batch_size
        return (x,y,l,xe,ye)
