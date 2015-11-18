import os
import zipfile
import math
import collections
from six.moves import urllib


class dataset:


    def __init__(self, vocabulary_size):
        self.vocabulary_size = vocabulary_size
        self.url = 'http://mattmahoney.net/dc/'


    def maybe_download(self, filename, expected_bytes):
        """Download a file if not present, and make sure it's the right size."""
        if not os.path.exists(filename):
            filename, _ = urllib.request.urlretrieve(self.url + filename, filename)

        statinfo = os.stat(filename)
        if statinfo.st_size == expected_bytes:
            print('Found and verified', filename)
        else:
            print(statinfo.st_size)
            raise Exception(
                'Failed to verify ' + filename + '. Can you get to it with a browser?')

        return filename


    # Read the data into a string.
    def read_data(self, filename):
        f = zipfile.ZipFile(filename)
        data = f.read(f.namelist()[0]).split()
        f.close()
        return data




    def build_dataset(self, words):
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(self.vocabulary_size - 1))
        dictionary = dict()

        for word, _ in count:
            dictionary[word] = len(dictionary)

        data = list()
        unk_count = 0

        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count = unk_count + 1

            data.append(index)

        count[0][1] = unk_count
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

        return data, count, dictionary, reverse_dictionary


    def load(self):

        filename = self.maybe_download('text8.zip', 31344016)
        words = self.read_data(filename)
        print('Data size', len(words))

        data, count, dictionary, reverse_dictionary = self.build_dataset(words)
        del words  # Hint to reduce memory.
        print('Most common words (+UNK)', count[:5])
        print('Sample data', data[:10])

        return data, count, dictionary, reverse_dictionary
