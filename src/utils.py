import codecs
import pickle
import sys

from Crypto.Hash import SHA256
sha256hash = lambda x: SHA256.new(x).digest()


def obj_to_pickle_string(x):
    return codecs.encode(pickle.dumps(x), "base64").decode()
    # return msgpack.packb(x, default=msgpack_numpy.encode)
    # TODO: compare pickle vs msgpack vs json for serialization; tradeoff: computation vs network IO

def pickle_string_to_obj(s):
    return pickle.loads(codecs.decode(s.encode(), "base64"))
    # return msgpack.unpackb(s, object_hook=msgpack_numpy.decode)
