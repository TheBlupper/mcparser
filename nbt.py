from __future__ import annotations
from typing import BinaryIO, Optional
from io import BytesIO
from struct import pack, unpack, calcsize
import numpy as np

__all__ = [
    'TAG_End',      'TAG_Byte',         'TAG_Short',
    'TAG_Int',      'TAG_Long',         'TAG_Float',
    'TAG_Double',   'TAG_Byte_Array',   'TAG_String',
    'TAG_List',     'TAG_Compound',     'TAG_Int_Array',
    'TAG_Long_Array',
    
    'load_nbt_file',
    'load_nbt_buf',
    'load_nbt_bytes',
    'save_nbt_file',
    'nbt_to_bytes'
]


class NbtTag:
    id: int

    @classmethod
    def from_buf(cls, buf: BinaryIO):
        '''Construct the TAG from a binary readable buffer'''
        raise NotImplementedError(
            'from_buf() needs to be declared on each TAG')

    def to_buf(self, buf: BinaryIO) -> int:
        '''Write the binary representation of a TAG to the provided buffer,
        returning the amount written'''
        raise NotImplementedError('to_buf() needs to be declared on each TAG')


class _primitive_tag(NbtTag):
    '''A tag easily representable by a format string
    for `struct.unpack` and `struct.pack`'''
    _fmt: str

    @classmethod
    def from_buf(cls, buf: BinaryIO):
        return cls(*unpack(cls._fmt, buf.read(calcsize(cls._fmt))))

    def to_buf(self, buf: BinaryIO) -> int:
        return buf.write(pack(self._fmt, self))


class TAG_Byte(_primitive_tag, int):
    id = 1
    _fmt = '>b'


class TAG_Short(_primitive_tag, int):
    id = 2
    _fmt = '>h'


class TAG_Int(_primitive_tag, int):
    id = 3
    _fmt = '>i'


class TAG_Long(_primitive_tag, int):
    id = 4
    _fmt = '>q'


class TAG_Float(_primitive_tag, float):
    id = 5
    _fmt = '>f'


class TAG_Double(_primitive_tag, float):
    id = 6
    _fmt = '>d'


class _array_tag(np.ndarray):
    '''An array tag defined by a numpy dtype'''
    _dtype: np.dtype

    @classmethod
    def from_buf(cls, buf: BinaryIO):
        arr_len = unpack('>i', buf.read(4))[0]
        arr_itemsize = cls._dtype.itemsize
        arr_size = arr_len * arr_itemsize
        return np.frombuffer(
            buf.read(arr_size), dtype=cls._dtype, count=arr_len
        ).view(cls)

    def to_buf(self, buf: BinaryIO) -> int:
        start_off = buf.tell()
        TAG_Int(len(self)).to_buf(buf)
        buf.write(self.tobytes())
        return buf.tell() - start_off


class TAG_Byte_Array(_array_tag, NbtTag):
    id = 7
    _dtype = np.dtype('>b')


class TAG_String(NbtTag, str):
    id = 8

    @classmethod
    def from_buf(cls, buf: BinaryIO):
        str_len = unpack('>H', buf.read(2))[0]
        return cls(buf.read(str_len).decode('utf8'))

    def to_buf(self, buf: BinaryIO) -> int:
        encoded = self.encode('utf8')
        return buf.write(pack('>H', len(encoded)) + encoded)


class TAG_List(list[NbtTag], NbtTag):
    id = 9
    item_tag_id: int

    def __init__(self, items: list[NbtTag], item_tag_id: Optional[int] = None):
        if item_tag_id is None:
            item_tag_id = 0 if not items else items[0].id
        self.item_tag_id = item_tag_id
        list.__init__(self, items)

    @classmethod
    def from_buf(cls, buf: BinaryIO):
        tag_id = buf.read(1)[0]
        tag_count = unpack('>i', buf.read(4))[0]
        items = [TAGS[tag_id].from_buf(buf) for _ in range(tag_count)]
        return cls(items, tag_id)

    def to_buf(self, buf: BinaryIO) -> int:
        start_off = buf.tell()
        TAG_Byte(self.item_tag_id).to_buf(buf)
        TAG_Int(len(self)).to_buf(buf)
        [tag.to_buf(buf) for tag in self]
        return buf.tell() - start_off


class TAG_Compound(dict[str, NbtTag]):
    id = 10

    @classmethod
    def from_buf(cls, buf: BinaryIO):
        '''DO NOT use this directly to load NBT files, use
        `load_nbt_file`, `load_nbt_buf` or `load_nbt_bytes` instead.'''
        compound_dict = {}
        while True:
            try:
                tag_id = buf.read(1)[0]
            except IndexError:  # EOF
                break
            if tag_id == TAG_End.id:
                break
            name_len = unpack('>H', buf.read(2))[0]
            name = buf.read(name_len).decode('utf8')
            compound_dict[name] = TAGS[tag_id].from_buf(buf)
        return cls(compound_dict)

    def to_buf(self, buf: BinaryIO) -> int:
        '''DO NOT use this directly to save NBT files, use
        `save_nbt_file`, `save_nbt_buf` or `nbt_to_bytes` instead.'''
        start_off = buf.tell()
        for k, v in self.items():
            buf.write(bytes([v.id]))
            TAG_String(k).to_buf(buf)
            v.to_buf(buf)
        TAG_End().to_buf(buf)
        return buf.tell() - start_off

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise ValueError(
                f'TAG_Compound may only have str\'s as keys, '
                f'not {type(key)}')
        if not isinstance(value, NbtTag):
            raise ValueError(
                f'TAG_Compound may only have NbtTag\'s as values, '
                f'not {type(value)}')
        super().__setitem__(key, value)


class TAG_Int_Array(_array_tag, NbtTag):
    id = 11
    _dtype = np.dtype('>i')


class TAG_Long_Array(_array_tag, NbtTag):
    id = 12
    _dtype = np.dtype('>q')


class TAG_End(NbtTag):
    id = 0

    def to_buf(self, buf: BinaryIO) -> int:
        return buf.write(b'\0')


# Index is the same as the tag's id
TAGS = [
    TAG_End,        TAG_Byte,       TAG_Short,
    TAG_Int,        TAG_Long,       TAG_Float,
    TAG_Double,     TAG_Byte_Array, TAG_String,
    TAG_List,       TAG_Compound,   TAG_Int_Array,
    TAG_Long_Array
]

# We access [''] on all tags presumably loaded from a file
# b/c an NBT file is a TAG_Compound with an empty name

# For the same reason we wrap tags in a TAG_Compound when saving them


def load_nbt_file(fn: str) -> TAG_Compound:
    with open(fn, 'rb') as f:
        root = TAG_Compound.from_buf(f)['']
        assert isinstance(root, TAG_Compound)
    return root


def load_nbt_buf(buf: BinaryIO) -> TAG_Compound:
    root = TAG_Compound.from_buf(buf)['']
    assert isinstance(root, TAG_Compound)
    return root


def load_nbt_bytes(by: bytes) -> TAG_Compound:
    root = TAG_Compound.from_buf(BytesIO(by))['']
    assert isinstance(root, TAG_Compound)
    return root


def save_nbt_file(nbt: TAG_Compound, fn: str):
    with open(fn, 'wb') as f:
        root = TAG_Compound({'': nbt}).to_buf(f)
        assert isinstance(root, TAG_Compound)
    return root


def save_nbt_buf(nbt: TAG_Compound, buf: BinaryIO):
    TAG_Compound({'': nbt}).to_buf(buf)


def nbt_to_bytes(nbt: TAG_Compound):
    buf = BytesIO()
    TAG_Compound({'': nbt}).to_buf(buf)
    return buf.getvalue()