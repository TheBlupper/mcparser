from __future__ import annotations
import os.path
import struct
import zlib
import gzip
from math import log2, ceil
from dataclasses import dataclass, field, replace
from typing import BinaryIO, Optional
from nbt import load_nbt_bytes, \
    TAG_Compound, TAG_Long_Array

'''
Notation:
x, y, z = coordinates of a block
cx, cy, cz = coordinates of a chunk
xc, yc, zc = coordinates within a chunk
rx, ry, rz = coordinates of a region
xr, yr, zr = coordinates within a region
cxr, czr = coordinates of a chunk witihin a region
'''

__all__ = [
    'Block', 'World',
]


@dataclass(frozen=True)
class Block:
    name: str
    properties: TAG_Compound = field(default_factory=dict)
    block_light: int = 0
    sky_light: int = 0
    block_entity: Optional[TAG_Compound] = None

    def replace(self, **changes):
        return replace(self, **changes)


AIR = Block('minecraft:air')


def _get_long_array_bits(
    long_arr: TAG_Long_Array,
    val_size: int,
    idx: int
) -> int:
    mask = (1 << val_size) - 1
    vals_per_long = 64 // val_size
    long_idx = idx // vals_per_long
    offset = (idx % vals_per_long)*val_size

    long = long_arr[long_idx]
    return (long >> offset) & mask


class _Chunk:
    _nbt: TAG_Compound
    _sections: list[TAG_Compound]
    _block_entities: dict[tuple[int, int], TAG_Compound]

    def __init__(self, nbt: TAG_Compound):
        self._nbt = nbt

        self.x = self._nbt['xPos'] << 4
        self.y = self._nbt['yPos'] << 4
        self.z = self._nbt['zPos'] << 4

        self._sections = {i-4: self._nbt['sections'][i] for i in range(24)}
        self._block_entities = {
            (be['x'], be['y'], be['z']): be
            for be in self._nbt['block_entities']
        }

    def get_block(self, xc, y, zc):
        yc = y % 16
        section = self._sections[y >> 4]
        block_states = section['block_states']
        palette = block_states['palette']

        block_entity = self._block_entities.get(
            (self.x+xc, y, self.z+zc), None)
        if 'data' not in block_states:
            tag = palette[0]
            return Block(
                tag['Name'],
                tag.get('Properties', TAG_Compound()),
                block_entity=block_entity
            )
        palette_data = block_states['data']

        bits_per_idx = max(4, ceil(log2(len(palette)-1)))
        block_idx = yc*16*16 + zc*16 + xc

        palette_id = _get_long_array_bits(
            palette_data, bits_per_idx, block_idx)
        tag = palette[palette_id]

        block_light = 0
        if 'BlockLight' in section:
            block_light = _get_long_array_bits(
                section['BlockLight'], 4, block_idx)

        sky_light = 0
        if 'SkyLight' in section:
            sky_light = _get_long_array_bits(section['SkyLight'], 4, block_idx)

        return Block(
            tag['Name'],
            tag.get('Properties', TAG_Compound()),
            block_light,
            sky_light,
            block_entity
        )

    @classmethod
    def from_bytes(cls, data: bytes):
        return cls(load_nbt_bytes(data))


class World:
    _region_fds: dict[tuple[int, int], BinaryIO]
    _chunks: dict[tuple[int, int], _Chunk]

    def __init__(self, wdir: str):
        if not (os.path.isdir(wdir) and self._is_world(wdir)):
            raise FileNotFoundError(
                f'Could not find a Minecraft world at {wdir!r}')
        self._wdir = wdir
        self._region_fds = {}
        self._chunks = {}

    def _load_region_fd(self, rx: int, rz: int):
        path = os.path.join(self._wdir, 'region', f'r.{rx}.{rz}.mca')
        self._region_fds[rx, rz] = open(path, 'rb')

    def _load_chunk(self, cx: int, cz: int):
        '''Will load the chunk at (cx, cz) into _chunks,
        will reload even if chunk is already loaded'''
        rx, rz = cx >> 5, cz >> 5
        if (rx, rz) not in self._region_fds:
            self._load_region_fd(rx, rz)
        fd = self._region_fds[rx, rz]

        cxr, czr = cx % 512, cz % 512
        header_off = 4 * ((cxr & 31) + (czr & 31)*32)
        fd.seek(header_off)

        # Offset is stored as 3 bytes, in increments of 4096
        data_off = 4096 * struct.unpack_from('>i', b'\0'+fd.read(3))[0]
        # Upper bound of chunk size, not interesting
        _ = 4096 * fd.read(1)[0]

        fd.seek(data_off)
        data_len = struct.unpack('>i', fd.read(4))[0]
        compr_type = fd.read(1)[0]
        compr_func = [
            None,       # Should never happen
            gzip.decompress,
            zlib.decompress,
            lambda x:x  # No compression
        ][compr_type]
        self._chunks[cx, cz] = _Chunk.from_bytes(
            compr_func(fd.read(data_len))
        )

    def get_block(self, x, y, z):
        # Chunk coordinates
        cx, cz = x >> 4, z >> 4
        if (cx, cz) not in self._chunks:
            self._load_chunk(cx, cz)
        chunk = self._chunks[cx, cz]
        return chunk.get_block(x & 0xf, y, z & 0xf)

    def __getitem__(self, coords: tuple[int, int, int]):
        assert isinstance(coords, tuple) and len(coords) == 3
        return self.get_block(*coords)

    @staticmethod
    def _is_world(wdir: str):
        return os.path.isfile(os.path.join(wdir, 'level.dat')) and \
            os.path.isdir(os.path.join(wdir, 'region'))