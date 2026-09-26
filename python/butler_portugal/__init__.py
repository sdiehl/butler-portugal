"""Python bindings for butler-portugal over its C API, using ctypes."""

import ctypes as C
import os
import sys
from pathlib import Path

__all__ = ["Tensor", "canonicalize", "SYMMETRIC", "ANTISYMMETRIC", "ABSENT"]

SYMMETRIC, ANTISYMMETRIC, ABSENT = 0, 1, 2


def _load():
    name = {"darwin": "libbutler_portugal.dylib", "win32": "butler_portugal.dll"}.get(
        sys.platform, "libbutler_portugal.so"
    )
    here = Path(__file__).parent
    bundled = [p for p in here.iterdir() if p.suffix in (".so", ".pyd")]
    for path in [
        os.environ.get("BUTLER_PORTUGAL_LIB"),
        *bundled,
        here.parents[1] / "target" / "release" / name,
    ]:
        if path and Path(path).exists():
            return C.CDLL(str(path))
    raise OSError(f"{name} not found; build it with `cargo build --release`")


_lib = _load()
_H, _S, _Z = C.c_void_p, C.c_char_p, C.c_size_t
for fn, res, args in [
    ("bp_index_new", _H, [_S, _Z]),
    ("bp_index_contravariant", _H, [_S, _Z]),
    ("bp_index_set_type", C.c_int, [_H, _S]),
    ("bp_index_free", None, [_H]),
    ("bp_symmetry_symmetric", _H, [C.POINTER(_Z), _Z]),
    ("bp_symmetry_antisymmetric", _H, [C.POINTER(_Z), _Z]),
    ("bp_symmetry_symmetric_pairs", _H, [C.POINTER(_Z), _Z]),
    ("bp_symmetry_cyclic", _H, [C.POINTER(_Z), _Z]),
    ("bp_symmetry_custom", _H, [C.POINTER(_Z), C.POINTER(C.c_int32), _Z, _Z]),
    ("bp_symmetry_free", None, [_H]),
    ("bp_tensor_with_coefficient", _H, [_S, C.POINTER(_H), _Z, C.c_int32]),
    ("bp_tensor_free", None, [_H]),
    ("bp_tensor_add_symmetry", C.c_int, [_H, _H]),
    ("bp_tensor_set_metric", C.c_int, [_H, _S, C.c_int32]),
    ("bp_tensor_product", _H, [C.POINTER(_H), _Z, C.POINTER(C.c_int)]),
    ("bp_tensor_rank", _Z, [_H]),
    ("bp_tensor_coefficient", C.c_int32, [_H]),
    ("bp_tensor_to_string", C.c_void_p, [_H]),
    ("bp_string_free", None, [C.c_void_p]),
    ("bp_canonicalize", _H, [_H, C.POINTER(C.c_int)]),
]:
    f = getattr(_lib, fn)
    f.restype, f.argtypes = res, args


def _array(ctype, xs):
    xs = list(xs)
    return (ctype * len(xs))(*xs), len(xs)


def _check(code, what):
    if code != 0:
        raise ValueError(f"{what} failed with code {code}")


class Tensor:
    """A tensor. Indices are names, `^a` for contravariant, `a:type` for a type."""

    def __init__(self, name, indices, coefficient=1):
        handles = []
        try:
            for i, spec in enumerate(indices):
                spec, _, kind = spec.partition(":")
                new = (
                    _lib.bp_index_contravariant
                    if spec.startswith("^")
                    else _lib.bp_index_new
                )
                h = new(spec.lstrip("^").encode(), i)
                handles.append(h)
                if kind:
                    _check(_lib.bp_index_set_type(h, kind.encode()), "index type")
            arr, n = _array(_H, handles)
            self._h = _lib.bp_tensor_with_coefficient(
                name.encode(), arr, n, coefficient
            )
        finally:
            for h in handles:
                _lib.bp_index_free(h)
        if not self._h:
            raise ValueError(f"invalid tensor {name}")

    @classmethod
    def _wrap(cls, handle, err):
        if not handle:
            raise ValueError(f"operation failed with code {err.value}")
        t = cls.__new__(cls)
        t._h = handle
        return t

    def __del__(self):
        if getattr(self, "_h", None):
            _lib.bp_tensor_free(self._h)
            self._h = None

    def _add(self, sym):
        if not sym:
            raise ValueError("invalid symmetry")
        try:
            _check(_lib.bp_tensor_add_symmetry(self._h, sym), "add symmetry")
        finally:
            _lib.bp_symmetry_free(sym)
        return self

    def symmetric(self, *slots):
        return self._add(_lib.bp_symmetry_symmetric(*_array(_Z, slots)))

    def antisymmetric(self, *slots):
        return self._add(_lib.bp_symmetry_antisymmetric(*_array(_Z, slots)))

    def symmetric_pairs(self, *pairs):
        arr, n = _array(_Z, (s for p in pairs for s in p))
        return self._add(_lib.bp_symmetry_symmetric_pairs(arr, n // 2))

    def cyclic(self, *slots):
        return self._add(_lib.bp_symmetry_cyclic(*_array(_Z, slots)))

    def custom(self, *generators):
        """Signed generators `(perm, sign)`; new slot i takes old slot perm[i]."""
        perms, _ = _array(_Z, (s for p, _ in generators for s in p))
        signs, n = _array(C.c_int32, (s for _, s in generators))
        return self._add(_lib.bp_symmetry_custom(perms, signs, n, self.rank))

    def metric(self, kind, index_type=""):
        _check(
            _lib.bp_tensor_set_metric(self._h, index_type.encode(), kind), "set metric"
        )
        return self

    def __mul__(self, other):
        arr, n = _array(_H, [self._h, other._h])
        err = C.c_int()
        return Tensor._wrap(_lib.bp_tensor_product(arr, n, C.byref(err)), err)

    def canonicalize(self):
        err = C.c_int()
        return Tensor._wrap(_lib.bp_canonicalize(self._h, C.byref(err)), err)

    @property
    def rank(self):
        return _lib.bp_tensor_rank(self._h)

    @property
    def coefficient(self):
        return _lib.bp_tensor_coefficient(self._h)

    def __str__(self):
        p = _lib.bp_tensor_to_string(self._h)
        try:
            return C.string_at(p).decode()
        finally:
            _lib.bp_string_free(p)

    __repr__ = __str__


def canonicalize(tensor):
    return tensor.canonicalize()
