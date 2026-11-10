from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TaskId(_message.Message):
    __slots__ = ("worker_id", "seq")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    SEQ_FIELD_NUMBER: _ClassVar[int]
    worker_id: int
    seq: int
    def __init__(self, worker_id: _Optional[int] = ..., seq: _Optional[int] = ...) -> None: ...

class KeyList(_message.Message):
    __slots__ = ("keys",)
    KEYS_FIELD_NUMBER: _ClassVar[int]
    keys: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, keys: _Optional[_Iterable[int]] = ...) -> None: ...

class KeyRange(_message.Message):
    __slots__ = ("begin", "end")
    BEGIN_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    begin: int
    end: int
    def __init__(self, begin: _Optional[int] = ..., end: _Optional[int] = ...) -> None: ...

class KeySelector(_message.Message):
    __slots__ = ("sparse", "dense")
    SPARSE_FIELD_NUMBER: _ClassVar[int]
    DENSE_FIELD_NUMBER: _ClassVar[int]
    sparse: KeyList
    dense: KeyRange
    def __init__(self, sparse: _Optional[_Union[KeyList, _Mapping]] = ..., dense: _Optional[_Union[KeyRange, _Mapping]] = ...) -> None: ...

class Tensor(_message.Message):
    __slots__ = ("width", "data")
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    width: int
    data: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, width: _Optional[int] = ..., data: _Optional[_Iterable[float]] = ...) -> None: ...

class PullRequest(_message.Message):
    __slots__ = ("model_id", "target", "task_id", "deps")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    DEPS_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    target: KeySelector
    task_id: TaskId
    deps: _containers.RepeatedCompositeFieldContainer[TaskId]
    def __init__(self, model_id: _Optional[str] = ..., target: _Optional[_Union[KeySelector, _Mapping]] = ..., task_id: _Optional[_Union[TaskId, _Mapping]] = ..., deps: _Optional[_Iterable[_Union[TaskId, _Mapping]]] = ...) -> None: ...

class PullResponse(_message.Message):
    __slots__ = ("weights", "keys")
    WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    KEYS_FIELD_NUMBER: _ClassVar[int]
    weights: Tensor
    keys: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, weights: _Optional[_Union[Tensor, _Mapping]] = ..., keys: _Optional[_Iterable[int]] = ...) -> None: ...

class PushRequest(_message.Message):
    __slots__ = ("model_id", "target", "gradients", "task_id", "deps")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    GRADIENTS_FIELD_NUMBER: _ClassVar[int]
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    DEPS_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    target: KeySelector
    gradients: Tensor
    task_id: TaskId
    deps: _containers.RepeatedCompositeFieldContainer[TaskId]
    def __init__(self, model_id: _Optional[str] = ..., target: _Optional[_Union[KeySelector, _Mapping]] = ..., gradients: _Optional[_Union[Tensor, _Mapping]] = ..., task_id: _Optional[_Union[TaskId, _Mapping]] = ..., deps: _Optional[_Iterable[_Union[TaskId, _Mapping]]] = ...) -> None: ...

class PushResponse(_message.Message):
    __slots__ = ("task_id",)
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    task_id: TaskId
    def __init__(self, task_id: _Optional[_Union[TaskId, _Mapping]] = ...) -> None: ...
