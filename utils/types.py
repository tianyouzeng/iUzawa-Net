from typing import Literal, TypedDict


class UzawaPtwiseModuleParams(TypedDict):
    qa_width: int
    qb_width: int
    qb_width_pre_lift: int
    fno_layers: int
    fno_width: int
    fno_width_pre_proj: int


SATypes = Literal['sa', 'sa_time', 'none']