"""deep2048 random router."""
from ..specs.random import Key, SplitKey
from fastapi import APIRouter
from typing import List

router = APIRouter()


@router.put('/split-key', response_model=List[Key])
def split_key(spec: SplitKey):
    """Split random key into n children keys."""
    return spec.split()
