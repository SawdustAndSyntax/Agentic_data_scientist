from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
from ..contracts import SemanticEntity, SemanticRelationship

class CatalogAdapter(ABC):
    platform='generic'
    @abstractmethod
    def entities(self) -> list[SemanticEntity]: ...
    @abstractmethod
    def relationships(self) -> list[SemanticRelationship]: ...

    def profile(self, qualified_name: str, sample_rows: int=1000) -> dict:
        return {'qualified_name':qualified_name,'sample_rows':0,'coverage':0.5,'quality':0.5}

    def sample(self, qualified_name: str, limit: int=1000) -> pd.DataFrame:
        raise NotImplementedError
