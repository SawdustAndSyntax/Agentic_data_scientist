from __future__ import annotations
from collections import deque
from .contracts import SemanticEntity, SemanticRelationship, JoinPlan

class SemanticGraph:
    def __init__(self, entities=None, relationships=None):
        self.entities: dict[str,SemanticEntity] = {}
        self.relationships: list[SemanticRelationship] = []
        for e in entities or []: self.add_entity(e)
        for r in relationships or []: self.add_relationship(r)

    def add_entity(self, entity: SemanticEntity):
        self.entities[entity.qualified_name]=entity
        return self

    def add_relationship(self, rel: SemanticRelationship):
        self.relationships.append(rel)
        return self

    def neighbors(self, node: str):
        out=[]
        for r in self.relationships:
            if r.source==node: out.append((r.target,r))
            elif r.target==node: out.append((r.source,r))
        return out

    def shortest_path(self, source: str, target: str, max_hops: int=3) -> JoinPlan | None:
        if source==target:
            return JoinPlan([source],[],1.0,[])
        q=deque([(source,[source],[])])
        seen={source}
        while q:
            node,path,rels=q.popleft()
            if len(rels)>=max_hops: continue
            for nxt,rel in self.neighbors(node):
                if nxt in seen: continue
                npath=path+[nxt]; nrels=rels+[rel]
                if nxt==target:
                    confidence=1.0
                    warnings=[]
                    for r in nrels:
                        confidence*=r.confidence
                        if not r.source_keys or not r.target_keys:
                            warnings.append(f'Relationship {r.name} lacks explicit join keys')
                    return JoinPlan(npath,nrels,confidence,warnings)
                seen.add(nxt); q.append((nxt,npath,nrels))
        return None
