"""Structures de données partagées."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Candidate:
    id: str
    label: str = ""
    description: str = ""

    def short(self) -> str:
        return f"{self.id} ({self.label})" if self.label else self.id


@dataclass
class Linking:
    """Résultat de la liaison : mentions de la question -> candidats Wikidata."""
    language: str = "en"
    entities: dict[str, list[Candidate]] = field(default_factory=dict)
    properties: dict[str, list[Candidate]] = field(default_factory=dict)

    def top_entities(self) -> list[Candidate]:
        return [c[0] for c in self.entities.values() if c]

    def top_properties(self) -> list[Candidate]:
        return [c[0] for c in self.properties.values() if c]

    def all_ids(self) -> set[str]:
        out: set[str] = set()
        for group in (self.entities, self.properties):
            for cands in group.values():
                out.update(c.id for c in cands)
        return out


@dataclass
class Generated:
    """Sortie brute d'une technique."""
    technique: str
    query: str | None = None
    raw: str | None = None
    error: str | None = None
    seconds: float = 0.0
    allowed_ids: set[str] = field(default_factory=set)  # identifiants fournis dans le prompt
    notes: list[str] = field(default_factory=list)      # parties optionnelles indisponibles


@dataclass
class Checked:
    """Sortie du filtre + normalisation pour une technique."""
    technique: str
    query: str | None
    ok: bool
    reasons: list[str] = field(default_factory=list)    # motifs de rejet
    warnings: list[str] = field(default_factory=list)   # alertes non bloquantes
    form: tuple | None = None                           # forme normalisée (triplets)


@dataclass
class Decision:
    query: str | None
    technique: str | None
    rule: str          # accord | priorite | unique | aucun
    detail: str = ""


@dataclass
class Agent1Result:
    question: str
    query: str | None
    technique: str | None
    rule: str
    detail: str
    linking: Linking
    generated: list[Generated]
    checked: list[Checked]
    seconds: float

    def to_dict(self) -> dict:
        d = asdict(self)
        for g in d["generated"]:
            g["allowed_ids"] = sorted(g["allowed_ids"])
        for c in d["checked"]:
            c["form"] = repr(c["form"]) if c["form"] is not None else None
        return d
