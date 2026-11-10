"""Agent 3 : verbalisation GraphRAG de la réponse, notes par correspondance de motifs
(DeBellis et Neches, 2026 : triplets -> phrases déterministes, regroupées par sujet).

Entrée : la question, l'exécution de l'Agent 2 et la liaison de l'Agent 1.
1. Si la requête n'a pas produit de réponse (vide, syntaxe, timeout, aucune requête),
   l'Agent 3 le dit tel quel, sans appeler le LLM : rien n'est inventé.
2. Sinon, il construit des notes (1 saut) pour les entités de la réponse et les entités
   principales de la question, puis demande au LLM de répondre UNIQUEMENT d'après ces notes
   et le résultat de la requête.
3. Contrôle de fidélité (sans LLM) : tout nombre ou identifiant Q cité dans la réponse doit
   figurer dans le contexte fourni ; sinon `fidelity.supported` est faux.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

from agent1.llm import LLMClient
from agent1.types import Linking

from .notes import NoteBuilder

SYSTEM = (
    "You answer a question using ONLY the information given below: the query result and "
    "the notes about the entities. Rules: (1) Answer in one "
    "or two sentences. (2) Never use knowledge that is not in the notes or the result. "
    "(3) Cite the notes you used as [N1], [N2]. (4) If the notes and result do not let you "
    "answer, reply exactly: INSUFFISANT. (5) The SPARQL query shows what the result means "
    "(which entity is the subject and which the object of each relation): respect that "
    "direction. (6) If the result is a count or a number, give exactly that number; never "
    "recount from the notes, which are only a sample. (7) Write the answer in the language "
    "given on the line 'Answer language', whatever language the notes are in.")

NO_ANSWER = {
    "vide": "Aucune réponse trouvée : la requête s'est exécutée mais n'a rien renvoyé.",
    "syntaxe": "Aucune réponse : la requête générée est rejetée par Wikidata (erreur de syntaxe).",
    "timeout": "Aucune réponse : la requête a dépassé le délai de Wikidata.",
    "service": "Aucune réponse : le service Wikidata est indisponible.",
    "sans_requete": "Aucune réponse : aucune requête valide n'a pu être produite.",
}

_NUM = re.compile(r"(?<![\w.])\d[\d,.]*\d|\d")
_QID = re.compile(r"\bQ[1-9]\d*\b")


@dataclass
class Agent3Result:
    question: str
    status: str                       # repondu | insuffisant | sans_reponse | erreur
    answer: str
    exec_status: str = ""
    notes: list[dict] = field(default_factory=list)
    fidelity: dict = field(default_factory=dict)
    error: str = ""
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
_WORDS.update({w: str(i) for i, w in enumerate(
    "zéro un deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze quinze "
    "seize dix-sept dix-huit dix-neuf vingt".split())})
_NUMERIC_CELL = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


def numbers_in(text: str, words: bool = True) -> set[str]:
    """Nombres d'un texte, par jetons entiers (« 3 » ne se trouve pas dans « 1829 »)."""
    out = set()
    for tok in _NUM.findall(text):
        t = tok.replace(",", "").rstrip(".")
        if t:
            out.add(t)
            if "." in t:                      # 5.0 -> 5
                head, _, tail = t.partition(".")
                if tail.strip("0") == "":
                    out.add(head)
    if words:
        for w in re.findall(r"[a-zéèêàùç-]+", text.lower()):
            if w in _WORDS:
                out.add(_WORDS[w])
    return out


def fidelity(answer: str, context: str, expected: str | None = None) -> dict:
    """Nombres et identifiants Q de la réponse absents du contexte ; valeur attendue présente."""
    answer = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", answer)
    ctx_nums = numbers_in(context, words=False)
    unsupported = [n for n in sorted(numbers_in(answer)) if n not in ctx_nums
                   and n not in {x.lstrip("0") or "0" for x in ctx_nums}]
    unsupported += [q for q in _QID.findall(answer) if q not in context]
    missing = None
    if expected is not None:
        want = {expected, expected.split(".")[0] if expected.endswith(".0") else expected}
        if not (want & numbers_in(answer)):
            missing = expected
    # tolère « [ N2 ] », « 【N2】 » et les espaces de largeur nulle écrits par certains modèles
    cite = r"[\[【]\s*(N\d+)\s*[\]】]"
    cited = set(re.findall(cite, answer))
    known = set(re.findall(cite, context))
    bad_cites = sorted(cited - known)
    return {"supported": not unsupported and not bad_cites and missing is None,
            "unsupported": unsupported, "unknown_citations": bad_cites,
            "result_value_missing": missing, "cites_a_note": bool(cited & known),
            "cites_result": bool(re.search(r"[\[【]\s*result\s*[\]】]", answer, re.I))}


def expected_value(execution) -> str | None:
    """Résultat à une seule valeur numérique (COUNT, année...) : la réponse doit la contenir."""
    p = execution.preview
    if execution.ask or len(p) != 1 or len(p[0]) != 1:
        return None
    v = next(iter(p[0].values())).strip()
    return v.lstrip("+") if _NUMERIC_CELL.match(v) else None


def _result_block(execution) -> str:
    if execution.ask:
        return "Query result (yes/no question): " + ("yes" if execution.answers == ["true"] else "no")
    lines = [", ".join(f"{k}={v}" for k, v in row.items()) for row in execution.preview]
    more = f"\n(… {execution.n_rows - len(lines)} more rows)" if execution.n_rows > len(lines) else ""
    return "Query result:\n" + "\n".join(lines) + more


class Agent3:
    def __init__(self, llm: LLMClient, notes: NoteBuilder, max_answer_entities: int = 8,
                 max_question_entities: int = 3):
        self.llm, self.notes = llm, notes
        self.n_ans, self.n_q = max_answer_entities, max_question_entities

    def run(self, question: str, execution, linking: Linking | None = None,
            query: str | None = None) -> Agent3Result:
        t0 = time.time()
        st = execution.status if execution is not None else "sans_requete"
        if execution is None or st not in ("ok",):
            return Agent3Result(question, "sans_reponse", NO_ANSWER.get(st, NO_ANSWER["service"]),
                                exec_status=st, seconds=time.time() - t0)
        try:
            ids: list[str] = []
            for row in execution.preview:
                for v in row.values():
                    if re.fullmatch(r"Q[1-9]\d*", v) and v not in ids:
                        ids.append(v)
            ids = ids[: self.n_ans]
            if linking is not None:
                ids += [c.id for c in linking.top_entities()[: self.n_q] if c.id not in ids]
            notes = self.notes.build(ids, question)
            lang = (linking.language if linking is not None and linking.language else "en")
            head = f"Answer language: {lang}\n"
            if query:
                head += f"SPARQL query that produced the result:\n{query.strip()}\n\n"
            context = head + _result_block(execution) + "\n\nNotes:\n" + "\n".join(n.render() for n in notes)
            raw = self.llm.chat(SYSTEM, f"{context}\n\nQuestion: {question}\nAnswer:").strip()
            if raw.upper().startswith("INSUFFISANT"):
                return Agent3Result(question, "insuffisant",
                                    "Les informations récupérées ne permettent pas de répondre.",
                                    exec_status=st, notes=[asdict(n) for n in notes],
                                    seconds=time.time() - t0)
            expected = expected_value(execution)
            fid = fidelity(raw, context + "\n" + question, expected)
            if not fid["supported"] and (fid["unsupported"] or fid["result_value_missing"]):
                why = (f"the query result is {expected}" if fid["result_value_missing"]
                       else f"these numbers are not in the notes or result: {fid['unsupported']}")
                raw2 = self.llm.chat(SYSTEM, f"{context}\n\nQuestion: {question}\n"
                                     f"Your previous answer was inconsistent ({why}). "
                                     "Answer again using only the result and the notes.\nAnswer:").strip()
                if not raw2.upper().startswith("INSUFFISANT"):
                    raw, fid = raw2, fidelity(raw2, context + "\n" + question, expected)
                fid["retried"] = True
            return Agent3Result(question, "repondu", raw, exec_status=st,
                                notes=[asdict(n) for n in notes], fidelity=fid,
                                seconds=time.time() - t0)
        except Exception as e:     # panne Wikidata/LLM : visible, jamais remplacée par du texte
            return Agent3Result(question, "erreur", "", exec_status=st,
                                error=f"{type(e).__name__}: {e}", seconds=time.time() - t0)
