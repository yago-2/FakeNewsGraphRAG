"""Checkpoint für Config 1 ohne Ollama: Datum, Parser, Prompt und Messgrössen.
Aufruf: python config_1/test_config1.py"""
from config1 import (datum_ausgeschrieben, parse_label_c1, build_reasoning_prompt_c1,
                     build_extraction_prompt_c1, macro_f1_ohne_conflict, f1_pro_label,
                     mcnemar_exakt, fingerabdruck, prompt_vorlage)
from gewichtung import macro_f1, SUPP, REFU, NEE, CONF


def pruefe(name, ergebnis, erwartet):
    print(f"{'OK    ' if ergebnis == erwartet else 'FEHLER'} {name}: {ergebnis!r}")


print("--- Datum")
pruefe("9-10-2020", datum_ausgeschrieben("9-10-2020"), "9. Oktober 2020")
pruefe("31-12-2019", datum_ausgeschrieben("31-12-2019"), "31. Dezember 2019")
pruefe("1-3-2020", datum_ausgeschrieben("1-3-2020"), "1. Maerz 2020")

print("--- Parser (die letzten zwei geben zusätzlich eine Warnung)")
pruefe("'supported'", parse_label_c1("supported"), (SUPP, False))
pruefe("'Refuted.'", parse_label_c1("Refuted."), (REFU, False))
pruefe("'**conflicting**'", parse_label_c1("**conflicting**"), (CONF, False))
pruefe("'unknown'", parse_label_c1("unknown"), (NEE, False))
pruefe("'neutral'", parse_label_c1("neutral"), (NEE, True))
pruefe("''", parse_label_c1(""), (NEE, True))

print("--- Messgrössen")
gold = [SUPP, SUPP, REFU, REFU, NEE, CONF]
vorh = [SUPP, REFU, REFU, REFU, CONF, CONF]
f1 = f1_pro_label(gold, vorh)
pruefe("Macro-F1 = Mittel der vier F1", round(sum(f1.values()) / 4, 6), round(macro_f1(gold, vorh), 6))
pruefe("Macro-F1 ohne Conflict", round(macro_f1_ohne_conflict(gold, vorh), 6),
       round((f1[SUPP] + f1[REFU] + f1[NEE]) / 3, 6))
pruefe("McNemar 5 gegen 5", mcnemar_exakt(5, 5), 1.0)
pruefe("McNemar 0 gegen 6", round(mcnemar_exakt(0, 6), 6), round(2 / 2 ** 6, 6))
pruefe("McNemar 40 gegen 8 unter 0.001", mcnemar_exakt(40, 8) < 0.001, True)

print("--- Prompt für ein Beispiel")
print(build_reasoning_prompt_c1("In a letter to Steve Jobs, Sean Connery refused to appear "
                                "in an apple commercial.", datum_ausgeschrieben("31-10-2020")))
print()
print(build_extraction_prompt_c1("<Begruendung von llama>"))
print(f"\nFingerabdruck des aktuellen Prompts: {fingerabdruck(prompt_vorlage())}")
