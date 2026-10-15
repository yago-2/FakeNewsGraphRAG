from transform_averitec import get_stance_two_step

faelle = [
    ("Amy Coney Barrett was confirmed as US Supreme Court Justice on October 26, 2020",
     "Is Amy Coney Barrett confirmed as supreme Court justice ?", "Yes"),      # erwartet: supports
    ("Hunter Biden, son of US President Joe Biden has died.",
     "Is Hunter Biden dead?", "No"),                                           # erwartet: refutes
    ("Hunter Biden, son of US President Joe Biden has died.",
     "When was this claim made?", "25/10/2020"),                               # erwartet: neutral
]
for claim, frage, antwort in faelle:
    print(get_stance_two_step(claim, frage, antwort))