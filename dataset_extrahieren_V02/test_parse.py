from transform_averitec import parse_stance
for r in ["supports", "Refutes.", "**neutral**", "Ich weiss nicht"]:
    print(f"{r!r:18} -> {parse_stance(r)}")