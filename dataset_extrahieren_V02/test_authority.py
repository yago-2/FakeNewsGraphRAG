from transform_averitec import get_authority, WERT_PLATTFORM, WERT_BEHOERDE, WERT_WISSENSCHAFT

print(f"Kategorie-Werte: Plattform {WERT_PLATTFORM:.2f}, Behörde {WERT_BEHOERDE:.2f}, Wissenschaft {WERT_WISSENSCHAFT:.2f}")
for d in ["nytimes.com", "edition.cnn.com", "thelancet.com", "aljazeera.com",
          "beehive.govt.nz", "gov.uk", "ox.ac.uk", "twitter.com", "archive.ph"]:
    print(f"{d:16s} {get_authority(d)}")