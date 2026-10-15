from transform_averitec import unwrap_archive_url, get_domain

tests = [
    "https://web.archive.org/web/20201129141238/https://scoopertino.com/exposed-the-imac-disaster-that-almost-was/",
    "https://web.archive.org/web/20210304085240mp_/https://www.foxnews.com/politics/fbi-purported-hunter-biden-laptop-sources",
    "https://web.archive.org/web/20200408020723/https:/blacklivesmatter.com/what-we-believe/",
    "https://web.archive.org/web/20201006135825/https://web.archive.org/web/20200917123421/https://www.facebook.com/photo.php?fbid=10220383578352003",
    "https://www.cdc.gov/flu/about/burden/2019-2020.html",
]
for u in tests:
    print(get_domain(unwrap_archive_url(u)))