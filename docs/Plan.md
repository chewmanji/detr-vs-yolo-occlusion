# Plan realizacji projektu

Szczegółowy plan prac na podstawie [Temat.md](Temat.md), który pozostaje źródłem prawdy co do zakresu, pytań badawczych i kamieni milowych. Ten plik rozpisuje je na zadania, decyzje metodyczne i kryteria ukończenia etapów. Pola wyboru służą do śledzenia postępu.

**Stan na 03.11.2026:** gotowe są opis tematu i lista literatury, a dane D2S i D2S Amodal leżą w `data/`; nie ma jeszcze kodu. M1 („pierwszy trening") wypada dziś, więc realnie przesuwa się na 10.11. Opóźnienie odrabiam, uruchamiając treningi R1 już w fazie generatora (Etap 4a), bo nie potrzebują one danych syntetycznych. Kamienie M2–M5 zostają bez zmian.

Decyzje, które doprecyzowują lub zmieniają Temat.md, są zebrane w sekcji 13; wszystkie przyjęto 03.11.

## 1. Kluczowe decyzje metodyczne

1. **Dokładna okluzja w teście A (sprawdzone na danych 03.11).** D2S Amodal (Follmann i in., WACV 2019) ma maski amodalne i widoczne dla wszystkich 3600 obrazów walidacji D2S. Obrazy i ich id są te same co w D2S; 1479 obrazów dopełniono zerami (padding) z różnych stron, a przesunięcie wynika z porównania ramek. Maski widoczne z D2S Amodal pokrywają się z maskami D2S (IoU ≥ 0,93; poniżej 0,99 tylko 35 z 15 654 obiektów). Brakuje adnotacji amodalnej jednego obiektu (klasa 33, scena 286, 29 obrazów); pomijam go w analizie per przedział. Okluzję liczę jako `1 − |maska widoczna| / |maska amodalna w kadrze|`, osobno od ucięcia krawędzią kadru (2997 obiektów ma ponad 1% maski amodalnej poza kadrem). Przybliżenie z Temat.md jest niepotrzebne.

   Rozkład okluzji w teście A (obiekty nieucięte):

   | Okluzja | 0%   | 0–20% | 20–40% | 40–60% | 60–80% | 80–100% |
   | ------- | ---- | ----- | ------ | ------ | ------ | ------- |
   | Obiekty | 8553 | 2976  | 831    | 228    | 51     | 18      |
   | Sceny   | 108  | 55    | 25     | 10     | 2      | 1       |

   Każda scena to te same obiekty na 30 zdjęciach, więc o wiarygodności przedziału decyduje liczba scen. Test A wystarcza do 0–40%, słabo do 40–60%; o przedziale 60–80% rozstrzyga test B.

2. **Scena jako jednostka danych.** Każda scena D2S to 30 zdjęć (10 obrotów co 36° × 3 oświetlenia), czyli prawie duplikaty. Numer obrazu koduje scenę, obrót i oświetlenie: `id = scena × 100 + k`, obrót = `k % 10`, oświetlenie = `k // 10` (trening: 146 scen, walidacja: 120). Zdjęcia jednej sceny nigdy nie trafiają jednocześnie do treningu i do testów, a przedziały ufności w teście A liczę bootstrapem po 120 scenach, nie po 3600 obrazach.
3. **Walidacja z obrotów, tylko do monitorowania.** Trening D2S ma tylko 146 scen: 12 klas ma jedną scenę, a 23 klasy dwie. Wydzielenie całych scen usunęłoby więc niektóre klasy z treningu. Do walidacji idą 2 z 10 obrotów każdej sceny treningowej (np. 4 i 9, wszystkie oświetlenia). Taka walidacja jest optymistyczna, więc służy tylko do śledzenia zbieżności i łapania błędów. Do testów biorę wagi końcowe (last/EMA) po ustalonej liczbie epok, bez wybierania najlepszego checkpointu; reguła jest ta sama dla wszystkich modeli.
4. **Wycinki do testu B z walidacji D2S.** Źródłem są obiekty walidacji D2S bez okluzji i nieucięte krawędzią kadru (według D2S Amodal). Spełnia to warunek „obrazy nieużyte w treningu" i nie zabiera i tak skromnych scen treningowych. Wycinki do R2 pochodzą tylko z części treningowej (bez obrotów walidacyjnych).
5. **Jedna konwersja, dwa zestawy etykiet.** Konwersja D2S → YOLO daje jeden zestaw obrazów i dwa zestawy etykiet: `c60` (60 klas) i `c1` (klasa „produkt"). Układ `train/`, `valid/`, `test/` z podkatalogami `images/` i `labels/` plus `data.yaml` czytają oba pakiety (RF-DETR wymaga dokładnie takiego). Tryb class-agnostic to osobne etykiety, a nie `single_cls` z Ultralytics, bo RF-DETR nie ma odpowiednika, a modele mają dostać identyczne dane.
6. **Jeden ewaluator.** Każdy model zapisuje surowe predykcje w formacie wyników COCO (JSON), a wszystkie metryki liczy jeden skrypt (pycocotools i własny recall per przedział). Nie używam wbudowanych `val()`, bo pakiety liczą mAP inaczej. GPU służy tylko do treningu i predykcji, analiza działa lokalnie.
7. **YOLO26 z jawnym `nms=False`.** Według dokumentacji Ultralytics `predict()` i `val()` domyślnie używają głowicy one-to-many z NMS. Te same wagi z `nms=True` dają bez dodatkowego treningu czwarty wariant, „YOLO26-S o2m+NMS", czyli porównanie z NMS i bez na jednym modelu.
8. **R1 i R2 z tą samą liczbą iteracji.** R2 ma ok. 2× więcej obrazów, więc przy tej samej liczbie epok trenowałby dwa razy dłużej. Dla R2 skaluję liczbę epok tak, by łączna liczba obejrzanych obrazów była taka sama. Różnica R2 − R1 wynika wtedy z treści danych, a nie z długości treningu.
9. **Tło scen syntetycznych = tło treningu.** R2 i test B składam na pustym talerzu obrotowym, z wycinkami w tym samym oświetleniu co tło. Test B mierzy wtedy samą okluzję. Test A dokłada zmianę tła i obiekty obce, czyli warunki realne.
10. **Przepis treningu zamrożony po pilocie.** Każdy model trenuję domyślnym przepisem swojego pakietu (augmentacje, optymalizator), z wejściem 640 px i liczbą epok z pilota. Po pilocie zamrażam wersje pakietów, podział danych, rozdzielczość, liczbę epok i regułę checkpointu (sekcja 10).

## 2. Harmonogram

| Okres       | Etap                                              | Kamień                    |
| ----------- | ------------------------------------------------- | ------------------------- |
| 03.11–09.11 | 1. Dane i konwersja, 2. Pilot                     | M1: 10.11 (planowo 03.11) |
| 10.11–24.11 | 3. Generator i test B, równolegle 4a. Treningi R1 | M2: 24.11                 |
| 25.11–15.12 | 4b. Treningi R2, bieżąca ewaluacja                | M3: 15.12                 |
| 16.12–12.01 | 5. Analiza, drugie ziarno (przerwa 22.12–06.01)   | M4: 12.01                 |
| 13.01–01.02 | 6. Raport i slajdy                                | M5: prezentacja 02.02     |

## 3. Etap 1: dane i konwersja (03–09.11)

**Cel:** D2S w formacie YOLO na Kaggle oraz GT testu A z okluzją każdego obiektu.

- [x] Pobrać D2S i D2S Amodal do `data/` ([źródło](https://www.mvtec.com/company/research/datasets/mvtec-d2s/downloads), licencja CC BY-NC-SA 4.0). Obrazy D2S Amodal (z paddingiem) nie są potrzebne, bo adnotacje przelicza się na oryginalne obrazy D2S.
- [x] Inspekcja danych:
  - [x] pliki: `D2S_training.json` (4380 obrazów / 6900 obiektów), `D2S_validation.json` (3600 / 15 683), obrazy 1920×1440 px, segmentacje jako RLE;
  - [x] scena, obrót i oświetlenie z id obrazu (sekcja 1, punkt 2);
  - [x] id kategorii: 1–60 bez luk we wszystkich plikach, wszystkie 60 klas występują w treningu i walidacji; indeks klasy YOLO = `id − 1`. Nazwy brać z `D2S_training.json` i sortować po id, bo lista kategorii w plikach nie jest posortowana. D2S Amodal ma te same id, ale 9 klas pod niemieckimi nazwami (np. `rocket` → `rucola`);
  - [x] obiekty obce (clutter) nie mają adnotacji (sprawdzone wzrokowo na podzbiorze `_clutter`: 18 scen, 540 obrazów). Ich wykrycie liczy się jako fałszywy alarm, więc wpływa na precyzję i punkt pracy, a nie na recall. D2S Amodal liczy zasłonięcie przez nie jako okluzję (np. butelki częściowo zasłonięte klockiem: 19–21%), więc pula wycinków testu B (okluzja 0) sama pomija takie obiekty.
- [x] Zgodność D2S Amodal z walidacją D2S (sekcja 1, punkt 1).
- [x] Środowisko przez uv: Python 3.11, `pyproject.toml` + `uv.lock` (`ultralytics` 8.4.201, `rfdetr` 1.11.2, `pycocotools`; lokalnie torch 2.15 w wersji CPU). `uv sync` odtwarza środowisko.
- [x] `scripts/prepare_d2s.py` (`uv run python scripts/prepare_d2s.py`, ok. 5 min, wynik w `data/d2s_yolo/`, 1,3 GB):
  - podział treningu: obroty walidacyjne → `valid/`, reszta → `train/`;
  - ramki COCO `[x, y, w, h]` → YOLO (znormalizowane `cx cy w h`), id kategorii → 0–59, nazwy klas w `data.yaml`;
  - etykiety `c60` i `c1` (dwa pliki `data.yaml`);
  - zmniejszenie obrazów do 1024 px dłuższego boku: Kaggle ma tylko kilka rdzeni CPU, a dekodowanie JPEG 1920×1440 spowalnia ładowanie danych; etykiety YOLO są znormalizowane, więc się nie zmieniają;
  - walidacja D2S → `test/` (test A) oraz `test_a_gt.json` (COCO GT w tej samej rozdzielczości) z polami obiektu `occlusion` i `truncated` (ponad 1% maski amodalnej poza kadrem; oba `null` dla obiektu bez adnotacji amodalnej) oraz `scene_id` obrazu;
  - asercje: liczby obrazów i etykiet, ramka po konwersji tam i z powrotem w granicy 1 px, każda klasa obecna w `train/`.
- [x] Kontrola: loadery Ultralytics i RF-DETR wczytują wszystkie podziały w obu trybach z tymi samymi liczbami (train 3504 obrazy / 5520 obiektów, valid 876 / 1380, test 3600 / 15 683). Etykiety narysowane z powrotem na 4 obrazach (train, valid, 2× test) pasują do obiektów.
- [x] Rozkład okluzji testu A (sekcja 1, punkt 1).
- [x] Prywatny dataset na Kaggle (`kaggle datasets create -p 'data\d2s_yolo' --dir-mode zip`) z `data/d2s_yolo/`, bez plików `labels.cache` (Ultralytics tworzy je przy lokalnym wczytaniu i zapisuje w nich lokalne ścieżki). Obrazy `c1` to lokalnie twarde dowiązania do `c60`, więc upload ma ok. 2,6 GB. CLI: `uv tool run --env-file .env --with six kaggle ...`; token jako `KAGGLE_API_TOKEN` w `.env`, który jest w `.gitignore` (pakiet `kaggle` 2.2.5 nie deklaruje zależności `six`). Na Windows ścieżka `-p` musi mieć `\`, nie `/`: CLI 2.2.5 zamienia w nazwie pliku postępu tylko separator systemowy i upload pojedynczych plików się wywraca. Repo jest publiczne, więc notatnik klonuje je zwykłym `git clone`.

**Gotowe, gdy:** oba pakiety wczytują `data.yaml` (sprawdzone lokalnie) i przechodzą 1 epokę bez błędu (w pilocie na Kaggle).

## 4. Etap 2: pilot i M1 (do 10.11)

**Cel:** potok trening → predykcja → ewaluacja działa end-to-end i znam koszt GPU.

- [ ] `scripts/train.py`: jeden punkt wejścia `--model {yolo11s,yolo26s,rfdetr-s} --regime {R1,R2} --labels {c60,c1} --seed N`; checkpoint co epokę i wznawianie (`resume`), bo sesja Kaggle trwa najwyżej ok. 12 h.
- [ ] `scripts/predict.py`: predykcje do COCO JSON, próg pewności 0,001, najwyżej 100 detekcji na obraz; YOLO26 w dwóch wariantach (`nms=False` i `nms=True`).
- [ ] `scripts/evaluate.py` w wersji minimalnej: mAP przez pycocotools i recall per przedział okluzji.
- [ ] Notatnik Kaggle: instalacja `ultralytics` i `rfdetr[train]` (bez dodatku `train` RF-DETR nie trenuje) w wersjach z `uv.lock`, bez nadpisywania torcha z obrazu Kaggle (wersję torcha zapisać w protokole); akcelerator T4 (sprawdzić, czy P100 jest jeszcze wspierany przez aktualny PyTorch), uruchamianie w tle przez „Save & Run All".
- [ ] Pomiar: każdy z 3 modeli, 2–3 epoki na R1/c60 → czas epoki, pamięć GPU, czy mAP walidacji rośnie.
- [ ] Rozdzielczość RF-DETR-S: sprawdzić, czy 640 px spełnia wymóg podzielności (natywnie 512 px); jeśli nie, najbliższa dozwolona wartość.
- [ ] Decyzja o drugim ziarnie (sekcja 9).

**Gotowe (M1), gdy:** jeden model przeszedł cały potok, a czasy epok trzech modeli i decyzja o drugim ziarnie są wpisane do sekcji 9.

## 5. Etap 3: generator scen i test B (10–24.11)

**Cel:** zbiory R2 i test B, w których okluzja każdego obiektu jest znana z konstrukcji. Generator i sposób pomiaru okluzji przechodzą później do pracy magisterskiej, więc mają krótką dokumentację.

- [ ] **Tła:** pusty talerz dla każdego oświetlenia jako mediana pikseli po obrazach treningowych z tym oświetleniem (obiekty stoją w różnych miejscach, więc mediana je usuwa). Jeśli talerz wyraźnie zmienia wygląd z obrotem, osobna mediana dla każdej pary obrót × oświetlenie. Kontrola wzrokowa.
- [ ] **Pula wycinków:** obiekty z maską (pycocotools), które nie stykają się z innymi maskami ani z krawędzią kadru; zapamiętane oświetlenie źródła. Pula R2 z `train/`, pula testu B z walidacji D2S (okluzja 0 według D2S Amodal).
- [ ] **Składanie sceny:** losowa liczba obiektów (np. 4–15), środek obiektu na talerzu, losowy obrót w płaszczyźnie, wycinki w oświetleniu tła, wklejanie po kolei z zapamiętaną kolejnością warstw. Wklejanie twarde po masce (maski D2S nie mają marginesów, a autorzy D2S uzyskali tak realistyczne obrazy); ewentualnie wygładzenie krawędzi o 1 px.
- [ ] **Okluzja:** maska widoczna obiektu = maska pełna minus suma masek wklejonych później; `okluzja = 1 − |widoczna| / |pełna|`.
- [ ] **Etykiety:** ramka z maski widocznej, tak jak w D2S, na którym modele się uczą i są oceniane. R2: odrzucam położenia, po których widoczna część obiektu ma mniej niż 5000 px (reguła z augmentacji D2S). Test B: obiekty zasłonięte w ponad 80% oznaczam jako ignorowane (`iscrowd=1`); nie wchodzą do recall, a ich wykrycie nie jest fałszywym alarmem.
- [ ] **Balans testu B:** generuję sceny, aż każdy przedział (0–20, 20–40, 40–60, 60–80%) ma co najmniej 1000 obiektów. Jeśli wysokie przedziały zapełniają się za wolno, tryb celowany: drugi obiekt wklejany tak, by zakrył zadany ułamek pierwszego. Do RQ2 dokładam pary o dużym IoU ramek (produkty ułożone jeden na drugim).
- [ ] **Rozmiary:** R2 = obrazy R1 (ok. 3500) + ok. 3500 scen syntetycznych; test B ok. 1000–2000 obrazów.
- [ ] Testy z asercjami: dwa prostokąty o znanym pokryciu dają oczekiwaną okluzję; okluzja w [0, 1]; ramki w granicach obrazu.
- [ ] Kontrola wzrokowa: siatka scen z obrysami i procentem okluzji.
- [ ] Wysłanie R2 i testu B na Kaggle (nowa wersja datasetu).
- [ ] Szkic rozdziału „Metody" (dane, podział, generator, pomiar okluzji), póki decyzje są świeże.

**Gotowe (M2), gdy:** R2 i test B są na Kaggle, a rozkłady okluzji testów A i B są na wykresie.

## 6. Etap 4: treningi (4a: 10–24.11, 4b: 25.11–15.12)

Runy R1 nie potrzebują generatora, więc ruszają zaraz po M1, równolegle z Etapem 3. Pierwsze runy R1/c60 (YOLO11-S i RF-DETR-S) są zarazem pilotem zbieżności: trenuję do plateau mAP walidacji, ustalam liczbę epok (YOLO26-S dostaje liczbę epok YOLO11-S) i wpisuję ją do protokołu przed kolejnymi runami.

Nazwa runu: `{model}_{reżim}_{etykiety}_s{ziarno}`, np. `rfdetrs_R2_c60_s1`.

| Grupa         | Modele                        | Runy | Okno                 | Priorytet |
| ------------- | ----------------------------- | ---- | -------------------- | --------- |
| R1 · c60 · s0 | YOLO11-S, YOLO26-S, RF-DETR-S | 3    | 10–24.11             | 1         |
| R1 · c1 · s0  | jw.                           | 3    | 10–24.11             | 2         |
| R2 · c60 · s0 | jw.                           | 3    | 25.11–15.12          | 1         |
| R2 · c1 · s0  | jw.                           | 3    | 25.11–15.12          | 2         |
| R1 · c60 · s1 | jw.                           | 3    | wolny limit od 10.11 | 3         |
| R2 · c60 · s1 | jw.                           | 3    | wolny limit do 21.12 | 3         |

Bez treningu, na wagach z tabeli: YOLO26-S z `nms=True` (o2m+NMS) oraz przemiatanie progu IoU NMS (0,5 / 0,6 / 0,7 / 0,8) dla YOLO11-S i YOLO26-S o2m.

Ziarno zmienia tylko losowość treningu (Ultralytics: `seed`, `deterministic=True`; w RF-DETR sprawdzić parametr). Dane, w tym zbiór R2, są te same dla obu ziaren.

Po każdym runie:

- [ ] wagi końcowe i predykcje na testach A i B (B, gdy gotowy) zapisane jako wynik notatnika Kaggle, poza gitem;
- [ ] wiersz w `results/runs.csv`: id runu, wersje pakietów, epoki, godziny GPU, mAP walidacji, gdzie leżą wagi i predykcje;
- [ ] szybka ewaluacja lokalna, żeby błąd wyszedł od razu, a nie po 12 runach.

**Gotowe (M3), gdy:** 12 runów z ziarnem 0 ma predykcje na obu testach.

## 7. Etap 5: ewaluacja i analiza (16.12–12.01)

### Definicje metryk

- **Dopasowanie:** zachłanne według malejącej pewności, IoU ≥ 0,5, każdy obiekt GT najwyżej raz. Regiony ignorowane nie dają ani TP, ani FP.
- **Recall_any** (metryka główna): obiekt trafiony ramką dowolnej klasy, czyli „produkt znaleziony". **Recall_cls**: trafiony ramką właściwej klasy (tylko modele c60).
- **Punkt pracy:** dla każdego modelu próg pewności, przy którym precyzja (dowolna klasa) na całym teście A wynosi 0,90; ten sam próg stosuję w teście B. Pewności DETR i YOLO mają różne skale, więc porównuję modele przy tej samej liczbie fałszywych alarmów. Uzupełnienie bez progu: recall przy progu 0,001.
- **mAP@50:95, mAP@50:** pycocotools, `maxDets=100`.
- **Udział pomyłek klasy:** `(Recall_any − Recall_cls) / Recall_any` w każdym przedziale.
- **Przedziały okluzji:** 0–20 / 20–40 / 40–60 / 60–80%; w teście A dodatkowo 80–100% informacyjnie, a obiekty `truncated` osobno. Druga oś do RQ2: największe IoU ramki obiektu z ramką innego obiektu GT (0–0,3 / 0,3–0,5 / 0,5–0,7 / > 0,7). To od niej, a nie od procentu okluzji, zależy, czy NMS może usunąć obiekt.
- **Niepewność:** 95% CI bootstrapem (1000 losowań) po scenach w teście A i po obrazach w teście B. Dla c60 dodatkowo rozrzut między ziarnami; różnice mniejsze od niego opisuję jako nierozstrzygnięte.
- **Latencja:** lokalny CPU, batch 1, 640 px, stała liczba wątków, 20 przebiegów rozgrzewki i 200 obrazów z testu A, mediana ms/obraz z post-processingiem (w tym NMS). W tabeli także liczba parametrów i FLOPs: RF-DETR-S ma 32,1 M parametrów, a modele YOLO-S ok. 10 M, więc porównanie dotyczy tej samej klasy szybkości, nie rozmiaru.

### Analiza per pytanie badawcze

- [ ] **RQ1:** krzywe Recall_any względem przedziału okluzji dla wszystkich modeli, osobno dla testów A i B. Tempo spadku: regresja logistyczna `wykryty ~ okluzja` dla każdego modelu; nachylenie i jego CI bootstrapem po scenach; różnice nachyleń między modelami.
- [ ] **RQ2:** YOLO11-S vs YOLO26-S e2e vs YOLO26-S o2m+NMS (te same wagi) vs RF-DETR-S na osi IoU ramek, plus przemiatanie progu NMS. Porównanie c60 i c1: w c1 NMS tłumi każdą nakładającą się ramkę, w c60 tylko ramki tej samej klasy (domyślnie `agnostic_nms=False`), więc efekt NMS powinien być w c1 silniejszy.
- [ ] **RQ3:** zysk R2 − R1 w każdym przedziale i dla każdego modelu. Zysk w przedziale 0–20% to efekt ogólny (gęstość sceny, kontekst); o okluzji świadczy dopiero jego wzrost w wyższych przedziałach. Zależność od architektury: interakcja model × reżim.
- [ ] **RQ4:** dla modeli c60 rozbicie obiektów w każdym przedziale na: nieznalezione / znalezione z błędną klasą / poprawne. Recall_any modelu c60 vs recall modelu c1: czy głowica 60-klasowa pogarsza samą lokalizację. Najczęstsze pary pomylonych klas przy wysokiej okluzji.
- [ ] Wykresy i tabele: krzywe recall, rozbicie błędów, tabela mAP + latencja; przykłady jakościowe (te same obrazy dla wszystkich modeli).
- [ ] Runy z drugim ziarnem, jeśli zostały, do 21.12.

**Gotowe (M4), gdy:** każde RQ ma tabelę lub wykres z CI i dwa–trzy zdania wniosku.

## 8. Etap 6: raport i prezentacja (13.01–01.02)

- [ ] Raport po polsku: wstęp (SCO), przegląd literatury, metody (szkic z Etapu 3), wyniki per RQ, dyskusja, ograniczenia, wnioski, związek z pracą magisterską (wybór detektora-lokalizatora w trybie class-agnostic, generator, pomiar okluzji).
- [ ] Ograniczenia: sceny syntetyczne bez cieni i fizyki układania; jeden zbiór danych; zmiana tła między treningiem a testem A; 1–2 ziarna; różna liczba parametrów; różne domyślne augmentacje pakietów (mozaika Ultralytics sama tworzy obrazy z wieloma obiektami).
- [ ] Cytowania: D2S (ECCV 2018), D2S Amodal (WACV 2019), RF-DETR, YOLO26, YOLO11.
- [ ] Slajdy i próba prezentacji.
- [ ] README z instrukcją odtworzenia: dane → `prepare_d2s.py` → `generate.py` → `train.py` → `predict.py` → `evaluate.py`.

**Gotowe (M5):** prezentacja 02.02.2027.

## 9. Budżet GPU i decyzja M1

Kaggle daje ok. 30 h GPU tygodniowo (sprawdzić aktualny limit konta). Okno 10.11–21.12 to 6 tygodni, czyli ok. 180 h. Przy równej liczbie iteracji run R2 kosztuje tyle co R1. Każdy model ma 6 runów (4 z ziarnem 0, 2 z ziarnem 1), a predykcja to ok. 0,3 h na run.

| Model     | s / epokę (R1) | Epoki | h / run | Runy | h łącznie |
| --------- | -------------- | ----- | ------- | ---- | --------- |
| YOLO11-S  |                |       |         | 6    |           |
| YOLO26-S  |                |       |         | 6    |           |
| RF-DETR-S |                |       |         | 6    |           |

Reguła z Temat.md: jeśli 18 runów przekracza 80% dostępnych godzin, rezygnuję z drugiego ziarna. Kolejność runów z sekcji 6 zapewnia, że przy cięciach odpadają najpierw runy o najniższym priorytecie.

## 10. Protokół (wypełnić po pilocie, potem bez zmian)

| Pole                                        | Wartość            |
| ------------------------------------------- | ------------------ |
| Wersje `ultralytics` / `rfdetr` / `torch`   |                    |
| Obroty walidacyjne                          |                    |
| Rozdzielczość obrazów w datasecie / wejście | 1024 px / 640 px   |
| Epoki R1 (YOLO / RF-DETR)                   |                    |
| Epoki R2 (równa liczba iteracji)            |                    |
| Batch, akumulacja gradientu                 |                    |
| Checkpoint do testów                        | końcowy (last/EMA) |
| GPU Kaggle                                  |                    |
| Drugie ziarno                               | tak / nie          |

## 11. Struktura repozytorium

```
scripts/prepare_d2s.py   konwersja D2S → YOLO (c60, c1), podział, GT testu A z okluzją
scripts/generate.py      generator copy-paste: tła, wycinki, R2 i test B
scripts/train.py         trening dowolnego runu z macierzy
scripts/predict.py       predykcje → COCO JSON
scripts/evaluate.py      mAP, recall per przedział, pomyłki klas, bootstrap
scripts/latency.py       pomiar CPU
notebooks/kaggle.ipynb   cienka nakładka: instalacja, klon repo, train + predict
tests/                   asercje dla konwersji ramek i liczenia okluzji
results/                 runs.csv i tabele metryk (małe pliki); wagi i predykcje poza gitem
pyproject.toml, uv.lock  zależności i przypięte wersje (uv)
```

Dane (`data/`) są w `.gitignore`.

## 12. Ryzyka

| Ryzyko                                                                          | Skutek                             | Reakcja                                                                                                        |
| ------------------------------------------------------------------------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| M1 przesunięty na 10.11                                                         | mniej czasu na generator           | treningi R1 równolegle z Etapem 3; M2 bez zmian                                                                |
| Za mało godzin GPU                                                              | niepełna macierz                   | priorytety runów; rezygnacja z drugiego ziarna; Colab jako zapas                                               |
| D2S Amodal nie pasuje do obrazów D2S                                            | brak dokładnej okluzji w teście A  | przybliżenie z nakładania się masek i ramek (Temat.md)                                                         |
| Mało silnie zasłoniętych obiektów w teście A (40–60%: 10 scen, 60–80%: 2 sceny) | test A nie rozstrzyga powyżej 40%  | wnioski dla 40–80% z testu B; test A tylko potwierdza kierunek trendu                                          |
| Sceny syntetyczne odbiegają od realnych                                         | zysk R2 nie przenosi się na test A | to też wynik dla RQ3; zawsze raportuję oba testy                                                               |
| Generator nie zdąży na 25.11                                                    | brak danych R2                     | awaryjnie R2 z obrazów augmentowanych dołączonych do D2S; tylko do treningu, bo powstały z całego treningu D2S |
| Zmiany API pakietów                                                             | niespójne runy                     | przypięte wersje od M1                                                                                         |
| Przerwana sesja Kaggle                                                          | utrata postępu                     | checkpoint co epokę i `resume`                                                                                 |

## 13. Decyzje do potwierdzenia

| #   | Decyzja                                                       | Co zmienia w Temat.md                            | Status                                                              |
| --- | ------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------- |
| 1   | Okluzja testu A z D2S Amodal                                  | zastępuje przybliżenie z nakładania się obiektów | przyjęte, sprawdzone na danych; Temat.md i CLAUDE.md zaktualizowane |
| 2   | Walidacja z 2 obrotów każdej sceny, testy na wagach końcowych | doprecyzowuje „walidację wydzielam z treningu"   | przyjęte                                                            |
| 3   | Wycinki testu B z niezasłoniętych obiektów walidacji D2S      | doprecyzowuje „obrazy nieużyte w treningu"       | przyjęte                                                            |
| 4   | Recall przy precyzji 0,90 oraz recall bez progu               | doprecyzowuje metrykę główną                     | przyjęte                                                            |
| 5   | RF-DETR-S przy 640 px zamiast natywnych 512 px                | doprecyzowuje „ok. 640 px"                       | przyjęte; jeśli 640 niedozwolone, najbliższa dozwolona wartość      |
| 6   | R2 = R1 + tyle samo scen syntetycznych, równa liczba iteracji | doprecyzowuje reżim R2                           | przyjęte                                                            |
| 7   | Obrazy zmniejszone do 1024 px przed wysłaniem na Kaggle       | nowe                                             | przyjęte                                                            |
