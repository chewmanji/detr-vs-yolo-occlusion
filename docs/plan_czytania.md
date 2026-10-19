# Plan czytania

## Etap 1: Podstawy

1. **You Only Look Once: Unified, Real-Time Object Detection** (`Przegląd/`)
   Pierwszy jednoetapowy detektor: obraz dzielony na siatkę, z której jedna sieć przewiduje boxy i klasy w jednym przebiegu. Wprowadza podstawowe pojęcia detekcji (IoU, confidence, NMS, mAP).

2. **Attention Is All You Need** (opcjonalnie): https://arxiv.org/abs/1706.03762
   Wprowadza architekturę Transformer opartą wyłącznie na mechanizmie attention. Fundament wszystkich dalszych prac o ViT i DETR.

3. **An Image is Worth 16x16 Words (ViT)**: https://arxiv.org/abs/2010.11929
   Pokazuje, że czysty Transformer działający na obrazie pociętym na łatki (patches) dorównuje CNN w klasyfikacji. Wprowadza pojęcia patch embedding i class token.

4. **End-to-End Object Detection with Transformers (DETR)** (`Przegląd/`)
   Traktuje detekcję jako przewidywanie zbioru obiektów: object queries i dopasowanie węgierskie zastępują anchory i NMS. Działa, ale trenuje się wolno i słabo radzi sobie z małymi obiektami.

5. **Deformable DETR**: https://arxiv.org/abs/2010.04159
   Zastępuje pełne attention attention typu deformable, które patrzy tylko na kilka punktów wokół referencji. Dzięki temu trening jest ok. 10x szybszy, a małe obiekty wykrywane są lepiej. Na tym mechanizmie opierają się nowsze DETR-y.

## Etap 2: Problem okluzji

6. **Occlusion Handling in Generic Object Detection: A Review** (`Przegląd/`)
   Przegląd rodzajów okluzji, zbiorów danych i metod radzenia sobie z nią w detektorach. Wyznacza ramy problemu badanego w projekcie.

7. **Intriguing Properties of Vision Transformers** (`Przegląd/`)
   Bada odporność ViT-ów m.in. na zasłonięcie dużej części obrazu (usuwanie łatek). Pokazuje, że ViT radzą sobie z tym znacznie lepiej niż CNN.

## Etap 3: Porównywane modele

8. **Ultralytics YOLO evolution** (`Przegląd/`)
   Przegląd kolejnych wersji YOLO od Ultralytics (v5, v8, 11 i nowsze): zmiany w architekturze, głowicach i treningu.

9. **Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models** (`Przegląd/`)
   Najnowsze YOLO działające end-to-end, bez NMS, czyli z ideą przejętą z DETR. Przy tym zachowuje szybkość typową dla rodziny YOLO.

10. **DINOv2**: https://arxiv.org/abs/2304.07193
    Backbone ViT trenowany samonadzorowo na dużym, oczyszczonym zbiorze danych. Daje uniwersalne cechy wizualne i jest backbone'em RF-DETR.

11. **LW-DETR**: https://arxiv.org/abs/2406.03459
    Lekki DETR czasu rzeczywistego (ViT + deformable decoder), który dorównuje YOLO szybkością. Bezpośredni poprzednik RF-DETR.

12. **RF-DETR: Neural Architecture Search for Real-Time Detection Transformers** (`Przegląd/`)
    Łączy LW-DETR z backbone'em DINOv2 i używa NAS do wyboru kompromisu między szybkością a dokładnością. Dobrze przenosi się na nowe zbiory danych.

## Etap 4: Augmentacja i generowanie okluzji

13. **Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection** (`Przegląd/`)
    Generuje dane treningowe przez wycinanie obiektów i wklejanie ich na losowe tła. Zwraca uwagę na artefakty krawędzi i sposoby ich maskowania (blending).

14. **Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation** (`Przegląd/`)
    Pokazuje, że losowe wklejanie obiektów między obrazami, bez żadnego dopasowania kontekstu, wyraźnie poprawia wyniki. To prosty sposób na sztuczne tworzenie okluzji.
