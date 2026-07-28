# tearsheet_spec

Specyfikacja jednostronicowego raportu modelu (tearsheet). Wielokrotnego użytku —
opisuje układ dla **dowolnego** modelu betowego w tym frameworku, nie zamraża
konkretnego modelu.

## Kontekst dla zimnej sesji

Powierzchnie i pojęcia, do których odwołują się bullety niżej:

- **Framework**: `02_src/evaluation/model_check.py` — `season_folds` (LOSO),
  `expanding_folds` (walk-forward), 3 bramki (LOSO ≥3/4 sezonów dodatnie,
  drop-best-league, walk-forward floor), `portfolio_roi` (stawkowanie implied).
- **Grupy ABT**: `major` = top-5 (dane deweloperskie). `other` = NL/PT
  (holdout OOD, oceniany raz po zamrożeniu modelu). `minor` = drugie ligi top-5.
- **Okno dev**: sezony **2223–2425** (3 sezony) — źródło wyboru bufora i ROI do
  stakingu. **2526** to najnowszy sezon — sekcja zgłębiająca, będąca zarazem
  najnowszym walk-forward (train 2223–2425 → 2526).
- **Filtr protokołu**: `hmt_game_number>8 & awt_game_number>8`.
- **Wybór bufora** (deterministyczny, bez oka): na siatce buforów licz profit;
  `thr = 0.9*profit.max()`, `band = bufory[profit>=thr]`;
  `left=band.min()`, `peak=bufory[profit.argmax()]`, `right=band.max()`.
- **ROI_dev**: pooled OOF ROI na 2223–2425 przy buforze `peak`.
- **Staking (bankroll)**: half Kelly z **bieżącego** bankrolla (compounding),
  start 100. Prawdopodobieństwo do Kelly = `implied_mecz × (1 + ROI_dev)` —
  **nie** modelowe `p`. Selekcja betów wciąż z modelu (`p > implied + buffer`).
  Stały mnożnik `(1+ROI_dev)` dla każdego betu (zakłada stały % edge — v1).
- **Reliability**: kubełkuj każdy predyktor osobno po jego własnym `p`
  (`mean p` vs `mean Y`); dwie niezależne krzywe (model, rynek).

## Bullety raportu

1. **nagłówek** — definicja modelu + liczby kluczowe (na walk-forward top-5;
   konkretna zawartość liczb ustalona później).

2. **dochód od bufora** — pooled OOF 2223–2425; zaznacz left/peak/right (0.9·max).
   Obok tabele ROI:
   - a) per sezon + ROI całości bez najlepszego sezonu
   - b) per liga + ROI całości bez najlepszej ligi

   Sens: krzywa z pikiem = model separuje dobre bety od złych; tabele = wynik nie
   stoi na jednym sezonie/lidze.

3. **reliability** — dwie krzywe (model + rynek). Lewo: 3 sezony, wszystkie mecze.
   Prawo: mecze bet (bufor peak). Mówi: czy `p` trafne → czy przed Kellym
   rekalibrować.

4. **sekcja 2526 — bankroll** — krzywa bankrolla zakład po zakładzie
   (chronologicznie), start 100, half Kelly wg `implied×(1+ROI_dev)` z bieżącego
   bankrolla; zacieniony drawdown; finalny bankroll obok.

5. **sekcja 2526 — 3 confusion matrix obok siebie** (bet = klasa pozytywna):
   1. zwykła, liczność meczów (bet → TP/FP, no-bet → TN/FN)
   2. profit per kafelek, staking płaski względem 100 (no-bet=0, TP=+, FP=−)
   3. profit per kafelek, staking z pełnego bankrolla (chronologicznie)

6. **sekcja 2526 — statystyki**:
   - wiersz 1: n meczów sezonu, n betów (+% całości), trafione (+% betów),
     nietrafione (+% betów)
   - wiersz 2: % trafionych, spodziewany % (mean implied), p-value na tych
     wartościach (dwumianowy: trafienia vs Σ implied)

7. **listing betów 2526** (opcjonalna, od nowej strony) — chronologicznie:
   home, away, odds, implied, model_p, stake/100, stake/bankroll, gole H, gole A,
   result (HDA), profit/100, profit/bankroll, bankroll skumulowany.
