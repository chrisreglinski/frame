## Kontrakt danych
06_docs/data.yaml to jedyne źródło prawdy o tabelach i kolumnach.

- Buildery implementują kontrakt, nie odwrotnie — kod produkuje
  dokładnie te kolumny, które definiuje data.yaml.
- Jeśli implementacja wymaga czegoś niespójnego z kontraktem
  (brakująca kolumna, inny typ, nowy wymiar) — NIE zmieniaj
  data.yaml samodzielnie i nie obchodź kontraktu w kodzie.
  Zatrzymaj się i przedyskutuj ze mną modyfikację kontraktu.
- Po uzgodnieniu: najpierw aktualizacja data.yaml, potem kod.