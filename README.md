# Startrepo — elprisprediktion

Data engineering och Agila metoder (YH-02032), grupprojekt vecka 3–9.

Det här repot är utgångspunkten, inte lösningen. Ni får en färdigtränad
modell och en fungerande men medvetet ofullständig träningspipeline.
Allt runt omkring den är ert jobb — se projektbeskrivningen, avsnitt 3.

---

## Snabbstart

```bash
cp .env.example .env          # fyll i PRICE_AREA och TARGET
uv sync                       # eller: pip install -e .
uv run python model/train.py
```

Det ska skriva ut en utvärdering och lägga en ny `model/model.pkl`. Går det
igenom har ni en fungerande miljö.

Öppna sedan `model/utforskning.ipynb`. **Kör den innan ni börjar bygga** —
nästan varje beslut i sprint 1 har sin grund i något som syns där.

---

## Innehåll

| Fil | Vad det är |
|---|---|
| `model/train.py` | Baslinjens träningspipeline. Fungerar. Läs `LIMITATIONS` längst ner i filen. |
| `model/utforskning.ipynb` | Rundtur i datan, med problemen utpekade. |
| `model/model.pkl` | Färdigtränad basmodell (SE3, target `mean`). |
| `data/historik_elpris.csv.gz` | Prishistorik. **Syntetisk tills ni byter ut den** — se nedan. |
| `data/historik_vader.csv.gz` | Väderhistorik. Samma sak. |
| `data/fetch_historik.py` | Hämtar riktig data från elprisetjustnu.se och SMHI. |
| `data/generate_synthetic_historik.py` | Genererade platshållardatan. Ni behöver den inte. |
| `db/schema.sql` | Startschema för bronze-tabellerna i er egen Postgres. |
| `.env.example` | Mall för konfiguration och hemligheter. |

---

## ⚠️ Datan är syntetisk tills ni hämtar riktig

`data/historik_elpris.csv.gz` och `data/historik_vader.csv.gz` innehåller
**påhittade siffror**. De finns där för att repot ska gå att köra direkt
efter en klon, innan någon pratat med ett API.

Formen är identisk med källorna — samma kolumnnamn, samma separatorer, samma
tidskonventioner, samma upplösningsbyte, samma sorters hål — så kod som
skrivs mot dem fortsätter fungera när ni byter ut innehållet. Men siffrorna
går inte att dra slutsatser av.

Byt ut dem tidigt:

```bash
uv run python data/fetch_historik.py --from 2024-08-01 --to 2026-08-20
```

Det tar 25–40 minuter för alla fyra elområden (ett anrop per dygn och
område, med paus emellan — API:erna är gratis och drivs för allas skull).
Hämta bara ert eget område om ni har bråttom:

```bash
uv run python data/fetch_historik.py --areas SE3 --from 2025-01-01
```

**Stationskoder:** bara station 98230 (Stockholm-Observatoriekullen A) är
verifierad mot API:et. Övriga tre i tabellen är platshållare. Slå upp rätt
nyckel innan ni hämtar:

```bash
uv run python data/fetch_historik.py --find-station "Lulea"
```

---

## Vad som är avsiktligt trasigt

Ingen av följande är buggar. Alla finns i den riktiga datan också, och att
hantera dem är en stor del av vad lärandemål 15 bedömer.

1. **Ett dygn har inte 24 rader.** Sverige gick från timpris till kvartspris
   1 oktober 2025. Före: 24 punkter per dygn. Efter: 96. Kod som antar 24
   går sönder tyst mitt i er träningsdata.

2. **Sommartid.** Sista söndagen i mars har 23 timmar, sista i oktober 25.
   "Dela med 24" är fel två gånger om året.

3. **Två tidskonventioner.** Priser kommer i lokal svensk tid med explicit
   UTC-offset. SMHI kommer i UTC helt utan offsetmarkör. Slår man ihop dem
   naivt hamnar man en till två timmar fel — utan att något kraschar.

4. **Luckor i väderdatan.** Enstaka timmar saknas, och några flerdygns-
   avbrott. Interpolering är rimlig för det ena och tveksam för det andra.

5. **Kvalitetskoder.** SMHI märker varje värde G (kontrollerat) eller Y
   (misstänkt eller aggregerat). Ungefär en femtedel är Y. `train.py`
   behandlar dem lika. Är det rätt?

6. **Negativa priser och spikar.** Både under noll och tio gånger normalt
   förekommer på riktigt. Att klippa bort dem utan att tänka förstör modellen.

7. **Dubbletter.** Ominhämtning lämnar exakta dubbletter. De måste bort före
   aggregering, annars viktas en timme dubbelt.

---

## Informationsgränsen — läs det här

Det enskilt viktigaste i hela uppgiften.

När ni predicerar en dag D vet ni: priser observerade till och med D, väder
observerat till och med D, och en **prognos** för morgondagens väder. Ni vet
inte morgondagens priser — källan publicerar dem först runt 13:00 dag D.

`train.py` använder morgondagens *observerade* temperatur som stand-in för
morgondagens *prognos*. Det är försvarbart vid träning och osant vid drift,
eftersom ni i produktion har en prognos med fel i, inte facit. Modellen ser
därför bättre ut i utvärderingen än den presterar i verkligheten.

Att kvantifiera det gapet är ett bra användande av en sprint.

---

## Om målvariabeln `peak`

Har ert team `TARGET=peak` är ni på ett svårare problem än `mean`, och det är
meningen. Två saker att veta i förväg:

- **Definitionen byter innebörd 2025-10-01.** Före det datumet är toppen den
  dyraste *timmen*; efter är den den dyraste *kvarten*, som systematiskt
  ligger högre. Brytpunkten sitter mitt i träningsdatan. `train.py` gör
  ingenting åt det. Att resampla allt till timupplösning först är en lösning;
  att bara träna på data efter bytet är en annan; att strunta i det är ingen.

- **R² kan bli negativt** samtidigt som modellen slår den naiva baslinjen.
  Det är inte ett fel i koden. Dygnstoppen drivs av knapphetshändelser som
  inte går att förutsäga ur temperatur, så modellen predicerar nära ett
  medelvärde — vilket slår "imorgon blir som idag" men förklarar mindre
  varians än att bara gissa medelvärdet. Fundera på vad det säger om
  problemet, och på om MAE eller R² är rätt mått för det ni faktiskt vill.

---

## Databasen

`db/schema.sql` skapar de två bronze-tabellerna. Kör den en gång mot er egen
Neon- eller Supabase-databas.

Er features-tabell och er predictions-tabell finns medvetet **inte** där.
Dem designar ni själva när ni bestämt vad en feature-rad är för just ert
target — det beslutet är en del av uppgiften.

---

## Vad ni inte ska göra

- Ni ska inte träna fram en egen modell från grunden. Modellen får ni.
- Ni ska inte förbättra modellens träffsäkerhet som självändamål. Kursen
  handlar om infrastrukturen runt modellen, inte om modellering.
- Ni ska inte lägga till fler tjänster för att imponera. Fler komponenter än
  projektbeskrivningens avsnitt 3 ger inte högre betyg.

Frågor mellan lektionerna: kursens Teams-kanal, gärna öppet.
