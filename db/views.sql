-- Silver layer.
--
-- raw__ is append-only: ask a source twice and you get two rows, on purpose.
-- These views are where that gets resolved. Each one picks the newest
-- ingestion of each natural key, then unpacks the JSON into typed columns.
--
-- Nothing here filters to one team's area or station. The view describes the
-- data, not your assignment — put the WHERE in the query that uses it.
--
--   psql "$DATABASE_URL" -f db/views.sql
--
-- Re-runnable: CREATE OR REPLACE, no data is moved or copied.

-- ---------------------------------------------------------------------------
-- Prices: one row per price interval.
--
-- Note that an interval is not always an hour. Sweden went from hourly to
-- 15-minute prices on 2025-10-01, so a day has 24 rows before that date and
-- 96 after. Do not assume 24 anywhere downstream.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW stg__elpris AS
WITH latest AS (
    SELECT DISTINCT ON (price_area, price_date)
           price_area,
           price_date,
           ingestion_timestamp,
           data
    FROM raw__elpris
    ORDER BY price_area, price_date, ingestion_timestamp DESC
)
SELECT
    l.price_area,
    l.price_date,
    (e ->> 'time_start')::timestamptz AS starts_at,
    (e ->> 'time_end')::timestamptz   AS ends_at,
    (e ->> 'SEK_per_kWh')::numeric    AS sek_per_kwh,
    (e ->> 'EUR_per_kWh')::numeric    AS eur_per_kwh,
    (e ->> 'EXR')::numeric            AS exr,
    l.ingestion_timestamp
FROM latest AS l,
     LATERAL jsonb_array_elements(l.data) AS e;

-- ---------------------------------------------------------------------------
-- Weather: one row per station, parameter and observation time.
--
-- Long, not wide: one row per measurement, not one column per parameter.
-- Pivoting to "one row per hour with temperature and wind side by side" is a
-- feature decision, so it belongs in the feature layer, not here.
--
-- `quality` is SMHI's own flag. G = controlled and approved, Y = coarser
-- control. Keep it: it is information, and dropping it now means you cannot
-- change your mind later.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW stg__weather AS
SELECT DISTINCT ON (station, parameter, observed_at)
       station,
       parameter,
       observed_at,
       (data ->> 'value')::numeric AS value,
       data ->> 'quality'          AS quality,
       ingestion_timestamp
FROM raw__weather
ORDER BY station, parameter, observed_at, ingestion_timestamp DESC;

-- ---------------------------------------------------------------------------
-- Acceptance test: run the backfill twice, then run this.
--
--   raw__  MUST grow      (bronze is append-only)
--   stg__  MUST NOT grow  (silver picks one version per natural key)
--
-- If raw did not grow, your raw layer is not append-only.
-- If stg grew, your silver layer is not picking a winner.
-- ---------------------------------------------------------------------------

-- SELECT (SELECT count(*) FROM raw__elpris)  AS raw_prices,
--        (SELECT count(*) FROM stg__elpris)  AS stg_prices,
--        (SELECT count(*) FROM raw__weather) AS raw_weather,
--        (SELECT count(*) FROM stg__weather) AS stg_weather;