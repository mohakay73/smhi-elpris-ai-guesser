-- Starter bronze schema for the project's own Postgres (Neon or Supabase).
--
-- Two raw (bronze) tables, one per source: SMHI weather (your features) and
-- elprisetjustnu.se (your target). Same pattern as Week 2's local pipeline —
-- append-only, JSONB payload untouched, a technical primary key that is not
-- from the source. Nothing here is UNIQUE on purpose: re-ingesting the same
-- station/hour or area/date is allowed and expected, and your silver-layer
-- view decides which version wins.
--
-- Run this once against your own Neon/Supabase database before Thursday's
-- session (psql, or your DB provider's SQL console). It's also a fine
-- starting point to extend if your team's design needs more columns later —
-- nothing downstream depends on you keeping it exactly as-is.

CREATE TABLE IF NOT EXISTS raw__weather (
    raw_id              BIGSERIAL PRIMARY KEY,   -- technical key, not from the source
    ingestion_timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
    station              TEXT       NOT NULL,    -- your team's assigned SMHI station
    parameter            TEXT       NOT NULL,    -- e.g. temperature, wind — SMHI parameter id/name
    observed_at          TIMESTAMPTZ NOT NULL,   -- when the observation is valid for
    data                 JSONB      NOT NULL     -- the untouched SMHI response fragment
);

CREATE INDEX IF NOT EXISTS raw__weather_natural_key_idx
    ON raw__weather (station, parameter, observed_at);

CREATE TABLE IF NOT EXISTS raw__elpris (
    raw_id              BIGSERIAL PRIMARY KEY,
    ingestion_timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
    price_area           TEXT       NOT NULL,    -- your team's assigned area, e.g. SE3
    price_date           DATE       NOT NULL,    -- the date the prices belong to
    data                 JSONB      NOT NULL     -- the untouched elprisetjustnu.se response
);

CREATE INDEX IF NOT EXISTS raw__elpris_natural_key_idx
    ON raw__elpris (price_area, price_date);

-- ---------------------------------------------------------------------------
-- A note on TIMESTAMPTZ, because it will bite you otherwise
-- ---------------------------------------------------------------------------
-- Both observed_at columns are TIMESTAMPTZ, which stores an absolute instant.
-- Your two sources do not agree on how to express one:
--
--   elprisetjustnu.se  "2026-08-20T00:00:00+02:00"   local time, offset given
--   SMHI               "2026-08-19";"22:00:00"       UTC, offset NOT given
--
-- Those two lines are the same moment. If you insert the SMHI value without
-- telling Postgres it is UTC, the server assumes its own timezone and your
-- weather silently lands one or two hours away from the prices it belongs to.
-- Nothing errors. Your model just gets quietly worse.
--
-- Set the offset explicitly on the way in, and verify once with:
--   SELECT observed_at AT TIME ZONE 'UTC', observed_at AT TIME ZONE 'Europe/Stockholm'
--   FROM raw__weather LIMIT 5;

-- ---------------------------------------------------------------------------
-- Not created here, on purpose
-- ---------------------------------------------------------------------------
-- Your features table (silver/gold) and your predictions/traceability table
-- (project brief, component D). Those are yours to design once you've decided
-- exactly what a "feature row" looks like for your team's target — that
-- decision is part of the exercise, not a given.
--
-- When you design the predictions table, component D requires that you can
-- answer four questions about any past prediction: which data it was built
-- on, which code produced it (git commit), which model version was used, and
-- when it was made. Sketch the columns before you write the insert.