from datetime import date
import pandas as pd
from model.train import daily_price_table


def test_daily_price_table_counts_intervals():
    prices = pd.DataFrame(
        {
            "date_local": [date(2026, 1, 1), date(2026, 1, 1)],
            "sek_per_kwh": [0.40, 0.60],
        }
    )

    daily = daily_price_table(prices)
    assert daily.loc[date(2026, 1, 1), "n_intervals"] == 2