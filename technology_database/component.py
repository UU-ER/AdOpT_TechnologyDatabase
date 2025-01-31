from pathlib import Path
import pandas as pd


class Component:
    def __init__(self):

        # Output units
        self.currency_out = None
        self.financial_year_out = None

        # Input units
        self.currency_in = None
        self.financial_year_in = None

        # Financial indicators
        self.lifetime = None
        self.discount_rate = None
        self.unit_capex = None
        self.opex_fix = None
        self.opex_var = None

    def _convert_currency(self, value):
        """
        Convert value to specified currency for given financial_year_in and currency_in

        Uses the average exchange rates for a year as given by the European Central Bank
        :param float value: value to convert
        :return:
        """
        rates_path = Path(__file__).parent / Path(
            "./data_currency_conversion/conversion_rates.csv"
        )
        rates = pd.read_csv(rates_path, index_col=0)
        rates.index = pd.to_datetime(rates.index)  # Convert the index to datetime
        rates_year_in = rates[rates.index.year == self.financial_year_in].mean()
        rates_year_out = rates[rates.index.year == self.financial_year_out].mean()

        # Convert to EUR
        if self.currency_in != "EUR":
            rate_eur = rates_year_in[self.currency_in]
            value = value / rate_eur

        value = self._correct_inflation(value)

        if self.currency_out != "EUR":
            rate_other = rates_year_out[self.currency_out]
            value = value * rate_other

        return value

    def _correct_inflation(self, value):

        ppi_path = Path(__file__).parent / Path(
            "./data_currency_conversion/producer_price_index_euro.csv"
        )
        ppi = pd.read_csv(ppi_path)
        ppi.index = pd.to_datetime(ppi["TIME_PERIOD"])  # Convert the index to datetime
        ppi_year_in = ppi[ppi.index.year == self.financial_year_in]["OBS_VALUE"].mean()
        ppi_year_out = ppi[ppi.index.year == self.financial_year_out][
            "OBS_VALUE"
        ].mean()

        return value * ppi_year_out / ppi_year_in

    def calculate_levelized_cost(self, capacity_factor):
        pass
