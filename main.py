import pandas as pd

from technology_database import create_component

l_km = 100
terrain = "Offshore"
year = 2024
r = 0.1
m_kg_per_s = range(5, 100, 5)
unit_capex_pipe = []
unit_capex_comp = []
lc = []


for m in m_kg_per_s:
    # CO2 pipeline
    co2_pipeline = create_component(technology="CO2 pipeline", source="Oeuvray")
    results = co2_pipeline.calculate_cost(
        currency="EUR",
        year=2022,
        discount_rate=r,
        timeframe="near-term",
        length_km=l_km,
        m_kg_per_s=m,
        terrain=terrain,
        electricity_price_eur_per_mw=90,
        operating_hours_per_a=8760,
        p_inlet_bar=1.1,
    )
    unit_capex_pipe.append(results["cost_pipeline"]["unit_capex"])
    unit_capex_comp.append(results["cost_compression"]["unit_capex"])
    lc.append(results["configuration"]["lc"])

pd.DataFrame(
    [m_kg_per_s, unit_capex_comp, unit_capex_pipe, lc],
    index=["m", "capex_comp", "capex_pipe", "lc"],
).T.to_excel("pipeline_costs.xlsx")

#
# # DAC
# dac = create_component(technology="DAC - Solid Sorbent", source="Sievert")
# dac.calculate_cost(currency="EUR",
#                    year=2022,
#                    discount_rate=r,
#                    cumulative_capacity=4000)
#
# print(dac.unit_capex)
# print(dac.opex_fix)
# print(dac.opex_var)
#
# dac.calculate_levelized_cost(capacity_factor=0.9)
