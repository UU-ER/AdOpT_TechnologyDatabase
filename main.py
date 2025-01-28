from technology_database import create_component

year = 2024
r = 0.1

# CO2 pipeline
co2_pipeline = create_component(technology="CO2 pipeline", source="Oeuvray")
best_config = co2_pipeline.calculate_cost(
    currency="EUR",
    year=2022,
    discount_rate=r,
    timeframe="near-term",
    length_km=10,
    m_kg_per_s=1,
    terrain="Onshore",
    electricity_price_eur_per_mw=90,
    operating_hours_per_a=8760,
    p_inlet_bar=1.1,
)

print(best_config)
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
