from .technologies import Dac_sievert
from .networks import Co2Pipeline_Oeuvray


def create_component(technology, source):
    if technology == "DAC - Solid Sorbent":
        if source == "Sievert":
            return Dac_sievert("SS")
    elif technology == "DAC - Liquid Solvent":
        if source == "Sievert":
            return Dac_sievert("LS")
    elif technology == "DAC - Calcium Looping":
        if source == "Sievert":
            return Dac_sievert("CaO")
    elif technology == "CO2 pipeline":
        if source == "Oeuvray":
            return Co2Pipeline_Oeuvray()


def show_available_technologies(self):
    # open json
    # print json
    pass
