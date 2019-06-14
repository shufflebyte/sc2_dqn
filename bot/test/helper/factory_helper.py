import unit_helper
from unit_helper import *


class Factory:
    def __init__(self, gamedata, old_state=[]):
        self.__gamedata = gamedata
        self.__units = []
        self.__old_state = old_state

    # name of unit class, amount of units to produce, arguments (see in unit_helper)
    def add_unit(self, name, count=1, **kwargs):
        for i in range(count):
            self.__units.append(getattr(unit_helper, name)(**kwargs).get_unit())

    # returns units array
    def get_units(self):
        if self.__old_state is []:
            return Units.from_proto(self.__units, self.__gamedata)
        else:
            units = Units.from_proto(self.__units, self.__gamedata)
            for unit in units:
                self.__old_state.append(unit)

            return self.__old_state

    def get_supply_cap(self):
        units = Units.from_proto(self.__units, self.__gamedata)
        supply_cap = (units(UnitTypeId.COMMANDCENTER).amount + units(UnitTypeId.COMMANDCENTERFLYING).amount + units(
            UnitTypeId.ORBITALCOMMAND).amount) * 15
        supply_cap += units(UnitTypeId.SUPPLYDEPOT).amount * 8
        return supply_cap
