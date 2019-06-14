import logbook
# from random import randint

from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from JAS.Agents.base_manager import BaseManager
# from JAS.ids.manager_type import ManagerType


class ProduceMarines(BaseManager):
    def __init__(self, env):
        super(ProduceMarines, self).__init__()
        self.logger = logbook.Logger(__name__)
        self.__combinedActions = []
        self.__env = env
        self.__build_site = Point2((40, 27.5))

        self.available_actions = {
            'build_worker': self.produce_workers,
            'build_marine': self.produce_marines,
            'build_supply': self.construct_building,
            'build_barracks':  self.construct_building
        }

    async def on_step(self, action=None):
        self.__combinedActions = []

        self.send_to_work()

        if action is None:
            pass
            # rnd = randint(0, len(self.available_actions) - 1)
            # action = list(self.available_actions.keys())[rnd]
        if action == 'build_worker':
            self.available_actions['build_worker']()
        elif action == 'build_marine':
            self.available_actions['build_marine']()
        elif action == 'build_supply':
            await self.available_actions['build_supply'](UnitTypeId.SUPPLYDEPOT)
        elif action == 'build_barracks':
            await self.available_actions['build_barracks'](UnitTypeId.BARRACKS)
        return self.__combinedActions

    def send_to_work(self):
        for worker in self.__env.workers.idle:
            mf = self.__env.state.mineral_field.closer_than(
                10, self.__env.start_location).closest_to(worker)
            self.__combinedActions.append(worker.gather(mf))

    def lower_depots(self):
        """ Sadly not available in minigame """
        for depot in self.__env.units(UnitTypeId.SUPPLYDEPOT).ready:
            self.__combinedActions.append(
                depot(AbilityId.MORPH_SUPPLYDEPOT_LOWER))

    def produce_workers(self):
        """ Produces SCV if affordable """
        for town_hall in self.__env.townhalls.idle:
            if self.__env.can_afford(UnitTypeId.SCV):
                self.__combinedActions.append(
                    (town_hall.train(UnitTypeId.SCV)))

    def produce_marines(self):
        """ Produces marine production if supply left """
        for rax in self.__env.units(UnitTypeId.BARRACKS).ready.idle:
            if self.__env.can_afford(UnitTypeId.MARINE):
                self.__combinedActions.append(rax.train(UnitTypeId.MARINE))

            # if rax.has_add_on and self.__env.units.find_by_tag(rax.add_on_tag).type_id == UnitTypeId.BARRACKSREACTOR:
            #     if self.__env.can_afford(UnitTypeId.MARINE):
            #         self.__combinedActions.append(
            #             rax.train(UnitTypeId.MARINE, queue=True))

    async def construct_building(self, building, target=None, placement_step=1):
        """ Construct a building near to location, if no location given, it takes start position town hall

        Keyword arguments:
        building -- UnitTypeId of a building
        target -- location where to place the building
        """
        if self.__env.can_afford(building):
            if target is None:
                target = self.__build_site

            location = await self.__env.find_placement(building, target, placement_step=placement_step)
            if location:
                workers = self.__env.workers.gathering
                if workers.amount > 0:
                    w = workers.closest_to(location)
                    if w:
                        self.__combinedActions.append(
                            w.build(building, location))

    async def build_refinery(self, geyser):
        if await self.__env.can_place(UnitTypeId.REFINERY, geyser.position):
            workers = self.__env.workers.gathering
            if workers.amount > 0:
                w = workers.closest_to(geyser.position)
                if w:
                    self.__combinedActions.append(
                        w.build(UnitTypeId.REFINERY, geyser))
