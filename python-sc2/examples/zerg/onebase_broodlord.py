from functools import reduce
from operator import or_
import random

import sc2
from sc2 import Race, Difficulty
from sc2.constants import *
from sc2.player import Bot, Computer
from sc2.data import race_townhalls

import enum

class BroodlordBot(sc2.BotAI):
    def select_target(self):
        if self.known_enemy_structures.exists:
            return random.choice(self.known_enemy_structures).position

        return self.enemy_start_locations[0]

    async def on_step(self, iteration):
        larvae = self.units(LARVA)
        forces = self.units(ZERGLING) | self.units(CORRUPTOR) | self.units(BROODLORD)

        if self.units(BROODLORD).amount > 2 and iteration % 50 == 0:
            for unit in forces:
                await self.do(unit.attack(self.select_target()))

        if self.supply_left < 2:
            if self.can_afford(OVERLORD) and larvae.exists:
                await self.do(larvae.random.train(OVERLORD))
                return

        if self.units(GREATERSPIRE).ready.exists:
            corruptors = self.units(CORRUPTOR)
            # build half-and-half corruptors and broodlords
            if corruptors.exists and corruptors.amount > self.units(BROODLORD).amount:
                if self.can_afford(BROODLORD):
                    await self.do(corruptors.random.train(BROODLORD))
            elif self.can_afford(CORRUPTOR) and larvae.exists:
                await self.do(larvae.random.train(CORRUPTOR))
                return

        if not self.townhalls.exists:
            for unit in self.units(DRONE) | self.units(QUEEN) | forces:
                await self.do(unit.attack(self.enemy_start_locations[0]))
            return
        else:
            hq = self.townhalls.first

        for queen in self.units(QUEEN).idle:
            abilities = await self.get_available_abilities(queen)
            if AbilityId.EFFECT_INJECTLARVA in abilities:
                await self.do(queen(EFFECT_INJECTLARVA, hq))

        if not (self.units(SPAWNINGPOOL).exists or self.already_pending(SPAWNINGPOOL)):
            if self.can_afford(SPAWNINGPOOL):
                await self.build(SPAWNINGPOOL, near=hq)

        if self.units(SPAWNINGPOOL).ready.exists:
            if not self.units(LAIR).exists and not self.units(HIVE).exists and hq.noqueue:
                if self.can_afford(LAIR):
                    await self.do(hq.build(LAIR))

        if self.units(LAIR).ready.exists:
            if not (self.units(INFESTATIONPIT).exists or self.already_pending(INFESTATIONPIT)):
                if self.can_afford(INFESTATIONPIT):
                    await self.build(INFESTATIONPIT, near=hq)

            if not (self.units(SPIRE).exists or self.already_pending(SPIRE)):
                if self.can_afford(SPIRE):
                    await self.build(SPIRE, near=hq)

        if self.units(INFESTATIONPIT).ready.exists and not self.units(HIVE).exists and hq.noqueue:
            if self.can_afford(HIVE):
                await self.do(hq.build(HIVE))

        if self.units(HIVE).ready.exists:
            spires = self.units(SPIRE).ready
            if spires.exists:
                spire = spires.random
                if self.can_afford(GREATERSPIRE) and spire.noqueue:
                    await self.do(spire.build(GREATERSPIRE))

        if self.units(EXTRACTOR).amount < 2 and not self.already_pending(EXTRACTOR):
            if self.can_afford(EXTRACTOR):
                drone = self.workers.random
                target = self.state.vespene_geyser.closest_to(drone.position)
                err = await self.do(drone.build(EXTRACTOR, target))

        if hq.assigned_harvesters < hq.ideal_harvesters:
            if self.can_afford(DRONE) and larvae.exists:
                larva = larvae.random
                await self.do(larva.train(DRONE))
                return

        for a in self.units(EXTRACTOR):
            if a.assigned_harvesters < a.ideal_harvesters:
                w = self.workers.closer_than(20, a)
                if w.exists:
                    await self.do(w.random.gather(a))

        if self.units(SPAWNINGPOOL).ready.exists:
            if not self.units(QUEEN).exists and hq.is_ready and hq.noqueue:
                if self.can_afford(QUEEN):
                    await self.do(hq.train(QUEEN))

        if self.units(ZERGLING).amount < 40 and self.minerals > 1000:
            if larvae.exists and self.can_afford(ZERGLING):
                await self.do(larvae.random.train(ZERGLING))

def main():
    sc2.run_game(sc2.maps.get("(2)CatalystLE"), [
        Bot(Race.Zerg, BroodlordBot()),
        Computer(Race.Terran, Difficulty.Medium)
    ], realtime=False, save_replay_as="ZvT.SC2Replay")

if __name__ == '__main__':
    main()
