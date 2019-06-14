""" 
Bot that stays on 1base, goes 4 rax mass reaper
This bot is one of the first examples that are micro intensive
Bot has a chance to win against elite (=Difficulty.VeryHard) zerg AI

Bot made by Burny
"""

import random

import sc2
from sc2 import Race, Difficulty
from sc2.constants import *
from sc2.position import Point2, Point3
from sc2.unit import Unit
from sc2.player import Bot, Computer
from sc2.player import Human
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId

class MassReaperBot(sc2.BotAI):
    def __init__(self):
        self.combinedActions = []

    async def on_step(self, iteration):
        self.combinedActions = []

        """
        -  depots when low on remaining supply
        - townhalls contains commandcenter and orbitalcommand
        - self.units(TYPE).not_ready.amount selects all units of that type, filters incomplete units, and then counts the amount
        - self.already_pending(TYPE) counts how many units are queued - but in this bot below you will find a slightly different already_pending function which only counts units queued (but not in construction)
        """
        if self.supply_left < 5 and self.townhalls.exists and self.supply_used >= 14 and self.can_afford(UnitTypeId.SUPPLYDEPOT) and self.units(UnitTypeId.SUPPLYDEPOT).not_ready.amount + self.already_pending(UnitTypeId.SUPPLYDEPOT) < 1:
            ws = self.workers.gathering
            if ws: # if workers found
                w = ws.furthest_to(ws.center)
                loc = await self.find_placement(UnitTypeId.SUPPLYDEPOT, w.position, placement_step=3)
                if loc: # if a placement location was found
                    # build exactly on that location
                    self.combinedActions.append(w.build(UnitTypeId.SUPPLYDEPOT, loc))

        # lower all depots when finished
        for depot in self.units(UnitTypeId.SUPPLYDEPOT).ready:
            self.combinedActions.append(depot(AbilityId.MORPH_SUPPLYDEPOT_LOWER))

        # morph commandcenter to orbitalcommand
        if self.units(UnitTypeId.BARRACKS).ready.exists and self.can_afford(UnitTypeId.ORBITALCOMMAND): # check if orbital is affordable
            for cc in self.units(UnitTypeId.COMMANDCENTER).idle: # .idle filters idle command centers
                self.combinedActions.append(cc(AbilityId.UPGRADETOORBITAL_ORBITALCOMMAND))

        # expand if we can afford and have less than 2 bases
        if 1 <= self.townhalls.amount < 2 and self.already_pending(UnitTypeId.COMMANDCENTER) == 0 and self.can_afford(UnitTypeId.COMMANDCENTER):
            # get_next_expansion returns the center of the mineral fields of the next nearby expansion
            next_expo = await self.get_next_expansion()
            # from the center of mineral fields, we need to find a valid place to place the command center
            location = await self.find_placement(UnitTypeId.COMMANDCENTER, next_expo, placement_step=1)
            if location:
                # now we "select" (or choose) the nearest worker to that found location
                w = self.select_build_worker(location)
                if w and self.can_afford(UnitTypeId.COMMANDCENTER):
                    # the worker will be commanded to build the command center
                    error = await self.do(w.build(UnitTypeId.COMMANDCENTER, location))
                    if error:
                        print(error)

        # make up to 4 barracks if we can afford them
        # check if we have a supply depot (tech requirement) before trying to make barracks
        if self.units.of_type([UnitTypeId.SUPPLYDEPOT, UnitTypeId.SUPPLYDEPOTLOWERED, UnitTypeId.SUPPLYDEPOTDROP]).ready.exists and self.units(UnitTypeId.BARRACKS).amount + self.already_pending(UnitTypeId.BARRACKS) < 4 and self.can_afford(UnitTypeId.BARRACKS):
            ws = self.workers.gathering
            if ws and self.townhalls.exists: # need to check if townhalls.amount > 0 because placement is based on townhall location
                w = ws.furthest_to(ws.center)
                # I chose placement_step 4 here so there will be gaps between barracks hopefully
                loc = await self.find_placement(UnitTypeId.BARRACKS, self.townhalls.random.position, placement_step=4)
                if loc:
                    self.combinedActions.append(w.build(UnitTypeId.BARRACKS, loc))

        # build refineries (on nearby vespene) when at least one barracks is in construction
        if self.units(UnitTypeId.BARRACKS).amount > 0 and self.already_pending(UnitTypeId.REFINERY) < 1:
            for th in self.townhalls:
                vgs = self.state.vespene_geyser.closer_than(10, th)
                for vg in vgs:
                    if await self.can_place(UnitTypeId.REFINERY, vg.position) and self.can_afford(UnitTypeId.REFINERY):
                        ws = self.workers.gathering
                        if ws.exists: # same condition as above
                            w = ws.closest_to(vg)
                            # caution: the target for the refinery has to be the vespene geyser, not its position!
                            self.combinedActions.append(w.build(UnitTypeId.REFINERY, vg))


        # make scvs until 18, usually you only need 1:1 mineral:gas ratio for reapers, but if you don't lose any then you will need additional depots (mule income should take care of that)
        # stop scv production when barracks is complete but we still have a command cender (priotize morphing to orbital command)
        if self.can_afford(UnitTypeId.SCV) and self.supply_left > 0 and self.units(UnitTypeId.SCV).amount < 18 and (self.units(UnitTypeId.BARRACKS).ready.amount < 1 and self.units(UnitTypeId.COMMANDCENTER).idle.exists or self.units(UnitTypeId.ORBITALCOMMAND).idle.exists):
            for th in self.townhalls.idle:
                self.combinedActions.append(th.train(UnitTypeId.SCV))

        # make reapers if we can afford them and we have supply remaining
        if self.can_afford(UnitTypeId.REAPER) and self.supply_left > 0:
            # loop through all idle barracks
            for rax in self.units(UnitTypeId.BARRACKS).idle:
                self.combinedActions.append(rax.train(UnitTypeId.REAPER))

        # send workers to mine from gas
        if iteration % 25 == 0:
            await self.distribute_workers()

        # reaper micro
        for r in self.units(UnitTypeId.REAPER):

            # move to range 15 of closest unit if reaper is below 20 hp and not regenerating
            enemyThreatsClose = self.known_enemy_units.filter(lambda x: x.can_attack_ground).closer_than(15, r) # threats that can attack the reaper
            if r.health_percentage < 2/5 and enemyThreatsClose.exists:
                retreatPoints = self.neighbors8(r.position, distance=2) | self.neighbors8(r.position, distance=4)
                # filter points that are pathable
                retreatPoints = {x for x in retreatPoints if self.inPathingGrid(x)}
                if retreatPoints:
                    closestEnemy = enemyThreatsClose.closest_to(r)
                    retreatPoint = closestEnemy.position.furthest(retreatPoints)
                    self.combinedActions.append(r.move(retreatPoint))
                    continue # continue for loop, dont execute any of the following

            # reaper is ready to attack, shoot nearest ground unit
            enemyGroundUnits = self.known_enemy_units.not_flying.closer_than(5, r) # hardcoded attackrange of 5
            if r.weapon_cooldown == 0 and enemyGroundUnits.exists:
                enemyGroundUnits = enemyGroundUnits.sorted(lambda x: x.distance_to(r))
                closestEnemy = enemyGroundUnits[0]
                self.combinedActions.append(r.attack(closestEnemy))
                continue # continue for loop, dont execute any of the following
            
            # attack is on cooldown, check if grenade is on cooldown, if not then throw it to furthest enemy in range 5
            reaperGrenadeRange = self._game_data.abilities[AbilityId.KD8CHARGE_KD8CHARGE.value]._proto.cast_range
            enemyGroundUnitsInGrenadeRange = self.known_enemy_units.not_structure.not_flying.exclude_type([UnitTypeId.LARVA, UnitTypeId.EGG]).closer_than(reaperGrenadeRange, r)
            if enemyGroundUnitsInGrenadeRange.exists and (r.is_attacking or r.is_moving):
                # if AbilityId.KD8CHARGE_KD8CHARGE in abilities, we check that to see if the reaper grenade is off cooldown
                abilities = (await self.get_available_abilities(r))
                enemyGroundUnitsInGrenadeRange = enemyGroundUnitsInGrenadeRange.sorted(lambda x: x.distance_to(r), reverse=True)
                furthestEnemy = None
                for enemy in enemyGroundUnitsInGrenadeRange:
                    if await self.can_cast(r, AbilityId.KD8CHARGE_KD8CHARGE, enemy, cached_abilities_of_unit=abilities):
                        furthestEnemy = enemy
                        break
                if furthestEnemy:
                    self.combinedActions.append(r(AbilityId.KD8CHARGE_KD8CHARGE, furthestEnemy))
                    continue # continue for loop, don't execute any of the following

            # move towards to max unit range if enemy is closer than 4
            enemyThreatsVeryClose = self.known_enemy_units.filter(lambda x: x.can_attack_ground).closer_than(4.5, r) # hardcoded attackrange minus 0.5
            # threats that can attack the reaper
            if r.weapon_cooldown != 0 and enemyThreatsVeryClose.exists:
                retreatPoints = self.neighbors8(r.position, distance=2) | self.neighbors8(r.position, distance=4)               
                # filter points that are pathable by a reaper
                retreatPoints = {x for x in retreatPoints if self.inPathingGrid(x)}
                if retreatPoints:
                    closestEnemy = enemyThreatsVeryClose.closest_to(r)
                    retreatPoint = max(retreatPoints, key=lambda x: x.distance_to(closestEnemy) - x.distance_to(r))
                    # retreatPoint = closestEnemy.position.furthest(retreatPoints)
                    self.combinedActions.append(r.move(retreatPoint))
                    continue # continue for loop, don't execute any of the following

            # move to nearest enemy ground unit/building because no enemy unit is closer than 5
            allEnemyGroundUnits = self.known_enemy_units.not_flying
            if allEnemyGroundUnits.exists:
                closestEnemy = allEnemyGroundUnits.closest_to(r)
                self.combinedActions.append(r.move(closestEnemy))
                continue # continue for loop, don't execute any of the following

            # move to random enemy start location if no enemy buildings have been seen
            self.combinedActions.append(r.move(random.choice(self.enemy_start_locations)))
            
        # manage idle scvs, would be taken care by distribute workers aswell
        if self.townhalls.exists:
            for w in self.workers.idle:
                th = self.townhalls.closest_to(w)
                mfs = self.state.mineral_field.closer_than(10, th)
                if mfs:
                    mf = mfs.closest_to(w)
                    self.combinedActions.append(w.gather(mf))

        # manage orbital energy and drop mules
        for oc in self.units(UnitTypeId.ORBITALCOMMAND).filter(lambda x: x.energy >= 50):
            mfs = self.state.mineral_field.closer_than(10, oc)
            if mfs:
                mf = max(mfs, key=lambda x:x.mineral_contents)
                self.combinedActions.append(oc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, mf))

        # when running out of mineral fields near command center, fly to next base with minerals

        # execuite actions
        await self.do_actions(self.combinedActions)



    # helper functions

    # this checks if a ground unit can walk on a Point2 position
    def inPathingGrid(self, pos):
        # returns True if it is possible for a ground unit to move to pos - doesnt seem to work on ramps or near edges
        assert isinstance(pos, (Point2, Point3, Unit))
        pos = pos.position.to2.rounded
        return self._game_info.pathing_grid[(pos)] != 0

    # stolen and modified from position.py
    def neighbors4(self, position, distance=1):
        p = position
        d = distance
        return {
            Point2((p.x - d, p.y)),
            Point2((p.x + d, p.y)),
            Point2((p.x, p.y - d)),
            Point2((p.x, p.y + d)),
        }

    # stolen and modified from position.py
    def neighbors8(self, position, distance=1):
        p = position
        d = distance
        return self.neighbors4(position, distance) | {
            Point2((p.x - d, p.y - d)),
            Point2((p.x - d, p.y + d)),
            Point2((p.x + d, p.y - d)),
            Point2((p.x + d, p.y + d)),
        }



    # already pending function rewritten to only capture units in queue and queued buildings
    # the difference to bot_ai.py alredy_pending() is: it will not cover structures in construction
    def already_pending(self, unit_type):
        ability = self._game_data.units[unit_type.value].creation_ability
        unitAttributes = self._game_data.units[unit_type.value].attributes

        buildings_in_construction = self.units.structure(unit_type).not_ready  
        if 8 not in unitAttributes and any(o.ability == ability for w in (self.units.not_structure) for o in w.orders): 
            return sum([o.ability == ability for w in (self.units - self.workers) for o in w.orders])
        # following checks for unit production in a building queue, like queen, also checks if hatch is morphing to LAIR
        elif any(o.ability.id == ability.id for w in (self.units.structure) for o in w.orders):
            return sum([o.ability.id == ability.id for w in (self.units.structure) for o in w.orders])
        # the following checks if a worker is about to start a construction (and for scvs still constructing if not checked for structures with same position as target)
        elif any(o.ability == ability for w in self.workers for o in w.orders):
            return sum([o.ability == ability for w in self.workers for o in w.orders]) \
                - buildings_in_construction.amount
        elif any(egg.orders[0].ability == ability for egg in self.units(UnitTypeId.EGG)):
            return sum([egg.orders[0].ability == ability for egg in self.units(UnitTypeId.EGG)])
        return 0


    # distribute workers function rewritten, the default distribute_workers() function did not saturate gas quickly enough
    async def distribute_workers(self, performanceHeavy=True, onlySaturateGas=False):
        # expansion_locations = self.expansion_locations
        # owned_expansions = self.owned_expansions


        mineralTags = [x.tag for x in self.state.units.mineral_field]
        # gasTags = [x.tag for x in self.state.units.vespene_geyser]
        geyserTags = [x.tag for x in self.geysers]

        workerPool = self.units & []
        workerPoolTags = set()

        # find all geysers that have surplus or deficit
        deficitGeysers = {}
        surplusGeysers = {}
        for g in self.geysers.filter(lambda x:x.vespene_contents > 0):
            # only loop over geysers that have still gas in them
            deficit = g.ideal_harvesters - g.assigned_harvesters
            if deficit > 0:
                deficitGeysers[g.tag] = {"unit": g, "deficit": deficit}
            elif deficit < 0:
                surplusWorkers = self.workers.closer_than(10, g).filter(lambda w:w not in workerPoolTags and len(w.orders) == 1 and w.orders[0].ability.id in [AbilityId.HARVEST_GATHER] and w.orders[0].target in geyserTags)
                # workerPool.extend(surplusWorkers)
                for i in range(-deficit):
                    if surplusWorkers.amount > 0:
                        w = surplusWorkers.pop()
                        workerPool.append(w)
                        workerPoolTags.add(w.tag)
                surplusGeysers[g.tag] = {"unit": g, "deficit": deficit}

        # find all townhalls that have surplus or deficit
        deficitTownhalls = {}
        surplusTownhalls = {}
        if not onlySaturateGas:
            for th in self.townhalls:
                deficit = th.ideal_harvesters - th.assigned_harvesters
                if deficit > 0:
                    deficitTownhalls[th.tag] = {"unit": th, "deficit": deficit}
                elif deficit < 0:
                    surplusWorkers = self.workers.closer_than(10, th).filter(lambda w:w.tag not in workerPoolTags and len(w.orders) == 1 and w.orders[0].ability.id in [AbilityId.HARVEST_GATHER] and w.orders[0].target in mineralTags)
                    # workerPool.extend(surplusWorkers)
                    for i in range(-deficit):
                        if surplusWorkers.amount > 0:
                            w = surplusWorkers.pop()
                            workerPool.append(w)
                            workerPoolTags.add(w.tag)
                    surplusTownhalls[th.tag] = {"unit": th, "deficit": deficit}

            if all([len(deficitGeysers) == 0, len(surplusGeysers) == 0, len(surplusTownhalls) == 0 or deficitTownhalls == 0]):
                # cancel early if there is nothing to balance
                return

        # check if deficit in gas less or equal than what we have in surplus, else grab some more workers from surplus bases
        deficitGasCount = sum(gasInfo["deficit"] for gasTag, gasInfo in deficitGeysers.items() if gasInfo["deficit"] > 0)
        surplusCount = sum(-gasInfo["deficit"] for gasTag, gasInfo in surplusGeysers.items() if gasInfo["deficit"] < 0)
        surplusCount += sum(-thInfo["deficit"] for thTag, thInfo in surplusTownhalls.items() if thInfo["deficit"] < 0)

        if deficitGasCount - surplusCount > 0:
            # grab workers near the gas who are mining minerals
            for gTag, gInfo in deficitGeysers.items():
                if workerPool.amount >= deficitGasCount:
                    break
                workersNearGas = self.workers.closer_than(10, gInfo["unit"]).filter(lambda w:w.tag not in workerPoolTags and len(w.orders) == 1 and w.orders[0].ability.id in [AbilityId.HARVEST_GATHER] and w.orders[0].target in mineralTags)
                while workersNearGas.amount > 0 and workerPool.amount < deficitGasCount:
                    w = workersNearGas.pop()
                    workerPool.append(w)
                    workerPoolTags.add(w.tag)

        # now we should have enough workers in the pool to saturate all gases, and if there are workers left over, make them mine at townhalls that have mineral workers deficit
        for gTag, gInfo in deficitGeysers.items():
            if performanceHeavy:
                # sort furthest away to closest (as the pop() function will take the last element)
                workerPool.sort(key=lambda x:x.distance_to(gInfo["unit"]), reverse=True)
            for i in range(gInfo["deficit"]):
                if workerPool.amount > 0:
                    w = workerPool.pop()
                    if len(w.orders) == 1 and w.orders[0].ability.id in [AbilityId.HARVEST_RETURN]:
                        self.combinedActions.append(w.gather(gInfo["unit"], queue=True))
                    else:
                        self.combinedActions.append(w.gather(gInfo["unit"]))

        if not onlySaturateGas:
            # if we now have left over workers, make them mine at bases with deficit in mineral workers
            for thTag, thInfo in deficitTownhalls.items():
                if performanceHeavy:
                    # sort furthest away to closest (as the pop() function will take the last element)
                    workerPool.sort(key=lambda x:x.distance_to(thInfo["unit"]), reverse=True)
                for i in range(thInfo["deficit"]):
                    if workerPool.amount > 0:
                        w = workerPool.pop()
                        mf = self.state.mineral_field.closer_than(10, thInfo["unit"]).closest_to(w)
                        if len(w.orders) == 1 and w.orders[0].ability.id in [AbilityId.HARVEST_RETURN]:
                            self.combinedActions.append(w.gather(mf, queue=True))
                        else:
                            self.combinedActions.append(w.gather(mf))



def main():
    # Multiple difficulties for enemy bots available https://github.com/Blizzard/s2client-api/blob/ce2b3c5ac5d0c85ede96cef38ee7ee55714eeb2f/include/sc2api/sc2_gametypes.h#L30
    sc2.run_game(sc2.maps.get("(2)CatalystLE"), [
        Bot(Race.Terran, MassReaperBot()),
        Computer(Race.Zerg, Difficulty.VeryHard)
    ], realtime=False)

if __name__ == '__main__':
    main()
