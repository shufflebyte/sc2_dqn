import sc2

from sc2.cache import property_cache_forever
from typing import List, Dict, Set, Tuple, Any, Optional, Union  # mypy type checking
from sc2.position import Point2, Point3
from sc2.units import Units

# import KMeans
from sklearn.cluster import KMeans
import numpy as np

from sc2.game_data import GameData
from sc2.game_state import GameState
from sc2.data import Race, race_worker, race_townhalls, race_gas
from sc2.ids.upgrade_id import UpgradeId

import pickle
from pathlib import Path
import os

import math


class BotAI(sc2.BotAI):
    def __init__(self):
        super.__init__()

    def prepare_step(self, game, state):
        self._game_info = game["game_info"]
        self._game_data = game["game_data"]
        self.player_id = game["player_id"]
        self.race = game["race"]
        self.units = game["units"]
        self._units_previous_map = state["map"]
        self.units = state["units"]
        self.workers = state["workers"]
        self.townhalls = state["townhalls"]
        self.geysers = state["geysers"]
        self.minerals = state["minerals"]
        self.vespene = state["vespene"]
        self.supply_used = state["supply_used"]
        self.supply_cap = state["supply_cap"]
        self.supply_left = state["supply_left"]

        self.units_enemy = state["state_enemy"]  # .structure set implicit
        self.mineral_field = state["state_mineral"]
        self.vespene_geyser = state["state_gyser"]
        self.upgrades = state["state_upgrades"]
        self.game_loop = state["state_game"]
        self.score = state["score"]

    def _prepare_start(self, client, player_id, game_info, game_data):
        """Ran until game start to set game and player data."""
        self._client: "Client" = client
        self._game_info: "GameInfo" = game_info
        self._game_data: GameData = game_data

        self.player_id: int = player_id
        self.race: Race = Race(self._game_info.player_races[self.player_id])
        self._units_previous_map = dict()
        self.units: Units = Units([], game_data)

        dirname = os.getcwd()
        filename = os.path.join(dirname, 'test/game_out')
        out_file = Path(filename)
        if out_file.is_file() == False:
            arr = {"game_info": self._game_info,
                   "game_data": self._game_data,
                   "player_id": self.player_id,
                   "race": self.race,
                   "units": self.units}
            outfile = open(out_file, 'wb')
            pickle.dump(arr, outfile)

            outfile.close()

    def _prepare_step(self, state):
        """Set attributes from new state before on_step."""
        self.state: GameState = state

        # Required for events
        self._units_previous_map.clear()
        for unit in self.units:
            self._units_previous_map[unit.tag] = unit

        self.units: Units = state.units.owned
        self.workers: Units = self.units(race_worker[self.race])
        self.townhalls: Units = self.units(race_townhalls[self.race])
        self.geysers: Units = self.units(race_gas[self.race])

        self.minerals: Union[float, int] = state.common.minerals
        self.vespene: Union[float, int] = state.common.vespene
        self.supply_used: Union[float, int] = state.common.food_used
        self.supply_cap: Union[float, int] = state.common.food_cap
        self.supply_left: Union[float,
                                int] = self.supply_cap - self.supply_used

        """ set manually to never need to use state no more... """
        self.units_enemy = self.state.units.enemy  # .structure set implicit
        self.mineral_field = self.state.mineral_field
        self.vespene_geyser = self.state.vespene_geyser
        self.upgrades = self.state.upgrades
        self.game_loop = self.state.game_loop
        self.score = self.state.score

        dirname = os.getcwd()
        filename = os.path.join(dirname, 'test/state_out')
        out_file = Path(filename)
        if out_file.is_file() == False:

            arr = {"map": self._units_previous_map,
                   "units": self.units,
                   "workers": self.workers,
                   "townhalls": self.townhalls,
                   "geysers": self.geysers,
                   "minerals": self.minerals,
                   "vespene": self.vespene,
                   "supply_used": self.supply_used,
                   "supply_cap": self.supply_cap,
                   "supply_left": self.supply_left,
                   # "state_enemy_struct": self.state.units.enemy.structure,
                   "state_enemy": self.units_enemy,
                   "state_mineral": self.mineral_field,
                   "state_gyser": self.vespene_geyser,
                   "state_upgrades": self.upgrades,
                   "state_game": self.game_loop,
                   "score": self.score
                   }
            outfile = open(out_file, 'wb')
            pickle.dump(arr, outfile)

            outfile.close()

    @property
    def time(self) -> Union[int, float]:
        """ Returns time in seconds, assumes the game is played on 'faster' """
        return self.game_loop / 22.4  # / (1/1.4) * (1/16)

    @property
    def known_enemy_units(self) -> Units:
        """List of known enemy units, including structures."""
        return self.units_enemy

    @property
    def known_enemy_structures(self) -> Units:
        """List of known enemy units, structures only."""
        return self.units_enemy.structure

    def already_pending_upgrade(self, upgrade_type: UpgradeId) -> Union[int, float]:
        """ Check if an upgrade is being researched
        Return values:
        0: not started
        0 < x < 1: researching
        1: finished
        """
        assert isinstance(upgrade_type, UpgradeId)
        if upgrade_type in self.upgrades:
            return 1
        creationAbilityID = self._game_data.upgrades[upgrade_type.value].research_ability.id
        for s in self.units.structure.ready:
            for o in s.orders:
                if o.ability.id == creationAbilityID:
                    return o.progress
        return 0

    @property_cache_forever
    def expansion_locations(self) -> Dict[Point2, Units]:
        """List of possible expansion locations."""

        resources = self.mineral_field | self.vespene_geyser

        # prepare data for k-means
        ressorces_array = []
        for res in resources:
            ressorces_array.append([res.position.x, res.position.y])

        geysers = self.vespene_geyser
        # determine number of cluster center (two geyser for each location)
        cluster_numbers = geysers.amount / 2

        # create kmeans object
        kmeans = KMeans(n_clusters=int(cluster_numbers))
        # fit kmeans object to data
        kmeans.fit(ressorces_array)
        # get cluster centers
        cluster_centers = kmeans.cluster_centers_

        # gives us dict with index (cluster point) and index of points in resources/point_array
        cluster_index_to_resources_index = {i: np.where(
            kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}
        result = dict()
        for cluster_index in cluster_index_to_resources_index:
            result[Point2((cluster_centers[cluster_index][0], cluster_centers[cluster_index][1]))] = \
                [resources[resource_index]
                    for resource_index in cluster_index_to_resources_index[cluster_index]]

        """ Returns dict with center of resources as key, resources (mineral field, vespene geyser) as value """
        return result

    async def get_enemy_natural(self):
        """ Returns closest expansion to enemy start location """

        natural = None
        shortest_distance = math.inf
        enemy_pos = self.enemy_start_locations[0]

        for expansion in self.expansion_locations:
            distance = await self._client.query_pathing(enemy_pos, expansion)

            if distance is None:
                continue

            if distance < self.EXPANSION_GAP_THRESHOLD:
                # is main
                continue

            if distance < shortest_distance:
                shortest_distance = distance
                natural = expansion

        return natural
