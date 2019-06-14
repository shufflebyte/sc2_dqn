from asyncio import coroutine, get_event_loop, sleep
import asyncio
import unittest

import mock
from mock import patch, Mock
from unittest import TestCase
from sc2.player import Bot, Computer
from sc2.data import Race
# Load bot
from JAS.game_commander import GameCommander
from sc2.position import Point2
from sc2.ids.unit_typeid import UnitTypeId

from JAS.Agents.production_manager import ProductionManager
import pickle
from pathlib import Path
import numpy as np
from sc2.unit import UnitOrder
from sc2.units import Units
from s2clientprotocol import raw_pb2
from sc2.position import Point2, Point3

from test.helper.unit_helper import *
from test.helper.factory_helper import Factory

from JAS.Helper.rl_helper import RLHelper
from JAS.Agents.strategy_manager import StrategyId

import os


def CoroMock():
    coro = Mock(name="CoroutineResult")
    corofunc = Mock(name="CoroutineFunction", side_effect=coroutine(coro))
    corofunc.coro = coro
    return corofunc


# async def coro(a, b):
#     return await sleep(1, result=a+b)
#
# def some_action(a, b):
#     return get_event_loop().run_until_complete(coro(a, b))


# class ProductionManager:
#     def __init__(self):
#         self.__env = GameCommander()
#     async def coro(a, b):
#         return await sleep(1, result=9)
#     async def on_step(self):
#         res = await self.coro(1,2)
#         location = await self.__env.find_placement(UnitTypeId.BARRACKS, Point2((1,2)), placement_step=1)
#         return res, location


class TestFoo(TestCase):
    def test_rl_helper_prepare_observation(self):
        dirname = os.path.dirname(__file__)

        filename = os.path.join(dirname, 'game_out')
        out_file = Path(filename)
        if out_file.is_file():
            infile = open(out_file, 'rb')
            game = pickle.load(infile)
            infile.close()

        filename = os.path.join(dirname, 'state_out')
        out_file = Path(filename)
        if out_file.is_file():
            infile = open(out_file, 'rb')
            state = pickle.load(infile)
            infile.close()

            erg = RLHelper.prepare_observation(units=state["units"],
                                               enemies=state["state_enemy"],
                                               minerals=state["minerals"],
                                               vespene=state["vespene"],
                                               supply_used=state["supply_used"])
            # print(len(erg))

    # @patch('__main__.coro', new_callable=CoroMock)
    # def test(self, corofunc):
    #     a, b, c = 1, 2, 3
    #     corofunc.coro.return_value = c
    #     result = some_action(a, b)
    #     corofunc.assert_called_with(a, b)
    #     assert result == c
    #     print ("done")

    # @patch.object(ProductionManager, 'coro', new_callable=CoroMock)
    @patch.object(GameCommander, 'find_placement', new_callable=CoroMock)
    @patch.object(GameCommander, 'can_place', new_callable=CoroMock)
    def test2(self, corofunc, corofunc2):

        dirname = os.path.dirname(__file__)

        filename = os.path.join(dirname, 'game_out')
        out_file = Path(filename)
        if out_file.is_file():
            infile = open(out_file, 'rb')
            game = pickle.load(infile)
            infile.close()

        filename = os.path.join(dirname, 'state_out')
        out_file = Path(filename)
        if out_file.is_file():
            infile = open(out_file, 'rb')
            state = pickle.load(infile)
            infile.close()

        state["minerals"] = 3000  #
        gamedata = game["game_data"]

        # start a "game"
        game_commander = GameCommander()
        game_commander.prepare_step(game, state)
        game_commander._prepare_first_step()

        # actual testcase starts here:
        expansion_location = next(iter(game_commander.expansion_locations))
        # use unit factory to generate new units
        unit_factory = Factory(gamedata, state["units"])
        unit_factory.add_unit("SCV", 5, pos=expansion_location, near=10)
        unit_factory.add_unit("CommandCenter", pos=expansion_location)
        state["units"] = unit_factory.get_units()

        # print(state["units"](UnitTypeId.COMMANDCENTER).amount)
        # set new state
        game_commander.prepare_step(game, state)
        production_manager = ProductionManager(game_commander)
        production_manager._strategy = StrategyId.TVT_ECONOMIC
        # run step

        corofunc2.coro.return_value = True
        res = get_event_loop().run_until_complete(production_manager.on_step())
        print(res)
        assert res == []


if __name__ == "__main__":
    unittest.main()
