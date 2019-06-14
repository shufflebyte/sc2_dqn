import numpy as np
from gym import spaces

from sc2.ids.unit_typeid import UnitTypeId


class RLHelper:

    @staticmethod
    def get_proper_action(rl_action):
        max_index = rl_action
        if max_index == 0:
            action = 'build_worker'
        if max_index == 1:
            action = 'build_marine'
        if max_index == 2:
            action = 'build_supply'
        if max_index == 3:
            action = 'build_barracks'
        if max_index == 4:
            action = None

        return action

    @staticmethod
    def action_space():
        # A3C Config: in A3C action space is given as Box
        # num_actions = 4
        # low = np.zeros(num_actions, dtype=np.float32)
        # high = np.full(num_actions, 1, dtype=np.float32)
        # return spaces.Box(low, high, dtype=np.float32)

        # in APEX-DQN action space is given as discrete space
        # you only give the number of possible actions
        return spaces.Discrete(5)

    @staticmethod
    def observation_space():
        # APEX-DQN observation space is given as Box
        # low gives the minimum value, that the input value can have
        low = np.array([-1000, 0, 0, 0, 0, 0, 0, 0])
        # high gives the maximum value, that the input value can have
        high = np.array([30000, 200, 200, 20000, 200, 200, 200, 200])
        return spaces.Box(low, high, dtype=np.float32)

    @staticmethod
    def get_reward(score, score_old, units):
        # new immediate reward is calculated by substracting the reward value from this timestep from the last timestep
        # only new units (or whatever) will be used to calculate the new immediate reward
        reward = (score.food_used_army - score_old.food_used_army)

        # information given by score from sc2 score
        # print("score.food_used_none:", score.food_used_none)
        # print("score.food_used_army:", score.food_used_army)
        # print("score.food_used_economy:", score.food_used_economy)
        # print("score.food_used_technology:", score.food_used_technology)
        # print("score.food_used_upgrade:", score.food_used_upgrade)

        # information given from units by python-sc2
        # print("units(UnitTypeId.MARINE).amount:", units(UnitTypeId.MARINE).amount)

        return reward

    @staticmethod
    def get_end_reward(units):
        reward = 0
        # here an extra reward for winning or surviving can be given. Extra information is given by
        # units which can for example be accessed as follows:
        # units(UnitTypeId.SCV).amount
        return reward

    @staticmethod
    def prepare_observation(units,
                            enemies,
                            minerals,
                            vespene,
                            frame,
                            supply_left,
                            supply_used):
        # SIEGETANKSIEGED = 32
        # VIKINGASSAULT = 34
        # VIKINGFIGHTER = 35
        # FACTORYFLYING = 43
        # STARPORTFLYING = 44
        feature_vector = []
        feature_vector.append(minerals)
        feature_vector.append(supply_left)
        feature_vector.append(supply_used)
        # feature_vector.append(vespene)
        # 24 frames per second * 60 seconds * 15 min in minigame buildmarines
        feature_vector.append(frame)
        # feature_vector.append(units(UnitTypeId.COMMANDCENTER).amount + units(
        #     UnitTypeId.COMMANDCENTERFLYING).amount + units(UnitTypeId.ORBITALCOMMAND).amount)
        feature_vector.append(units(UnitTypeId.SUPPLYDEPOT).amount)
        # feature_vector.append(units(UnitTypeId.REFINERY).amount)
        feature_vector.append(units(UnitTypeId.BARRACKS).amount)
        # feature_vector.append(units(UnitTypeId.ENGINEERINGBAY).amount)
        # feature_vector.append(units(UnitTypeId.MISSILETURRET).amount)
        # feature_vector.append(units(UnitTypeId.BUNKER).amount)
        # feature_vector.append(units(UnitTypeId.SENSORTOWER).amount)
        # feature_vector.append(units(UnitTypeId.GHOSTACADEMY).amount)
        # feature_vector.append(units(UnitTypeId.FACTORY).amount)
        # feature_vector.append(
        #     units(UnitTypeId.STARPORT).amount + units(UnitTypeId.STARPORTFLYING).amount)
        # feature_vector.append(units(UnitTypeId.ARMORY).amount)
        # feature_vector.append(units(UnitTypeId.FUSIONCORE).amount)
        # feature_vector.append(units(UnitTypeId.AUTOTURRET).amount)
        # feature_vector.append(units(UnitTypeId.SIEGETANK).amount)
        # feature_vector.append(units(UnitTypeId.BARRACKSTECHLAB).amount)
        # feature_vector.append(units(UnitTypeId.BARRACKSREACTOR).amount)
        # feature_vector.append(units(UnitTypeId.FACTORYTECHLAB).amount)
        # feature_vector.append(units(UnitTypeId.FACTORYREACTOR).amount)
        # feature_vector.append(units(UnitTypeId.STARPORTTECHLAB).amount)
        # feature_vector.append(units(UnitTypeId.STARPORTREACTOR).amount)

        feature_vector.append(units(UnitTypeId.SCV).amount)
        feature_vector.append(units(UnitTypeId.MARINE).amount)
        # feature_vector.append(units(UnitTypeId.REAPER).amount)
        # feature_vector.append(units(UnitTypeId.GHOST).amount)
        # feature_vector.append(units(UnitTypeId.MARAUDER).amount)
        # feature_vector.append(units(UnitTypeId.THOR).amount)
        # feature_vector.append(units(UnitTypeId.HELLION).amount)
        # feature_vector.append(units(UnitTypeId.MEDIVAC).amount)
        # feature_vector.append(units(UnitTypeId.BANSHEE).amount)
        # feature_vector.append(units(UnitTypeId.RAVEN).amount)
        # feature_vector.append(units(UnitTypeId.BATTLECRUISER).amount)

        # feature_vector.append(enemies(UnitTypeId.COMMANDCENTER).amount + units(
        #     UnitTypeId.COMMANDCENTERFLYING).amount + units(UnitTypeId.ORBITALCOMMAND).amount)
        # feature_vector.append(enemies(UnitTypeId.SUPPLYDEPOT).amount)
        # feature_vector.append(enemies(UnitTypeId.REFINERY).amount)
        # feature_vector.append(enemies(UnitTypeId.BARRACKS).amount)
        # feature_vector.append(enemies(UnitTypeId.ENGINEERINGBAY).amount)
        # feature_vector.append(enemies(UnitTypeId.MISSILETURRET).amount)
        # feature_vector.append(enemies(UnitTypeId.BUNKER).amount)
        # feature_vector.append(enemies(UnitTypeId.SENSORTOWER).amount)
        # feature_vector.append(enemies(UnitTypeId.GHOSTACADEMY).amount)
        # feature_vector.append(enemies(UnitTypeId.FACTORY).amount)
        # feature_vector.append(
        #     enemies(UnitTypeId.STARPORT).amount + units(UnitTypeId.STARPORTFLYING).amount)
        # feature_vector.append(enemies(UnitTypeId.ARMORY).amount)
        # feature_vector.append(enemies(UnitTypeId.FUSIONCORE).amount)
        # feature_vector.append(enemies(UnitTypeId.AUTOTURRET).amount)
        # feature_vector.append(enemies(UnitTypeId.SIEGETANK).amount)
        # feature_vector.append(enemies(UnitTypeId.BARRACKSTECHLAB).amount)
        # feature_vector.append(enemies(UnitTypeId.BARRACKSREACTOR).amount)
        # feature_vector.append(enemies(UnitTypeId.FACTORYTECHLAB).amount)
        # feature_vector.append(enemies(UnitTypeId.FACTORYREACTOR).amount)
        # feature_vector.append(enemies(UnitTypeId.STARPORTTECHLAB).amount)
        # feature_vector.append(enemies(UnitTypeId.STARPORTREACTOR).amount)

        # feature_vector.append(enemies(UnitTypeId.SCV).amount)
        # feature_vector.append(enemies(UnitTypeId.MARINE).amount)
        # feature_vector.append(enemies(UnitTypeId.REAPER).amount)
        # feature_vector.append(enemies(UnitTypeId.GHOST).amount)
        # feature_vector.append(enemies(UnitTypeId.MARAUDER).amount)
        # feature_vector.append(enemies(UnitTypeId.THOR).amount)
        # feature_vector.append(enemies(UnitTypeId.HELLION).amount)
        # feature_vector.append(enemies(UnitTypeId.MEDIVAC).amount)
        # feature_vector.append(enemies(UnitTypeId.BANSHEE).amount)
        # feature_vector.append(enemies(UnitTypeId.RAVEN).amount)
        # feature_vector.append(enemies(UnitTypeId.BATTLECRUISER).amount)

        return np.array(feature_vector)
