from JAS.Agents.produce_marines import ProduceMarines
from JAS.Helper.bot_ai import BotAI
from JAS.Helper.rl_helper import RLHelper

import logbook

# frames that pass per second
FRAMES_PER_SECOND = 22  # gerundet, eigentlich 22.4  # / (1/1.4) * (1/16)
# number of actions, the agent is allowed to make per second
ACTION_EVERY_SECONDS = 1 * FRAMES_PER_SECOND
# currently commented, because it produces errors in training...


class GameCommander(BotAI):
    def __init__(self, client=None):
        #super(GameCommander, self).__init__()
        self.logger = logbook.Logger(__name__)
        self.logger.level = logbook.INFO

        self.__combinedActions = []

        self.produce_marines = ProduceMarines(self)
        self.score_old = None
        self.client = client

        if self.client is not None:
            self.eid = self.client.start_episode(training_enabled=True)
            self.action_space = RLHelper.action_space()
            self.observation_space = RLHelper.observation_space()

    async def on_step(self, iteration):
        action = None
        if self.client is not None:
            # if iteration % ACTION_EVERY_SECONDS == 0:  # send message every ACTION_EVERY_SECONDS
            if True:
                # first set reward of previous action
                if self.score_old is None:
                    self.score_old = self.score
                reward = RLHelper.get_reward(self.score, self.score_old, self.units)
                self.score_old = self.score

                self.client.log_returns(self.eid, reward)  # , info=info)

                obs = RLHelper.prepare_observation(
                    self.units, self.units_enemy, self.minerals, self.vespene, self.game_loop, self.supply_left, self.supply_used)

                rl_action = self.client.get_action(self.eid, obs)

                action = RLHelper.get_proper_action(rl_action)

        self.__combinedActions = []

        actions = await self.produce_marines.on_step(action)
        self.__combinedActions.extend(actions)

        return await self.do_actions(self.__combinedActions)

    def on_end(self, game_result):
        """Ran at the end of a game."""
        reward = RLHelper.get_end_reward(self.units)
        print("END!! ", game_result, "Reward: ", reward)
        if self.client is not None:
            # Todo: set reward for end of game!

            self.client.log_returns(self.eid, reward)
            obs = RLHelper.prepare_observation(self.units, self.units_enemy, self.minerals, self.vespene,
                                               self.game_loop, self.supply_left, self.supply_used)
            self.client.end_episode(self.eid, obs)
            #self.eid = self.client.start_episode(training_enabled=True)
