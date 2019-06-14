import sc2
from sc2 import Race, Difficulty
from sc2.player import Bot, Computer

# Load bot
from JAS.game_commander import GameCommander

try:
    from ray.rllib.utils.policy_client import PolicyClient
except ImportError:
    print("Get an Operating System!")

import sys
import getopt


def main(argv):
    port = 0
    client = None
    real_time = False
    forever = False

    # read out command line arguments
    try:
        opts, args = getopt.getopt(
            argv, "hp:rf", ["port=", "real-time", "forever"])
    except getopt.GetoptError:
        print('run.py -p <port>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('run.py -p <port> -r')
            print('-r --real-time  - run game in real-time')
            print('-f --forever  - run game in a cycle')
            sys.exit()
        elif opt in ("-p", "--port"):
            port = int(arg)
        elif opt in ("-r", "--real-time"):
            real_time = True
        elif opt in ("-f", "--forever"):
            forever = True

    # if port is given start client
    if port > 0:
        client = PolicyClient("http://localhost:" + str(port))

    if not forever:
        start_game(client, real_time)

    if forever:
        while True:
            start_game(client, real_time)


def start_game(client, real_time):
    # Local game
    bot = Bot(Race.Terran, GameCommander(client))
    print("Starting local game...")
    sc2.run_game(sc2.maps.get("BuildMarines"), [
        bot,
        Computer(Race.Terran, Difficulty.VeryHard)
    ], realtime=real_time)


if __name__ == "__main__":
    main(sys.argv[1:])
