# u = {display_type: Visible
# alliance: Self
# tag: 4348968961
# unit_type: 45
# owner: 1
# pos {
#   x: 37.220458984375
#   y: 127.62646484375
#   z: 11.984375
# }
# facing: 1.8188343048095703
# radius: 0.375
# build_progress: 1.0
# cloak: NotCloaked
# is_selected: false
# is_on_screen: true
# is_blip: false
# health: 45.0
# health_max: 45.0
# shield: 0.0
# energy: 0.0
# is_flying: false
# is_burrowed: false
# orders {
#   ability_id: 295
#   target_unit_tag: 4294967297
# }
# assigned_harvesters: 0
# ideal_harvesters: 0
# weapon_cooldown: 0.0
# shield_max: 0.0
# energy_max: 0.0}

from s2clientprotocol import raw_pb2
from random import randint
from sc2.units import Units
from sc2.position import Point2, Point3
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
import random

class BaseUnit(object):
    def __init__(self, unit_type, tag=0, owner=1, pos=None, near=0, facing=2, radius=0.375,
                 build_progress=1, display_type=raw_pb2.Visible, cloak=raw_pb2.NotCloaked, is_selected=False,
                 is_on_screen=True,
                 is_blip=False, health=0, health_max=0, shield=0, energy=0,
                 is_flying=False, is_burrowed=False, orders=[], assigned_harvesters=0, ideal_harvesters=0,
                 weapon_cooldown=0, shield_max=0, energy_max=0, args=[]):

        if "near" in args:
            near = args["near"]
        if "pos" in args:
            pos = args["pos"]
        if "orders" in args:
            orders = args["orders"]

        pb_unit = raw_pb2.Unit()
        if tag == 0:
            tag = randint(1000000000, 99999999990)
        pb_unit.tag = tag
        pb_unit.unit_type = unit_type.value
        pb_unit.owner = owner if "owner" not in args else args["owner"]

        if pos is None:     # if pos not set, set random location
            x = randint(30, 200)
            y = randint(30, 200)
            z = randint(10, 20)
        else:
            if type(pos) == Point2:     # if point set
                if near > 0:            # and should be near a position
                    # generate a new point with random number near former position in boarders of "near"
                    pos = Point2((pos.x + random.randint(-near, near), pos.y + random.randint(-near, near)))
                pos = pos.to3
            x = pos.x
            y = pos.y
            z = pos.z
        pb_unit.pos.x = x
        pb_unit.pos.y = y
        pb_unit.pos.z = z

        pb_unit.facing = facing
        pb_unit.radius = radius
        pb_unit.build_progress = build_progress
        pb_unit.display_type = display_type
        pb_unit.cloak = cloak
        pb_unit.is_selected = is_selected
        pb_unit.is_on_screen = is_on_screen
        pb_unit.is_blip = is_blip
        pb_unit.health = health if "health" not in args else args["health"]
        if "health_max" in args:
            health_max = args["health_max"]
        pb_unit.health_max = health_max if health_max > 0 else health
        pb_unit.shield = shield
        pb_unit.energy = energy
        pb_unit.is_flying = is_flying
        pb_unit.is_burrowed = is_burrowed
        for order in orders:
            new_order = pb_unit.orders.add()
            new_order.ability_id = order["ability_id"].value
            new_order.target_unit_tag = order["target_unit_tag"]
        # pb_unit.orders = orders
        pb_unit.assigned_harvesters = assigned_harvesters
        pb_unit.ideal_harvesters = ideal_harvesters
        pb_unit.weapon_cooldown = weapon_cooldown
        pb_unit.shield_max = shield_max if shield_max > 0 else shield
        pb_unit.energy_max = energy_max
        self.pb_unit = pb_unit

    def get_unit(self):
        return self.pb_unit


"""

TODO: UnitFactory: add supply ( for each depot and CC!)

"""


class CommandCenter(BaseUnit):
    def __init__(self, **kwargs):
        super(CommandCenter, self).__init__(UnitTypeId.COMMANDCENTER, ideal_harvesters=16, assigned_harvesters=16,
                                            health=1500, health_max=1500, shield=1, shield_max=3, args=kwargs)


class OrbitalCommand(BaseUnit):
    def __init__(self, **kwargs):
        super(OrbitalCommand, self).__init__(UnitTypeId.ORBITALCOMMAND, ideal_harvesters=16, assigned_harvesters=16,
                                             health=1500, health_max=1500, shield=1, shield_max=3, args=kwargs)


class SupplyDepot(BaseUnit):
    def __init__(self, **kwargs):
        super(SupplyDepot, self).__init__(UnitTypeId.SUPPLYDEPOT, health=400, shield=1, shield_max=3, args=kwargs)


class Refinery(BaseUnit):
    def __init__(self, **kwargs):
        super(Refinery, self).__init__(UnitTypeId.REFINERY, health=500, shield=1, shield_max=3, args=kwargs)


class Barracks(BaseUnit):
    def __init__(self, **kwargs):
        super(Barracks, self).__init__(UnitTypeId.BARRACKS, health=1000, shield=1, shield_max=3, args=kwargs)


class Factory(BaseUnit):
    def __init__(self, **kwargs):
        super(Factory, self).__init__(UnitTypeId.FACTORY, health=1250, shield=1, shield_max=3, args=kwargs)


class GhostAcademy(BaseUnit):
    def __init__(self, **kwargs):
        super(GhostAcademy, self).__init__(UnitTypeId.GHOSTACADEMY, health=1250, shield=1, shield_max=3, args=kwargs)


class Bunker(BaseUnit):
    def __init__(self, **kwargs):
        super(Bunker, self).__init__(UnitTypeId.BUNKER, health=400, shield=1, shield_max=3, args=kwargs)


class Armory(BaseUnit):
    def __init__(self, **kwargs):
        super(Armory, self).__init__(UnitTypeId.ARMORY, health=750, shield=1, shield_max=3, args=kwargs)


class Starport(BaseUnit):
    def __init__(self, **kwargs):
        super(Starport, self).__init__(UnitTypeId.STARPORT, health=1300, shield=1, shield_max=3, args=kwargs)


class FusionCore(BaseUnit):
    def __init__(self, **kwargs):
        super(FusionCore, self).__init__(UnitTypeId.FUSIONCORE, health=750, shield=1, shield_max=3, args=kwargs)


class EngineeringBay(BaseUnit):
    def __init__(self, **kwargs):
        super(EngineeringBay, self).__init__(UnitTypeId.ENGINEERINGBAY, health=850, shield=1, shield_max=3, args=kwargs)


class PlanetaryFortress(BaseUnit):
    def __init__(self, **kwargs):
        super(PlanetaryFortress, self).__init__(UnitTypeId.PLANETARYFORTRESS, ideal_harvesters=16,
                                                assigned_harvesters=16, health=1500, shield=3, shield_max=5, args=kwargs)


class SensorTower(BaseUnit):
    def __init__(self, **kwargs):
        super(SensorTower, self).__init__(UnitTypeId.SENSORTOWER, health=200, shield=0, shield_max=2, args=kwargs)


class MissileTurret(BaseUnit):
    def __init__(self, **kwargs):
        super(MissileTurret, self).__init__(UnitTypeId.MISSILETURRET, health=250, shield=0, shield_max=2, args=kwargs)


# Units of CommandCenter


class SCV(BaseUnit):
    def __init__(self, **kwargs):
        super(SCV, self).__init__(UnitTypeId.SCV, health=45, health_max=45,
                                  orders=[{"ability_id": AbilityId.HARVEST_GATHER_SCV, "target_unit_tag": 4294967297}], args=kwargs)


# Units of Barracks


class Marine(BaseUnit):
    def __init__(self, **kwargs):
        super(Marine, self).__init__(UnitTypeId.MARINE, health=45, health_max=55, shield_max=1, args=kwargs)


class Reaper(BaseUnit):
    def __init__(self, **kwargs):
        super(Reaper, self).__init__(UnitTypeId.MARINE, health=60, health_max=60, shield_max=1, args=kwargs)


class Maurader(BaseUnit):
    def __init__(self, **kwargs):
        super(Maurader, self).__init__(UnitTypeId.MARAUDER, health=125, health_max=125, shield=1, shield_max=2, args=kwargs)


class Ghost(BaseUnit):
    def __init__(self, **kwargs):
        super(Ghost, self).__init__(UnitTypeId.GHOST, health=100, health_max=100, shield_max=1, args=kwargs)


# Units of Starport


class Viking(BaseUnit):
    def __init__(self, **kwargs):
        super(Viking, self).__init__(UnitTypeId.VIKINGSKY_UNIT, health=135, shield_max=1, is_flying=True, args=kwargs)


class Medivac(BaseUnit):
    def __init__(self, **kwargs):
        super(Medivac, self).__init__(UnitTypeId.MEDIVAC, health=150, shield=1, shield_max=2, is_flying=True, args=kwargs)


class Liberator(BaseUnit):
    def __init__(self, **kwargs):
        super(Liberator, self).__init__(UnitTypeId.LIBERATOR, health=180, shield_max=1, is_flying=True, args=kwargs)


class Raven(BaseUnit):
    def __init__(self, **kwargs):
        super(Raven, self).__init__(UnitTypeId.RAVEN, health=140, shield_max=2, shield=1, is_flying=True, args=kwargs)


class Banshee(BaseUnit):
    def __init__(self, **kwargs):
        super(Banshee, self).__init__(UnitTypeId.BANSHEE, health=140, shield_max=1, is_flying=True, args=kwargs)


class Battlecruiser(BaseUnit):
    def __init__(self, **kwargs):
        super(Battlecruiser, self).__init__(UnitTypeId.BATTLECRUISER, health=550, shield=3, shield_max=4,
                                            is_flying=True, args=kwargs)


# Units of Factory


class Hellion(BaseUnit):
    def __init__(self, **kwargs):
        super(Hellion, self).__init__(UnitTypeId.HELLION, health=90, shield_max=1, args=kwargs)


class WidowMine(BaseUnit):
    def __init__(self, **kwargs):
        super(WidowMine, self).__init__(UnitTypeId.WIDOWMINE, health=90, shield_max=1, args=kwargs)


class SiegeTank(BaseUnit):
    def __init__(self, **kwargs):
        super(SiegeTank, self).__init__(UnitTypeId.SIEGETANK, health=175, shield=1, shield_max=2, args=kwargs)


class Cyclone(BaseUnit):
    def __init__(self, **kwargs):
        super(Cyclone, self).__init__(UnitTypeId.CYCLONE, health=180, shield=1, shield_max=2, args=kwargs)


class Hellbat(BaseUnit):
    def __init__(self, **kwargs):
        print("Attention: Maybe wrong UnitTypeID")
        super(Hellbat, self).__init__(UnitTypeId.HELLBATACGLUESCREENDUMMY, health=135, shield_max=1, args=kwargs)


class Thor(BaseUnit):
    def __init__(self, **kwargs):
        super(Thor, self).__init__(UnitTypeId.THOR, health=400, shield=2, shield_max=3, args=kwargs)
