import sys
from features.ability_used import save_abilityused
from features.battle_damage import save_damage
from features.critical_hits_precision import save_cprecision
from features.hit_distance import save_distance
from features.level_upgrade import save_levelupgrade
from features.precision import save_precision
from features.status_percentage import save_status
from features.travel_speed import save_travelspeed
from features.weapon_use import save_weaponuse
from features.view_rotation import save_view

if len(sys.argv) != 2:
    print("Usage: python main.py DATA_DIR")
    sys.exit(1)

data_dir = sys.argv[1]

save_abilityused(data_dir)
save_damage(data_dir)
save_cprecision(data_dir)
save_distance(data_dir)
save_levelupgrade(data_dir)
save_precision(data_dir)
save_view(data_dir)
save_status(data_dir)
save_travelspeed(data_dir)
save_weaponuse(data_dir)
