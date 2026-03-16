"""
GPU-accelerated simulation compute engine using Taichi.
============================================================
Struct-of-Arrays storage for creature/food state as Taichi fields,
with GPU kernels for the hot-path operations in World.update().

Requires: pip install taichi  (or install pangea[gpu])

Usage:
    engine = ComputeEngine(use_gpu=True)
    engine.upload_species(registry)
    engine.upload_environment(biomes, hazards)
    engine.upload_creatures(creatures)
    engine.upload_food(food_list)
    engine.run_frame(dt, world_w, world_h, wrap, daylight)
    engine.download_creatures(creatures)
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pangea.creature import Creature
    from pangea.species import SpeciesRegistry
    from pangea.world import Biome, Food, Hazard

from pangea.config import (
    CARNIVORE_ATTACK_RANGE,
    GPU_MAX_BIOMES,
    GPU_MAX_CREATURES,
    GPU_MAX_FOOD,
    GPU_MAX_HAZARDS,
    GPU_MAX_SPECIES,
    MIN_THRUST_FRACTION,
    NN_HIDDEN_SIZE,
    NN_INPUT_SIZE,
    NN_OUTPUT_SIZE,
)


# ── Module State ────────────────────────────────────────────
# Everything Taichi-related (ti import, fields, kernels) is deferred
# until _ensure_initialized() is called, because ti.init() must
# precede ti.field() and @ti.kernel decoration.

_ti = None  # taichi module reference
_ready = False

# Field references (populated by _ensure_initialized)
_cx = _cy = _cheading = _cspeed = _cenergy = None
_calive = _cage = _cfood_eaten = _cunder_attack = None
_clast_turn = _cdist_traveled = _cdeath_processed = _cbreed_cooldown = None
_cspecies_idx = None
_cmax_speed = _cradius = _cvision = _cefficiency = _clifespan = None
_cW1 = _cb1 = _cW2 = _cb2 = None
_cinputs = _chidden = _coutputs = None
_cbiome_speed = _cbiome_danger = None
_fx = _fy = _fenergy = _fradius = _fage = _flifetime = None
_fis_corpse = _fspecies_idx = _falive = None
_sp_can_eat_plants = _sp_plant_food_mult = None
_sp_can_attack_other = _sp_can_attack_own = None
_sp_can_eat_other_corpse = _sp_can_eat_own_corpse = None
_sp_attack_damage = _sp_energy_steal = None
_sp_scavenge_radius = _sp_scavenge_energy = None
_sp_base_energy = _sp_energy_cost = _sp_turn_cost = _sp_food_heal = None
_sp_night_vision = None
_bx = _by = _bradius = _bspeed_mult = _benergy_drain = None
_hx = _hy = _hradius = _hdamage = None

# Kernel references (populated by _ensure_initialized)
_k_reset_under_attack = None
_k_age_food = None
_k_sense = None
_k_brain_forward = None
_k_think_and_act = None
_k_physics = None
_k_combat = None
_k_collisions = None


def _ensure_initialized(use_gpu: bool = True) -> None:
    """Import taichi, call ti.init(), allocate fields, define kernels."""
    global _ti, _ready
    if _ready:
        return

    import taichi as ti
    _ti = ti

    if use_gpu:
        ti.init(arch=ti.gpu, offline_cache=True)
    else:
        ti.init(arch=ti.cpu, offline_cache=True)

    arch = ti.cfg.arch
    print(f"[Pangea GPU] Taichi backend: {arch}")

    _allocate_fields(ti)
    _define_kernels(ti)
    _ready = True


def _allocate_fields(ti) -> None:
    """Create all Taichi fields."""
    # fmt: off
    global _cx, _cy, _cheading, _cspeed, _cenergy
    global _calive, _cage, _cfood_eaten, _cunder_attack
    global _clast_turn, _cdist_traveled, _cdeath_processed, _cbreed_cooldown
    global _cspecies_idx
    global _cmax_speed, _cradius, _cvision, _cefficiency, _clifespan
    global _cW1, _cb1, _cW2, _cb2
    global _cinputs, _chidden, _coutputs
    global _cbiome_speed, _cbiome_danger
    global _fx, _fy, _fenergy, _fradius, _fage, _flifetime
    global _fis_corpse, _fspecies_idx, _falive
    global _sp_can_eat_plants, _sp_plant_food_mult
    global _sp_can_attack_other, _sp_can_attack_own
    global _sp_can_eat_other_corpse, _sp_can_eat_own_corpse
    global _sp_attack_damage, _sp_energy_steal
    global _sp_scavenge_radius, _sp_scavenge_energy
    global _sp_base_energy, _sp_energy_cost, _sp_turn_cost, _sp_food_heal
    global _sp_night_vision
    global _bx, _by, _bradius, _bspeed_mult, _benergy_drain
    global _hx, _hy, _hradius, _hdamage
    # fmt: on

    MC = GPU_MAX_CREATURES
    MF = GPU_MAX_FOOD
    MS = GPU_MAX_SPECIES
    MB = GPU_MAX_BIOMES
    MH = GPU_MAX_HAZARDS
    NI = NN_INPUT_SIZE
    NH = NN_HIDDEN_SIZE
    NO = NN_OUTPUT_SIZE

    _cx = ti.field(dtype=ti.f32, shape=MC)
    _cy = ti.field(dtype=ti.f32, shape=MC)
    _cheading = ti.field(dtype=ti.f32, shape=MC)
    _cspeed = ti.field(dtype=ti.f32, shape=MC)
    _cenergy = ti.field(dtype=ti.f32, shape=MC)
    _calive = ti.field(dtype=ti.i32, shape=MC)
    _cage = ti.field(dtype=ti.f32, shape=MC)
    _cfood_eaten = ti.field(dtype=ti.i32, shape=MC)
    _cunder_attack = ti.field(dtype=ti.f32, shape=MC)
    _clast_turn = ti.field(dtype=ti.f32, shape=MC)
    _cdist_traveled = ti.field(dtype=ti.f32, shape=MC)
    _cdeath_processed = ti.field(dtype=ti.i32, shape=MC)
    _cbreed_cooldown = ti.field(dtype=ti.f32, shape=MC)
    _cspecies_idx = ti.field(dtype=ti.i32, shape=MC)

    _cmax_speed = ti.field(dtype=ti.f32, shape=MC)
    _cradius = ti.field(dtype=ti.f32, shape=MC)
    _cvision = ti.field(dtype=ti.f32, shape=MC)
    _cefficiency = ti.field(dtype=ti.f32, shape=MC)
    _clifespan = ti.field(dtype=ti.f32, shape=MC)

    _cW1 = ti.field(dtype=ti.f32, shape=(MC, NI, NH))
    _cb1 = ti.field(dtype=ti.f32, shape=(MC, NH))
    _cW2 = ti.field(dtype=ti.f32, shape=(MC, NH, NO))
    _cb2 = ti.field(dtype=ti.f32, shape=(MC, NO))

    _cinputs = ti.field(dtype=ti.f32, shape=(MC, NI))
    _chidden = ti.field(dtype=ti.f32, shape=(MC, NH))
    _coutputs = ti.field(dtype=ti.f32, shape=(MC, NO))

    _cbiome_speed = ti.field(dtype=ti.f32, shape=MC)
    _cbiome_danger = ti.field(dtype=ti.f32, shape=MC)

    _fx = ti.field(dtype=ti.f32, shape=MF)
    _fy = ti.field(dtype=ti.f32, shape=MF)
    _fenergy = ti.field(dtype=ti.f32, shape=MF)
    _fradius = ti.field(dtype=ti.f32, shape=MF)
    _fage = ti.field(dtype=ti.f32, shape=MF)
    _flifetime = ti.field(dtype=ti.f32, shape=MF)
    _fis_corpse = ti.field(dtype=ti.i32, shape=MF)
    _fspecies_idx = ti.field(dtype=ti.i32, shape=MF)
    _falive = ti.field(dtype=ti.i32, shape=MF)

    _sp_can_eat_plants = ti.field(dtype=ti.i32, shape=MS)
    _sp_plant_food_mult = ti.field(dtype=ti.f32, shape=MS)
    _sp_can_attack_other = ti.field(dtype=ti.i32, shape=MS)
    _sp_can_attack_own = ti.field(dtype=ti.i32, shape=MS)
    _sp_can_eat_other_corpse = ti.field(dtype=ti.i32, shape=MS)
    _sp_can_eat_own_corpse = ti.field(dtype=ti.i32, shape=MS)
    _sp_attack_damage = ti.field(dtype=ti.f32, shape=MS)
    _sp_energy_steal = ti.field(dtype=ti.f32, shape=MS)
    _sp_scavenge_radius = ti.field(dtype=ti.f32, shape=MS)
    _sp_scavenge_energy = ti.field(dtype=ti.f32, shape=MS)
    _sp_base_energy = ti.field(dtype=ti.f32, shape=MS)
    _sp_energy_cost = ti.field(dtype=ti.f32, shape=MS)
    _sp_turn_cost = ti.field(dtype=ti.f32, shape=MS)
    _sp_food_heal = ti.field(dtype=ti.f32, shape=MS)
    _sp_night_vision = ti.field(dtype=ti.f32, shape=MS)

    _bx = ti.field(dtype=ti.f32, shape=MB)
    _by = ti.field(dtype=ti.f32, shape=MB)
    _bradius = ti.field(dtype=ti.f32, shape=MB)
    _bspeed_mult = ti.field(dtype=ti.f32, shape=MB)
    _benergy_drain = ti.field(dtype=ti.f32, shape=MB)

    _hx = ti.field(dtype=ti.f32, shape=MH)
    _hy = ti.field(dtype=ti.f32, shape=MH)
    _hradius = ti.field(dtype=ti.f32, shape=MH)
    _hdamage = ti.field(dtype=ti.f32, shape=MH)


def _define_kernels(ti) -> None:
    """Define all Taichi kernels after ti.init() and field allocation."""
    # fmt: off
    global _k_reset_under_attack, _k_age_food, _k_sense
    global _k_brain_forward, _k_think_and_act, _k_physics
    global _k_combat, _k_collisions
    # fmt: on

    # ── Helper function ──

    @ti.func
    def copysign(magnitude, sign_val):
        result = ti.abs(magnitude)
        if sign_val < 0.0:
            result = -result
        return result

    # ── Kernels ──

    @ti.kernel
    def k_reset_under_attack(n: int):
        for i in range(n):
            _cunder_attack[i] = 0.0

    @ti.kernel
    def k_age_food(n: int, dt: float):
        for f in range(n):
            if _falive[f] == 0:
                continue
            _fage[f] += dt
            if _flifetime[f] > 0.0 and _fage[f] >= _flifetime[f]:
                _falive[f] = 0

    @ti.kernel
    def k_sense(nc: int, nf: int, nb: int,
                world_w: float, world_h: float, wrap: int, daylight: float):
        half_w = world_w * 0.5
        half_h = world_h * 0.5
        pi = ti.math.pi

        for i in range(nc):
            if _calive[i] == 0:
                continue

            sp_idx = _cspecies_idx[i]
            night_mult = _sp_night_vision[sp_idx]
            vision_mult = night_mult + (1.0 - night_mult) * daylight
            vision = _cvision[i] * vision_mult
            heading = _cheading[i]
            cx = _cx[i]
            cy = _cy[i]

            # Biome lookup
            speed_mult = 1.0
            biome_danger = 0.0
            for b in range(nb):
                dx = cx - _bx[b]
                dy = cy - _by[b]
                if ti.sqrt(dx * dx + dy * dy) < _bradius[b]:
                    speed_mult = _bspeed_mult[b]
                    biome_danger = _benergy_drain[b]
                    break
            _cbiome_speed[i] = speed_mult
            _cbiome_danger[i] = biome_danger

            # Nearest food
            best_food_dist = ti.f32(1e9)
            best_food_angle = ti.f32(0.0)
            can_plants = _sp_can_eat_plants[sp_idx]
            can_own_corpse = _sp_can_eat_own_corpse[sp_idx]
            can_other_corpse = _sp_can_eat_other_corpse[sp_idx]

            for fi in range(nf):
                if _falive[fi] == 0:
                    continue
                if _fis_corpse[fi] == 1:
                    same_sp = ti.i32(_fspecies_idx[fi] == sp_idx)
                    if same_sp != 0 and can_own_corpse == 0:
                        continue
                    if same_sp == 0 and can_other_corpse == 0:
                        continue
                else:
                    if can_plants == 0:
                        continue

                dx = _fx[fi] - cx
                dy = _fy[fi] - cy
                if wrap != 0:
                    if ti.abs(dx) > half_w:
                        dx -= copysign(world_w, dx)
                    if ti.abs(dy) > half_h:
                        dy -= copysign(world_h, dy)
                dist = ti.sqrt(dx * dx + dy * dy)
                if dist < best_food_dist and dist <= vision:
                    best_food_dist = dist
                    angle = ti.atan2(dy, dx) - heading
                    best_food_angle = (angle + pi) % (2.0 * pi) - pi

            food_d = ti.f32(1.0)
            food_a = ti.f32(0.0)
            if best_food_dist <= vision:
                food_d = best_food_dist / vision
                food_a = best_food_angle / pi

            # Wall distance
            wall_d = ti.f32(1.0)
            if wrap == 0:
                min_wall = ti.min(cx, world_w - cx, cy, world_h - cy)
                wall_d = ti.min(min_wall / vision, 1.0)

            # Energy
            base_e = _sp_base_energy[sp_idx]
            energy_norm = ti.min(_cenergy[i] / base_e, 1.0)

            # Nearest creature
            best_cr_dist = ti.f32(1e9)
            best_cr_angle = ti.f32(0.0)
            for j in range(nc):
                if j == i or _calive[j] == 0:
                    continue
                dx = _cx[j] - cx
                dy = _cy[j] - cy
                if wrap != 0:
                    if ti.abs(dx) > half_w:
                        dx -= copysign(world_w, dx)
                    if ti.abs(dy) > half_h:
                        dy -= copysign(world_h, dy)
                dist = ti.sqrt(dx * dx + dy * dy)
                if dist < best_cr_dist and dist <= vision:
                    best_cr_dist = dist
                    angle = ti.atan2(dy, dx) - heading
                    best_cr_angle = (angle + pi) % (2.0 * pi) - pi

            cr_d = ti.f32(1.0)
            cr_a = ti.f32(0.0)
            if best_cr_dist <= vision:
                cr_d = best_cr_dist / vision
                cr_a = best_cr_angle / pi

            # Speed
            max_spd = ti.max(_cmax_speed[i], 0.01)
            speed_norm = ti.min(_cspeed[i] / max_spd, 1.0)

            # Nearest threat
            best_thr_dist = ti.f32(1e9)
            best_thr_angle = ti.f32(0.0)
            my_sp = sp_idx
            for j in range(nc):
                if j == i or _calive[j] == 0:
                    continue
                osp = _cspecies_idx[j]
                can_atk_other = _sp_can_attack_other[osp]
                can_atk_own = _sp_can_attack_own[osp]
                if can_atk_other == 0 and can_atk_own == 0:
                    continue
                same_species = ti.i32(osp == my_sp)
                if same_species != 0 and can_atk_own == 0:
                    continue
                if same_species == 0 and can_atk_other == 0:
                    continue

                dx = _cx[j] - cx
                dy = _cy[j] - cy
                if wrap != 0:
                    if ti.abs(dx) > half_w:
                        dx -= copysign(world_w, dx)
                    if ti.abs(dy) > half_h:
                        dy -= copysign(world_h, dy)
                dist = ti.sqrt(dx * dx + dy * dy)
                if dist < best_thr_dist and dist <= vision:
                    best_thr_dist = dist
                    angle = ti.atan2(dy, dx) - heading
                    best_thr_angle = (angle + pi) % (2.0 * pi) - pi

            thr_d = ti.f32(1.0)
            thr_a = ti.f32(0.0)
            if best_thr_dist <= vision:
                thr_d = best_thr_dist / vision
                thr_a = best_thr_angle / pi

            # Biome sensors
            biome_speed_norm = ti.min(ti.max((speed_mult - 0.3) / 1.2, 0.0), 1.0)
            biome_danger_norm = ti.min(biome_danger, 1.0)

            # Write
            _cinputs[i, 0] = food_d
            _cinputs[i, 1] = food_a
            _cinputs[i, 2] = wall_d
            _cinputs[i, 3] = energy_norm
            _cinputs[i, 4] = cr_d
            _cinputs[i, 5] = cr_a
            _cinputs[i, 6] = speed_norm
            _cinputs[i, 7] = thr_d
            _cinputs[i, 8] = thr_a
            _cinputs[i, 9] = ti.min(_cunder_attack[i], 1.0)
            _cinputs[i, 10] = biome_speed_norm
            _cinputs[i, 11] = biome_danger_norm

    @ti.kernel
    def k_brain_forward(n: int):
        for i in range(n):
            if _calive[i] == 0:
                continue
            for h in range(NN_HIDDEN_SIZE):
                acc = _cb1[i, h]
                for inp in range(NN_INPUT_SIZE):
                    acc += _cinputs[i, inp] * _cW1[i, inp, h]
                _chidden[i, h] = ti.tanh(acc)
            for o in range(NN_OUTPUT_SIZE):
                acc = _cb2[i, o]
                for h in range(NN_HIDDEN_SIZE):
                    acc += _chidden[i, h] * _cW2[i, h, o]
                _coutputs[i, o] = ti.tanh(acc)

    @ti.kernel
    def k_think_and_act(n: int):
        pi = ti.math.pi
        for i in range(n):
            if _calive[i] == 0:
                continue
            turn = _coutputs[i, 0] * pi
            _clast_turn[i] = ti.abs(turn)
            _cheading[i] = (_cheading[i] + turn) % (2.0 * pi)
            raw_thrust = (_coutputs[i, 1] + 1.0) / 2.0
            thrust = MIN_THRUST_FRACTION + (1.0 - MIN_THRUST_FRACTION) * raw_thrust
            _cspeed[i] = thrust * _cmax_speed[i]

    @ti.kernel
    def k_physics(nc: int, nh: int,
                  dt: float, world_w: float, world_h: float, wrap: int):
        for i in range(nc):
            if _calive[i] == 0:
                continue
            sp_idx = _cspecies_idx[i]
            speed_mult = _cbiome_speed[i]

            cos_h = ti.cos(_cheading[i])
            sin_h = ti.sin(_cheading[i])
            mx = cos_h * _cspeed[i] * speed_mult * dt * 60.0
            my = sin_h * _cspeed[i] * speed_mult * dt * 60.0
            _cx[i] += mx
            _cy[i] += my
            _cdist_traveled[i] += ti.sqrt(mx * mx + my * my)

            ecpt = _sp_energy_cost[sp_idx]
            eff = _cefficiency[i]
            move_cost = _cspeed[i] * (1.0 / eff) * ecpt * dt * 60.0
            idle_cost = 0.05 * (1.0 + _clast_turn[i]) * dt * 60.0
            _cenergy[i] -= move_cost + idle_cost

            tc = _sp_turn_cost[sp_idx]
            if tc > 0.0:
                _cenergy[i] -= _clast_turn[i] * tc * dt * 60.0

            biome_drain = _cbiome_danger[i]
            if biome_drain > 0.0:
                _cenergy[i] -= biome_drain * dt * 60.0

            for hi in range(nh):
                dx = _cx[i] - _hx[hi]
                dy = _cy[i] - _hy[hi]
                dist = ti.sqrt(dx * dx + dy * dy)
                if dist < _hradius[hi]:
                    intensity = (1.0 - dist / _hradius[hi]) * _hdamage[hi]
                    _cenergy[i] -= intensity * dt * 60.0

            _cage[i] += dt
            if _cbreed_cooldown[i] > 0.0:
                _cbreed_cooldown[i] -= dt

            if _cenergy[i] <= 0.0:
                _cenergy[i] = 0.0
                _calive[i] = 0

            if _calive[i] != 0 and _cage[i] >= _clifespan[i]:
                _calive[i] = 0

            r = _cradius[i]
            if wrap != 0:
                if _cx[i] < 0.0:
                    _cx[i] += world_w
                elif _cx[i] > world_w:
                    _cx[i] -= world_w
                if _cy[i] < 0.0:
                    _cy[i] += world_h
                elif _cy[i] > world_h:
                    _cy[i] -= world_h
            else:
                _cx[i] = ti.max(r, ti.min(world_w - r, _cx[i]))
                _cy[i] = ti.max(r, ti.min(world_h - r, _cy[i]))

    @ti.kernel
    def k_combat(nc: int, dt: float, attack_range_mult: float):
        for i in range(nc):
            if _calive[i] == 0:
                continue
            sp_i = _cspecies_idx[i]
            can_atk_other = _sp_can_attack_other[sp_i]
            can_atk_own = _sp_can_attack_own[sp_i]
            if can_atk_other == 0 and can_atk_own == 0:
                continue

            atk_range = _cradius[i] * attack_range_mult
            damage = _sp_attack_damage[sp_i] * dt * 60.0
            steal_frac = _sp_energy_steal[sp_i]

            for j in range(nc):
                if j == i or _calive[j] == 0:
                    continue
                same_sp = ti.i32(_cspecies_idx[j] == sp_i)
                if same_sp != 0 and can_atk_own == 0:
                    continue
                if same_sp == 0 and can_atk_other == 0:
                    continue

                dx = _cx[i] - _cx[j]
                dy = _cy[i] - _cy[j]
                dist = ti.sqrt(dx * dx + dy * dy)
                contact = atk_range + _cradius[j]
                if dist < contact:
                    ti.atomic_sub(_cenergy[j], damage)
                    _cunder_attack[j] = 1.0
                    ti.atomic_add(_cenergy[i], damage * steal_frac)

    @ti.kernel
    def k_collisions(nc: int, nf: int,
                     world_w: float, world_h: float, wrap: int):
        half_w = world_w * 0.5
        half_h = world_h * 0.5

        for i in range(nc):
            if _calive[i] == 0:
                continue
            cr = _cradius[i]
            sp_idx = _cspecies_idx[i]
            can_plants = _sp_can_eat_plants[sp_idx]
            food_mult = _sp_plant_food_mult[sp_idx]
            sp_food_heal = _sp_food_heal[sp_idx]
            can_own_corpse = _sp_can_eat_own_corpse[sp_idx]
            can_other_corpse = _sp_can_eat_other_corpse[sp_idx]

            for fi in range(nf):
                if _falive[fi] == 0:
                    continue
                if _fis_corpse[fi] == 1:
                    same_sp = ti.i32(_fspecies_idx[fi] == sp_idx)
                    if same_sp != 0 and can_own_corpse == 0:
                        continue
                    if same_sp == 0 and can_other_corpse == 0:
                        continue
                else:
                    if can_plants == 0:
                        continue

                dx = _cx[i] - _fx[fi]
                dy = _cy[i] - _fy[fi]
                if wrap != 0:
                    if ti.abs(dx) > half_w:
                        dx -= copysign(world_w, dx)
                    if ti.abs(dy) > half_h:
                        dy -= copysign(world_h, dy)
                dist = ti.sqrt(dx * dx + dy * dy)
                if dist < cr + _fradius[fi]:
                    old = ti.atomic_sub(_falive[fi], 1)
                    if old == 1:
                        energy_gain = _fenergy[fi] * food_mult
                        ti.atomic_add(_cenergy[i], energy_gain)
                        ti.atomic_add(_cfood_eaten[i], 1)
                        if sp_food_heal > 0.0:
                            new_age = _cage[i] - sp_food_heal
                            if new_age < 0.0:
                                new_age = 0.0
                            _cage[i] = new_age
                    else:
                        ti.atomic_add(_falive[fi], 1)

    # Store kernel references
    _k_reset_under_attack = k_reset_under_attack
    _k_age_food = k_age_food
    _k_sense = k_sense
    _k_brain_forward = k_brain_forward
    _k_think_and_act = k_think_and_act
    _k_physics = k_physics
    _k_combat = k_combat
    _k_collisions = k_collisions


# ── ComputeEngine ───────────────────────────────────────────

class ComputeEngine:
    """Manages GPU state and synchronization with Python objects."""

    def __init__(self, use_gpu: bool = True) -> None:
        _ensure_initialized(use_gpu)
        self._species_id_to_idx: dict[str, int] = {}
        self._idx_to_species_id: dict[int, str] = {}
        self._n_creatures: int = 0
        self._n_food: int = 0
        self._n_biomes: int = 0
        self._n_hazards: int = 0

    # ── Species Table ───────────────────────────────────────

    def upload_species(self, registry: "SpeciesRegistry") -> None:
        """Upload species lookup table."""
        self._species_id_to_idx.clear()
        self._idx_to_species_id.clear()

        species_list = registry.all()

        for idx, sp in enumerate(species_list):
            self._species_id_to_idx[sp.id] = idx
            self._idx_to_species_id[idx] = sp.id

            _sp_can_eat_plants[idx] = int(sp.can_eat_plants)
            _sp_plant_food_mult[idx] = sp.plant_food_multiplier
            _sp_can_attack_other[idx] = int(sp.can_attack_other_species)
            _sp_can_attack_own[idx] = int(sp.can_attack_own_species)
            _sp_can_eat_other_corpse[idx] = int(sp.can_eat_other_corpse)
            _sp_can_eat_own_corpse[idx] = int(sp.can_eat_own_corpse)
            _sp_attack_damage[idx] = sp.attack_damage
            _sp_energy_steal[idx] = sp.energy_steal_fraction
            _sp_scavenge_radius[idx] = sp.scavenge_death_radius
            _sp_scavenge_energy[idx] = sp.scavenge_death_energy
            _sp_base_energy[idx] = sp.settings.base_energy
            _sp_energy_cost[idx] = sp.settings.energy_cost_per_thrust
            _sp_turn_cost[idx] = sp.settings.turn_cost
            _sp_food_heal[idx] = sp.settings.food_heal
            _sp_night_vision[idx] = sp.settings.night_vision_multiplier

    def _species_idx(self, species_id: str) -> int:
        return self._species_id_to_idx.get(species_id, 0)

    # ── Environment ─────────────────────────────────────────

    def upload_environment(self, biomes: "list[Biome]", hazards: "list[Hazard]") -> None:
        from pangea.config import BIOME_ENERGY_DRAIN

        nb = min(len(biomes), GPU_MAX_BIOMES)
        self._n_biomes = nb
        for idx in range(nb):
            b = biomes[idx]
            _bx[idx] = b.x
            _by[idx] = b.y
            _bradius[idx] = b.radius
            _bspeed_mult[idx] = b.speed_multiplier
            _benergy_drain[idx] = BIOME_ENERGY_DRAIN.get(b.biome_type, 0.0)

        nh = min(len(hazards), GPU_MAX_HAZARDS)
        self._n_hazards = nh
        for idx in range(nh):
            h = hazards[idx]
            _hx[idx] = h.x
            _hy[idx] = h.y
            _hradius[idx] = h.radius
            _hdamage[idx] = h.damage_rate

    # ── Creature Upload/Download ────────────────────────────

    def upload_creatures(self, creatures: "list[Creature]") -> None:
        n = min(len(creatures), GPU_MAX_CREATURES)
        self._n_creatures = n

        for i in range(n):
            c = creatures[i]
            _cx[i] = c.x
            _cy[i] = c.y
            _cheading[i] = c.heading
            _cspeed[i] = c.speed
            _cenergy[i] = c.energy
            _calive[i] = int(c.alive)
            _cage[i] = c.age
            _cfood_eaten[i] = c.food_eaten
            _cunder_attack[i] = c.under_attack
            _clast_turn[i] = c.last_turn
            _cdist_traveled[i] = c.distance_traveled
            _cdeath_processed[i] = int(c.death_processed)
            _cbreed_cooldown[i] = c.breed_cooldown

            sp_id = c.dna.species_id if c.dna else ""
            _cspecies_idx[i] = self._species_idx(sp_id)

            _cmax_speed[i] = c.dna.max_speed
            _cradius[i] = c.dna.effective_radius
            _cvision[i] = c.dna.effective_vision
            _cefficiency[i] = c.dna.effective_efficiency
            _clifespan[i] = c.dna.effective_lifespan

            brain = c.brain
            for r in range(NN_INPUT_SIZE):
                for col in range(NN_HIDDEN_SIZE):
                    _cW1[i, r, col] = brain.W1[r, col]
            for h in range(NN_HIDDEN_SIZE):
                _cb1[i, h] = brain.b1[h]
            for r in range(NN_HIDDEN_SIZE):
                for col in range(NN_OUTPUT_SIZE):
                    _cW2[i, r, col] = brain.W2[r, col]
            for o in range(NN_OUTPUT_SIZE):
                _cb2[i, o] = brain.b2[o]

    def download_creatures(self, creatures: "list[Creature]") -> None:
        n = min(len(creatures), self._n_creatures)
        for i in range(n):
            c = creatures[i]
            c.x = float(_cx[i])
            c.y = float(_cy[i])
            c.heading = float(_cheading[i])
            c.speed = float(_cspeed[i])
            c.energy = float(_cenergy[i])
            c.alive = bool(_calive[i])
            c.age = float(_cage[i])
            c.food_eaten = int(_cfood_eaten[i])
            c.under_attack = float(_cunder_attack[i])
            c.last_turn = float(_clast_turn[i])
            c.distance_traveled = float(_cdist_traveled[i])
            c.death_processed = bool(_cdeath_processed[i])
            c.breed_cooldown = float(_cbreed_cooldown[i])

    # ── Food Upload/Download ────────────────────────────────

    def upload_food(self, food_list: "list[Food]") -> None:
        n = min(len(food_list), GPU_MAX_FOOD)
        self._n_food = n

        for i in range(n):
            f = food_list[i]
            _fx[i] = f.x
            _fy[i] = f.y
            _fenergy[i] = f.energy
            _fradius[i] = f.radius
            _fage[i] = f.age
            _flifetime[i] = f.lifetime
            _fis_corpse[i] = int(f.is_corpse)
            sp_id = f.species_id if f.species_id else ""
            _fspecies_idx[i] = self._species_idx(sp_id)
            _falive[i] = 1

    def download_food_compacted(self, food_list: "list[Food]") -> "list[Food]":
        result = []
        n = min(len(food_list), self._n_food)
        for i in range(n):
            if int(_falive[i]) == 1:
                f = food_list[i]
                f.age = float(_fage[i])
                result.append(f)
        return result

    # ── Kernel Dispatch ─────────────────────────────────────

    def run_frame(
        self,
        dt: float,
        world_w: float,
        world_h: float,
        wrap: bool,
        daylight: float,
    ) -> None:
        """Run all GPU kernels for one simulation frame."""
        nc = self._n_creatures
        nf = self._n_food
        nb = self._n_biomes
        nh = self._n_hazards
        w = int(wrap)

        _k_reset_under_attack(nc)
        _k_age_food(nf, dt)
        _k_sense(nc, nf, nb, world_w, world_h, w, daylight)
        _k_brain_forward(nc)
        _k_think_and_act(nc)
        _k_physics(nc, nh, dt, world_w, world_h, w)
        _k_combat(nc, dt, CARNIVORE_ATTACK_RANGE)
        _k_collisions(nc, nf, world_w, world_h, w)
