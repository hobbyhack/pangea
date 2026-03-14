# Pangea - Feature Backlog

## Environment

- [ ] Hazards/obstacles — static or moving danger zones that drain energy on contact (lava, cold zones)
- [ ] Day/night cycle — vision range shrinks at "night," creatures must adapt to low-visibility periods
- [ ] Seasonal food — food spawn rate oscillates over time (feast/famine cycles)
- [ ] Food decay — uneaten food disappears after N seconds, rewarding faster foragers
- [ ] Predators — NPC predators that hunt creatures, adding survival pressure beyond starvation
- [ ] Terrain/biomes — regions with different movement speed multipliers (water slows, roads speed up)
- [ ] Food clusters — food spawns in clumps vs uniformly, rewarding exploration vs camping

## Genetic / Creature

- [ ] Crossover toggle — enable sexual reproduction (blend two parents' weights) vs mutation-only
- [ ] Speciation threshold — creatures too genetically different can't crossover, encouraging niche species
- [ ] Lifespan gene — a heritable max-age trait (trade longevity vs other stats)
- [ ] Diet specialization — multiple food types with color, creatures evolve preference/efficiency for specific types
- [ ] Reproduction cost — creatures that eat enough can reproduce mid-generation (asexual budding), spending energy to spawn offspring in real-time instead of waiting for generation end
- [ ] Aggression trait — creatures can steal energy from others on contact, with a budget trade-off
- [ ] Camouflage/detection — prey creatures evolve to be harder to sense, predators evolve better detection

## Brain / Sensors

- [ ] More sensor inputs — nearest creature distance/angle, population density, own speed
- [ ] Hidden layer size — make NN architecture itself evolvable (bigger brain = more energy cost)
- [ ] Memory — add recurrent connection (simple hidden state) so creatures can remember
