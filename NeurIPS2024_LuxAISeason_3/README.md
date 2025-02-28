
# Overview

- competition url: [https://www.kaggle.com/competitions/lux-ai-season-3/overview](https://www.kaggle.com/competitions/lux-ai-season-3/overview)
- Github: [Lux-Design-S3](https://github.com/Lux-AI-Challenge/Lux-Design-S3?tab=readme-ov-file)
- Abastract:
  - 2D Map: 24x24. 
    - Whatever is the state of the map at the end of one match is what is used for the next match.
  - Gaming: 5 match, 100 time steps of one match
    - NOTE: 
      - Each match is played with fog of war, where each team can only see what their own units can see, with everything else being hidden
      -  It is recommended to explore more in the first match or two before leveraging gained knowledge about the map and opponent behavior to win the latter matches.
    - Units:  
      - ships that can move one tile in 5 directions (center, up, right, down, left) 
      - Units can overlap with other friendly units if they move onto the same tile
    - Move action:
      - All move actions except moving center cost `params.unit_move_cost` energy to perform
    - Sap Actions
      - The sap action lets a unit target a specific tile on the map within a range called `params.unit_sap_range`
      - reduces the energy of each opposition unit -- `params.unit_sap_cost`
      - costing energy -- `unit_sap_cost ` 
      - reduces the energy of any opposition units on the 8 adjacent tiles  -- `params.unit_sap_cost` * `params.unit_sap_dropoff_factor`
      - The $\Delta x$ and $\Delta y$ value magnitudes must both be $<= params.unit_sap_range$
    - Vision: A team's vision is the combined vision of all units on that team
      - read URL
    - Collisions / Energy Void Fields
      - In the event of two or more units from opposing teams occupy the same tile at the end of a turn, the team with the highest aggregate energy among its units on that tile survive, while the units of the opposing teams are removed from the game. If it is a tie, all units are removed from the game.
    - Win Conditions
      - To win a match, the team must have gained more relic points than the other team at the end of the match. If the relic points scores are tied, then the match winner is decided by who has more total unit energy. If that is also tied then the winner is chosen at random.
    - Match Resolution Order
      1. Move all units that have enough energy to move
      2. Execute the sap actions of all units that have enough energy to do so
      3. Resolve collisions and apply energy void fields
      4. Update the energy of all units based on their position (energy fields and nebula tiles)
      5. Spawn units for all teams. Remove units that have less than 0 energy.
      6. Determine the team vision / sensor masks for all teams and mask out observations accordingly
      7. Environment objects like asteroids/nebula tiles/energy nodes move around in space
      8. Compute new team points

# Training 

## V2.0 2025-02-20
1. action adv: 
   1. [stay, up, right, down, left, sap(上), sap(右上), sap(右), sap(右下), sap(下), sap(左下), sap(左), sap(左上)]
   2. action_dim = 5+8=13 
   3. action_out: 0~4: [x, 0, 0]; 5:   [5, x, y]
2. state:
   1. global_map & near_space_map & unit_info

## V2.1 
### 2025-02-26
1. reward update `unitReward`
   1. reference: [ppo-stable-baselines3](https://www.kaggle.com/code/yizhewang3/ppo-stable-baselines3#train.py)
   2. sap reward optimate
   3. random action update
3. V3 MultiAgent prepare

### 2025-02-27
1. Update training planning 
   1. dqn VS random  -> dqn1
   2. random VS dqn  -> dqn2
   3. dqn1 VS dqn2   -> dqn3 
      1. Parameter smooth fusion

### 2025-02-28
1. Update training planning 
   1. dqn VS random  -> dqn1
   2. random VS dqn  -> dqn2
   3. main use 2 model 


