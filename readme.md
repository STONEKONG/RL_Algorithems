

```bash
cd RL_Algorthems
export PYTHONPATH="$PWD"
```

# Grid world game
An easy example of grid world game based on deep q networks. In this world, the agent learns to search gold and elude fire.

<img src=./grid_world_game.png>

### Train

```bash
python3 DQN/deep_q_networks.py
```
### Generate pb file
```bash
python3 generate_pb.py --output_node_names target_net/Q_value --ckpt_path --save_path 
```
### Test
```bash
python3 DQN/test.py --pb_path 
```

# Rebot arm
An easy example of rebot arm based on deep deterministic policy gradient. In this world, the agent learns to grab boxes.

<img src=./rebot_arm.png>

### Train
```bash
python3 DDPG/deep_deterministic_policy_gradient.py
```

### Generate pb file
```bash
python3 generate_pb.py --output_node_name target_actor/fully_connected_3/Tanh --ckpt_path --save_path 
```

### Test
```bash
python3 python3 DDPG/test.py --pb_path --output_node_name target_actor/fully_connected_3/Tanh --input_node_name state_
```




