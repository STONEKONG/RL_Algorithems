
# Grid world game
An easy example of grid world game based on deep q networks. In this world, the agent learns to search gold and elude fire.
<img src=./grid_world_game.png>

### Train

```bash
export PYTHONPATH="$PWD" python3 QDN/deep_q_networks.py
```
### Generate pb file
```bash
python3 generate_pb.py
```
### Test
```bash
export PYTHONPATH="$PWD" python3 DQN/test.py 
```

# Rebot arm
An easy example of rebot arm based on deep deterministic policy gradient. In this world, the agent learns to grab boxes.

<img src=./rebot_arm.png>

### Train
```bash
export PYTHONPATH="$PWD" DDPG/deep_deterministic_policy_gradient.py
```




