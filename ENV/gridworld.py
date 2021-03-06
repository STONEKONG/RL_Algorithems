
import numpy as np 
import itertools
from PIL import Image
import matplotlib.pyplot as plt 
import yaml 


class game_ob():
    def __init__(self, coordinates, size, intensity, channel, reward, name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel 
        self.reward = reward 
        self.name = name 
        
class game_env():
    def __init__(self, env_config):

        self.actions = 4 
        self.objects = []     
        self.size_x = env_config['world_size'][0]
        self.size_y = env_config['world_size'][1]
        self.num_hero = env_config['role_params']['hero_num']
        self.num_fire = env_config['role_params']['fire_num']
        self.num_goal = env_config['role_params']['goal_num']
        self.dency = env_config['reward_params']['dency'] 
        self.reward_goal = env_config['reward_params']['goal'] 
        self.reward_fire = env_config['reward_params']['fire'] 
        self.penalize = env_config['reward_params']['penalize'] 
        self.points = self.generate_points()
        self.reset()

    def reset(self):
        self.objects = []
        for _ in range(self.num_hero):
            obj = game_ob(self.new_position(), 1, 1, 2, None, 'hero')
            self.objects.append(obj)
        for _ in range(self.num_fire):
            obj = game_ob(self.new_position(), 1, 1, 0, -1, 'fire')
            self.objects.append(obj)
        for _ in range(self.num_goal):
            obj = game_ob(self.new_position(), 1, 1, 1, 1, 'goal')
            self.objects.append(obj)
        self.state = self.render_env()
     

    def move_char(self, direction):
        # 0 - up, 1 - down, 2 - left, 3 - right
        hero = self.objects[0]
        heroX = hero.x
        heroY = hero.y
        penalize = 0.
        if direction == 0 and hero.y >= 1:
            hero.y -= 1
        if direction == 1 and hero.y <= self.size_y-2:
            hero.y += 1
        if direction == 2 and hero.x >= 1:
            hero.x -= 1
        if direction == 3 and hero.x <= self.size_x-2:
            hero.x += 1     
        if hero.x == heroX and hero.y == heroY:
            penalize = self.penalize
        self.objects[0] = hero
        return penalize

    def generate_points(self):
        iterables = [range(self.size_x), range(self.size_y)]
        points = []
        for point in itertools.product(*iterables):
            points.append(point)
        return points

    def new_position(self):
        points = self.points.copy()
        current_positions = []
        for ob in self.objects:
            if (ob.x,ob.y) not in current_positions:
                current_positions.append((ob.x, ob.y))
        for pos in current_positions:
            points.remove(pos)
        location = np.random.choice(range(len(points)), replace=False)
        return points[location]

    def check_goal(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        ended = False
        for other in others:
            if hero.x == other.x and hero.y == other.y:
                self.objects.remove(other)
                if other.reward == 1:
                    self.objects.append(game_ob(self.new_position(), 1 ,1 ,1, self.reward_goal, 'goal'))
                else: 
                    self.objects.append(game_ob(self.new_position(), 1, 1, 0, self.reward_fire, 'fire'))
                return other.reward, False
        if ended == False:
            return 0.0, False

    def render_env(self):
        board = np.ones([self.size_y+2, self.size_x+2, 3])
        board[1:-1, 1:-1, :] = 0
        for item in self.objects:
            board[item.y+1:item.y+item.size+1, item.x+1:item.x+item.size+1, item.channel] = item.intensity
        board = Image.fromarray(np.uint8(255 * board))
        board = np.array(board.resize((84, 84), Image.NEAREST))
        return board

    def step(self, action):
        penalty = self.move_char(action) 
        reward,done = self.check_goal()
        state = self.render_env()
        if reward > 0:
            return state, (reward + penalty), done
        else:
            return state, (reward + penalty + self.dency), done


if __name__ == '__main__':


    config_path = 'config/gridworld_config.yaml'

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f)

    env = game_env(config)