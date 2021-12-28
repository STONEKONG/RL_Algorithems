import numpy as np
import pyglet
import yaml
# pyglet.clock.set_fps_limit(10000)
class game_env():

    def __init__(self, env_config):

        self.win_size = env_config["win_size"]
        self.action_dim = env_config["action_dim"]
        self.angle_s = env_config["angle_scale"]
        self.arm_length = env_config["arm_length"]
        self.point_size = env_config["point_size"]
        self.ms = env_config["measuring_scale"]
        self.gct = env_config["grab_counter_thre"]
        self.ga = env_config["grab_accuary"]
        self.arm_info = np.zeros((self.action_dim, 3))
        self.center_coord = np.array(self.win_size) / 2
        self.viewer = None
      
    def step(self, action):

        action = np.clip(action, -1, 1)
        self.arm_info[:, 0] += action * self.angle_s
        self.arm_info[:, 0] %= 2 * np.pi  
        self.arm_coord()
        state = self.get_state()
        reward = self.reward_function(state)

        return state, reward, self.get_point


    def random_point(self):

        r_x, r_y = 2 * np.random.rand(2) - 1
        max_arm_len = np.sum(self.arm_length)
        max_x = min(max_arm_len, self.win_size[1]/2)
        max_y = min(max_arm_len, self.win_size[0]/2)
        p_x = r_x * max_x
        scope_y = (max_arm_len ** 2 - p_x ** 2) ** 0.5
        if scope_y > max_y:
            scope_y = max_y
        p_y = r_y * scope_y

        return np.array([self.win_size[1]/2+p_x, self.win_size[0]/2+p_y])


    def reset(self):

        self.get_point = False
        self.grab_counter = 0
        self.arm_info[:, 0] = np.random.rand(self.action_dim) * np.pi * 2
        self.arm_coord()
        self.point_info = self.random_point()
        state = self.get_state()

        return state 

    def arm_coord(self):

        arm_dxdy = np.tile(self.arm_length, (2,1)) * \
                   np.array([np.cos(self.arm_info[:, 0]), 
                             np.sin(self.arm_info[:, 0])])
        arm_dxdy = arm_dxdy.T
        for i in range(self.action_dim-1):
            arm_dxdy[i+1] += arm_dxdy[i]
        self.arm_info[:, 1:] = arm_dxdy + self.center_coord
       
    def get_state(self):

        arms_end = self.arm_info[:, 1:3]
        arms_dis = np.ravel(arms_end - self.point_info) / self.ms
        point_dis = (self.center_coord - self.point_info) / self.ms
        in_point = 1 if self.grab_counter > 0 else 0 
        state = np.hstack([in_point, arms_dis, point_dis])

        return state
        
    def reward_function(self, state):

        end_dist = state[-4:-2]
        euc_dist = np.sqrt(np.sum(np.square(end_dist)))
        reward = -euc_dist 
        if euc_dist < (1 - self.ga) * self.point_size / self.ms and (not self.get_point):
            reward += 1.0
            self.grab_counter += 1
            if self.grab_counter > self.gct:
                reward += 10.
                self.get_point = True
        elif euc_dist > (1 - self.ga) * self.point_size / self.ms:
            self.grab_counter = 0
            self.get_point = False

        return reward

    def render(self):

        if self.viewer is None:
            self.viewer = Viewer(*self.win_size, self.arm_length, self.point_size)
        self.viewer.render(self.point_info, self.arm_info)

class Viewer(pyglet.window.Window):

    fps_display = pyglet.clock.ClockDisplay()
    batch = pyglet.graphics.Batch()
    bar_thc = 5
    def __init__(self, width, height, arm_length, point_s):
        super(Viewer, self).__init__(width, height, resizable=False, caption="Rebort Arm", vsync=False)
        self.set_location(x=80, y=10)

        bk_color = [1] * 4
        pyglet.gl.glClearColor(*bk_color)
        self.arm_length = arm_length
        self.point_s = point_s
        self.center_coord = np.array([width/2, height/2])
        self.arm_num = len(arm_length)
        self.point = self.batch.add(4, pyglet.gl.GL_QUADS, None, ("v2f", [0]*8), ("c3B", (86, 109, 249)*4))
        self.arms = dict()
        for i in range(self.arm_num):
            armi = self.batch.add(4, pyglet.gl.GL_QUADS, None, ("v2f", [0]*8), ("c3B", (249, 86, 86)*4))
            self.arms['arm' + str(i)] = armi

    def render(self, point_info, arm_info):
        pyglet.clock.tick()
        self.update_arm(point_info, arm_info)
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event("on_draw")
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def update_arm(self, point_info, arm_info):

        point_box = (
            point_info[0] - self.point_s, point_info[1] - self.point_s,
            point_info[0] + self.point_s, point_info[1] - self.point_s,
            point_info[0] + self.point_s, point_info[1] + self.point_s,
            point_info[0] - self.point_s, point_info[1] + self.point_s
            )
        
        self.point.vertices = point_box

        start_coord = self.center_coord
        for i in range(self.arm_num): 
            end_coord = arm_info[i, 1:3]
            armi_coord = (*start_coord, *end_coord)
            
            armi_thick_rad = np.pi / 2 - arm_info[i, 0]
            xi_1 = armi_coord[0] - np.cos(armi_thick_rad) * self.bar_thc 
            yi_1 = armi_coord[1] + np.sin(armi_thick_rad) * self.bar_thc
            xi_2 = armi_coord[0] + np.cos(armi_thick_rad) * self.bar_thc
            yi_2 = armi_coord[1] - np.sin(armi_thick_rad) * self.bar_thc

            xi_3 = armi_coord[2] + np.cos(armi_thick_rad) * self.bar_thc 
            yi_3 = armi_coord[3] - np.sin(armi_thick_rad) * self.bar_thc
            xi_4 = armi_coord[2] - np.cos(armi_thick_rad) * self.bar_thc
            yi_4 = armi_coord[3] + np.sin(armi_thick_rad) * self.bar_thc
            armi_box = (xi_1, yi_1, xi_2, yi_2, xi_3, yi_3, xi_4, yi_4)
            self.arms["arm"+str(i)].vertices = armi_box
            start_coord = end_coord

if __name__ == "__main__":

    config_path = 'config/rebort_arm_config.yaml'    

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f)
    env = game_env(config)
    state = env.reset()
    while True:
        a = 2 * np.random.rand(2) - 1
        s_, r, done = env.step(a)
        env.render()
       
