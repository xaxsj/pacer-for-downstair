__all__ = ['flags', 'summation']

class Flags(object):
    def __init__(self, items):
        for key, val in items.items():
            setattr(self,key,val)

flags = Flags({
    'test': False, 
    'debug': False,
    # 固定路径（世界坐标 JSON）。非空时 TrajGenerator 用它替代程序化随机轨迹。
    'path_file': '',
    # 无窗口自动录制：跑 record_frames 步后把状态导成 pkl 并退出（供本地渲染）。
    'auto_record': False,
    'record_frames': 300,
    # 同时用 Isaac Gym 相机传感器录 mp4（无需窗口）。
    'auto_record_video': True,
    # 固定出生点 (x, y)：用于"从平台顶出发"的演示；None 表示随机采样。
    'start_xy': None,
    # 地形文件名（data/mesh/<mesh_name>.pkl）；空 = 用代码里的默认值。
    'mesh_name': '',
    # 固定初始朝向（弧度，绕 z 轴叠加到参考姿态上）；None 表示不改。
    'start_yaw': None,
    })
