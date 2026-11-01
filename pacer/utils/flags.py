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
    })
