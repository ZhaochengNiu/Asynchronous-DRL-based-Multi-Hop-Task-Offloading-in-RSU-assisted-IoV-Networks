import numpy as np
from env.config import VehicularEnvConfig


class Greedy(object):
    """ 贪心算法实现思路 """


    def __init__(self):
        # # 初始化方法，配置环境
        self.config=VehicularEnvConfig()
        pass

    def choose_action(self, state,function) -> int:
        """ 根据任务队列选择合适的卸载节点 """
        action_list = []  # 存储选择的动作
        State = state  # 当前状态
        Function = function   # 任务函数
        function_size = []   # 函数的数据大小
        # 获取每个任务函数的数据大小
        for i in range(self.config.rsu_number):
            function_size.append(Function[i].get_task_datasize())
        # 选择动作
        for i in range(len(function_size)):
            # 找到当前状态中任务队列数最少的索引
            min_index = np.argmin(State)
            # 将任务分配给任务队列数最少的节点
            action_list.append(min_index+1)
            # 更新当前状态，加上相应的任务数据大小
            State[i]=State[i]+function_size[i]
        # 将三个节点的动作编码为单个动作
        x=action_list[0]
        y=action_list[1]
        z=action_list[2]
        action= (x - 1) + (y - 1) * 14 + (z - 1) * 14 * 14 + 1
        return action
