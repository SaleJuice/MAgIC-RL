'''
FilePath: /MAgIC-RL/magic_rl/utils/utils.py
Date: 2022-09-07 13:28:25
LastEditTime: 2022-09-14 21:44:46
Author: Xiaozhu Lin
E-Mail: linxzh@shanghaitech.edu.cn
Institution: MAgIC Lab, ShanghaiTech University, China
SoftWare: VSCode
'''

from prettytable import PrettyTable


class LinearAnneal:
    """Linear Annealing Schedule.

    Args:
        start: The initial value of epsilon.
        end: The final value of epsilon.
        duration: The number of anneals from start value to end value.

    """

    def __init__(self, start: float, end: float, duration: int):
        self.val = start
        self.min = end
        self.duration = duration

    def anneal(self):
        self.val = max(self.min, self.val - (self.val - self.min) / self.duration)


def formate_args_as_table(args):
    args_table = PrettyTable()
    args_table.field_names = ["arguments", "content"]
    
    for arg in vars(args):
        args_table.add_row([arg, getattr(args, arg)])
    args_table.add_autoindex("idx")

    args_table.align["idx"] = "r"
    args_table.align["arguments"] = "c"
    args_table.align["content"] = "c"

    return args_table


if __name__ == "__main__":
    pass
