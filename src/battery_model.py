# battery_model.py
# 实现 SOC–Tb 的连续时间动力系统
import math


class BatteryModel:
    """
    连续时间电池模型
    内部状态变量：
        SOC : State of Charge（0~1）
        Tb  : 电池内部平均温度（K）
    """

    def __init__(
        self,
        SOC0=1.0,
        Tb0=298.15,
        E0=18.0 * 3600,     # 标称能量（J），例如 18Wh
        C_th=60.0,          # 电池等效热容（J/K）
        h=1.0,              # 散热系数（W/K）
        eta_heat=0.85,      # 功耗转化为热的比例
        T_ref=298.15,       # 参考温度（K）
        alpha=0.03,         # 低温容量敏感系数
        aging_loss=0.1      # 老化损失比例 δ
    ):
        # 状态变量
        self.SOC = SOC0
        self.Tb = Tb0

        # 电池与热学参数
        self.E0 = E0
        self.C_th = C_th
        self.h = h
        self.eta_heat = eta_heat

        # 温度与老化参数
        self.T_ref = T_ref
        self.alpha = alpha
        self.fA = 1.0 - aging_loss  # 老化修正因子 f_A

    def temperature_factor(self):
        """
        温度修正因子 f_T(Tb)
        低温时有效容量指数下降，高温不额外增加
        """
        if self.Tb >= self.T_ref:
            return 1.0
        else:
            return math.exp(-self.alpha * (self.T_ref - self.Tb))

    def effective_energy(self):
        """
        温度 + 老化修正后的有效可用能量 E_eff
        """
        return self.E0 * self.temperature_factor() * self.fA

    def step(self, P_tot, T_amb, dt):
        """
        连续时间模型的显式时间推进（Euler）
        参数：
            P_tot : 当前瞬时总功耗（W）
            T_amb : 环境温度（K）
            dt    : 时间步长（s）
        """

        # ===== 电池温度动力学 =====
        # C_th * dTb/dt = η_heat * P_tot - h * (Tb - T_amb)
        dTb_dt = (
            self.eta_heat * P_tot
            - self.h * (self.Tb - T_amb)
        ) / self.C_th

        self.Tb += dTb_dt * dt

        # ===== SOC 动力学 =====
        # dSOC/dt = - P_tot / E_eff
        E_eff = self.effective_energy()

        if E_eff > 0:
            dSOC_dt = - P_tot / E_eff
        else:
            dSOC_dt = 0.0

        self.SOC += dSOC_dt * dt

        # 数值稳定性：SOC 不允许越界
        self.SOC = max(0.0, min(1.0, self.SOC))
