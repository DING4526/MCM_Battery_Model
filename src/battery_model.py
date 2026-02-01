# battery_model.py
#
# 电池 SOC–温度 Tb 连续时间动力系统
#
# 支持两种数值积分方法：
#   - Euler（显式一阶）
#   - RK4（四阶 Runge–Kutta）
#
# 用于仿真手机在不同使用状态下的电量与温度演化

import math


class BatteryModel:
    """
    连续时间电池模型

    状态变量：
        SOC : 电池荷电状态（0~1）
        Tb  : 电池内部平均温度（K）

    支持积分方法：
        method="euler" ：显式欧拉法
        method="rk4"   ：四阶 Runge–Kutta
    """

    def __init__(
        self,
        SOC0=1.0,
        Tb0=298.15,
        E0=18.0 * 3600,     # 标称能量（J）
        C_th=60.0,         # 等效热容（J/K）
        h=1.0,             # 散热系数（W/K）
        eta_heat=0.85,     # 功耗转化为热的比例
        T_ref=298.15,     # 参考温度（K）
        alpha=0.03,       # 低温容量敏感系数
        aging_loss=0.1    # 老化损失比例 δ
    ):

        # ===== 状态变量 =====
        self.SOC = SOC0
        self.Tb = Tb0

        # ===== 电池与热学参数 =====
        self.E0 = E0
        self.C_th = C_th
        self.h = h
        self.eta_heat = eta_heat

        # ===== 温度与老化 =====
        self.T_ref = T_ref
        self.alpha = alpha
        self.fA = 1.0 - aging_loss   # 老化修正因子


    # =====================================================
    # 物理子模型
    # =====================================================

    def temperature_factor(self, Tb):
        """
        温度修正因子 f_T(Tb)

        当温度低于参考值时，容量指数下降；
        高温不额外增加容量。
        """

        if Tb >= self.T_ref:
            return 1.0
        else:
            return math.exp(-self.alpha * (self.T_ref - Tb))

    def effective_energy(self, Tb):
        """
        计算温度 + 老化修正后的有效可用能量
        """

        return self.E0 * self.temperature_factor(Tb) * self.fA

    def _derivatives(self, SOC, Tb, P_tot, T_amb):
        """
        连续动力学方程

        返回：
            dSOC/dt, dTb/dt
        """

        # ===== 温度动力学 =====
        # C_th * dTb/dt = η_heat * P - h * (Tb - T_amb)
        dTb_dt = (
            self.eta_heat * P_tot
            - self.h * (Tb - T_amb)
        ) / self.C_th

        # ===== SOC 动力学 =====
        # dSOC/dt = - P / E_eff
        E_eff = self.effective_energy(Tb)

        if E_eff > 0:
            dSOC_dt = - P_tot / E_eff
        else:
            dSOC_dt = 0.0

        return dSOC_dt, dTb_dt


    # =====================================================
    # 时间推进接口（Euler / RK4）
    # =====================================================

    def step(self, P_tot, T_amb, dt, method="euler"):
        """
        电池状态推进 dt 秒

        参数：
            P_tot : 当前总功耗（W）
            T_amb : 环境温度（K）
            dt    : 时间步长（s）
            method:
                "euler" —— 显式欧拉
                "rk4"   —— 四阶 Runge–Kutta
        """

        if method == "euler":
            self._step_euler(P_tot, T_amb, dt)

        elif method == "rk4":
            self._step_rk4(P_tot, T_amb, dt)

        else:
            raise ValueError("method 必须为 'euler' 或 'rk4'")

        # 数值稳定性保护：SOC 限制在 [0,1]
        self.SOC = max(0.0, min(1.0, self.SOC))


    # -----------------------------------------------------
    # 显式 Euler 法（一阶）
    # -----------------------------------------------------

    def _step_euler(self, P_tot, T_amb, dt):
        """
        显式欧拉积分
        """

        dSOC, dTb = self._derivatives(self.SOC, self.Tb, P_tot, T_amb)

        self.SOC += dSOC * dt
        self.Tb += dTb * dt


    # -----------------------------------------------------
    # 四阶 Runge–Kutta
    # -----------------------------------------------------

    def _step_rk4(self, P_tot, T_amb, dt):
        """
        四阶 Runge–Kutta 时间积分
        """

        SOC0 = self.SOC
        Tb0 = self.Tb

        k1_SOC, k1_Tb = self._derivatives(SOC0, Tb0, P_tot, T_amb)

        k2_SOC, k2_Tb = self._derivatives(
            SOC0 + 0.5 * dt * k1_SOC,
            Tb0 + 0.5 * dt * k1_Tb,
            P_tot,
            T_amb,
        )

        k3_SOC, k3_Tb = self._derivatives(
            SOC0 + 0.5 * dt * k2_SOC,
            Tb0 + 0.5 * dt * k2_Tb,
            P_tot,
            T_amb,
        )

        k4_SOC, k4_Tb = self._derivatives(
            SOC0 + dt * k3_SOC,
            Tb0 + dt * k3_Tb,
            P_tot,
            T_amb,
        )

        self.SOC += dt / 6.0 * (k1_SOC + 2*k2_SOC + 2*k3_SOC + k4_SOC)
        self.Tb += dt / 6.0 * (k1_Tb + 2*k2_Tb + 2*k3_Tb + k4_Tb)
