# battery_model.py
#
# 温度与老化修正的连续时间电池模型（升级版：加入 OCV–SOC 闭合）
#
# 状态变量：
#   SOC(t) : 荷电状态 (0~1)
#   Tb(t)  : 电池内部平均温度 (K)
#
# 外部输入：
#   P_tot(t) : 手机总功耗 (W)
#   T_amb(t) : 环境温度 (K)
#
# 模型：
#   dSOC/dt = - P_tot / ( V_oc(SOC) * Q_eff(Tb, A) )
#   C_th dTb/dt = eta_heat * P_tot - h (Tb - T_amb)
#
# 其中：
#   V_oc(SOC) = V0 + a1*s + a2*ln(s) + a3*ln(1-s)   (eps 截断)
#   Q_eff(Tb,A) = Q_nom * f_T(Tb) * f_A
#   f_T(Tb) = exp( -alpha * max(T_ref - Tb, 0) )
#   f_A = 1 - aging_loss
#
# 数值积分：
#   - Euler (显式一阶)
#   - RK4   (四阶 Runge–Kutta)

import math
from dataclasses import dataclass


@dataclass
class OCVParams:
    """OCV–SOC 参数组：V_oc(s)=V0 + a1*s + a2*ln(s) + a3*ln(1-s)"""
    V0: float = 3.6
    a1: float = -0.06
    a2: float = 0.16
    a3: float = -0.1
    eps: float = 1e-4  # SOC 截断阈值，避免 ln(0)


class BatteryModel:
    """
    连续时间电池模型（升级版：加入 OCV-SOC 闭合）

    状态变量：
        SOC : 0~1
        Tb  : K

    使用方式（对外接口保持一致）：
        model = BatteryModel(...)
        model.step(P_tot, T_amb, dt, method="euler"/"rk4")
        model.SOC, model.Tb
    """

    def __init__(
        self,
        SOC0: float = 1.0,
        Tb0: float = 298.15,

        # ===== 电池容量（对外只保留这一种标称输入）=====
        # 标称容量范围建议 4–6 Ah（约 4000–6000 mAh）
        capacity_Ah: float = 5.0,     # 标称容量（Ah）

        # ===== 热学参数 =====
        C_th: float = 60.0,           # 等效热容（J/K）
        h: float = 1.0,               # 散热系数（W/K）
        eta_heat: float = 0.85,       # 功耗转化为热的比例

        # ===== 温度与老化 =====
        T_ref: float = 298.15,        # 参考温度（K）
        alpha: float = 0.03,          # 低温容量敏感系数
        aging_loss: float = 0.10,     # 老化损失比例 delta（0~1）

        # ===== OCV 参数 =====
        ocv_params: OCVParams | None = None,

        # ===== 数值安全保护 =====
        v_min: float = 2.5,           # OCV 最小夹紧（避免除零/负电压）
        v_max: float = 4.5,           # OCV 最大夹紧
    ):
        # ---- 状态 ----
        self.SOC = float(SOC0)
        self.Tb = float(Tb0)

        # ---- 热学 ----
        self.C_th = float(C_th)
        self.h = float(h)
        self.eta_heat = float(eta_heat)

        # ---- 温度/老化 ----
        self.T_ref = float(T_ref)
        self.alpha = float(alpha)

        aging_loss = float(aging_loss)
        if not (0.0 <= aging_loss < 1.0):
            raise ValueError("aging_loss 必须在 [0, 1) 范围内")
        self.fA = 1.0 - aging_loss  # f_A = 1 - delta

        # ---- OCV ----
        self.ocv = ocv_params if ocv_params is not None else OCVParams()
        self.v_min = float(v_min)
        self.v_max = float(v_max)

        # ---- 标称容量（库仑）----
        cap_ah = float(capacity_Ah)
        if cap_ah <= 0:
            raise ValueError("capacity_Ah 必须 > 0")
        # 你要求标称容量 4–6Ah：这里不强制卡死，但给出安全检查（可改成 raise）
        if cap_ah < 4.0 or cap_ah > 6.0:
            # 这里选择“容错不报错”，但你也可以改成 raise ValueError(...)
            pass
        self.Q_nom = cap_ah * 3600.0  # Ah -> C

        # ---- 数值稳定性保护 ----
        self.SOC = max(0.0, min(1.0, self.SOC))

    # =====================================================
    # 子模型：温度修正、有效容量、OCV
    # =====================================================

    def temperature_factor(self, Tb: float) -> float:
        """f_T(Tb)=exp(-alpha*(T_ref-Tb)_+)，高温不额外增加容量"""
        if Tb >= self.T_ref:
            return 1.0
        return math.exp(-self.alpha * (self.T_ref - Tb))

    def effective_capacity(self, Tb: float) -> float:
        """Q_eff(Tb,A)=Q_nom*f_T(Tb)*f_A（单位：库仑 C）"""
        return self.Q_nom * self.temperature_factor(Tb) * self.fA

    def voc(self, soc: float) -> float:
        """
        开路电压 OCV：V0 + a1*s + a2*ln(s) + a3*ln(1-s)
        带 eps 截断避免 ln(0)，并做电压夹紧保证稳定。
        """
        eps = self.ocv.eps
        s = max(eps, min(1.0 - eps, soc))

        v = (
            self.ocv.V0
            + self.ocv.a1 * s
            + self.ocv.a2 * math.log(s)
            + self.ocv.a3 * math.log(1.0 - s)
        )

        # 夹紧，避免分母出问题
        if v < self.v_min:
            v = self.v_min
        elif v > self.v_max:
            v = self.v_max
        return v

    # =====================================================
    # 动力学方程
    # =====================================================

    def _derivatives(self, SOC: float, Tb: float, P_tot: float, T_amb: float) -> tuple[float, float]:
        """
        返回 (dSOC/dt, dTb/dt)
        """
        # --- 温度动力学 ---
        dTb_dt = (self.eta_heat * P_tot - self.h * (Tb - T_amb)) / self.C_th

        # --- SOC 动力学（OCV 闭合） ---
        Q_eff = self.effective_capacity(Tb)
        V_oc = self.voc(SOC)

        if Q_eff > 0.0 and V_oc > 0.0:
            dSOC_dt = - P_tot / (V_oc * Q_eff)
        else:
            dSOC_dt = 0.0

        return dSOC_dt, dTb_dt

    # =====================================================
    # 时间推进接口（Euler / RK4）
    # =====================================================

    def step(self, P_tot: float, T_amb: float, dt: float, method: str = "euler"):
        """
        推进 dt 秒

        参数：
            P_tot : 当前总功耗（W）
            T_amb : 环境温度（K）
            dt    : 时间步长（s）
            method: "euler" 或 "rk4"
        """
        if dt <= 0:
            raise ValueError("dt 必须 > 0")

        if method == "euler":
            self._step_euler(P_tot, T_amb, dt)
        elif method == "rk4":
            self._step_rk4(P_tot, T_amb, dt)
        else:
            raise ValueError("method 必须为 'euler' 或 'rk4'")

        # 稳定性保护
        self.SOC = max(0.0, min(1.0, self.SOC))

    def _step_euler(self, P_tot: float, T_amb: float, dt: float):
        dSOC, dTb = self._derivatives(self.SOC, self.Tb, P_tot, T_amb)
        self.SOC += dSOC * dt
        self.Tb += dTb * dt

    def _step_rk4(self, P_tot: float, T_amb: float, dt: float):
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

        self.SOC += dt / 6.0 * (k1_SOC + 2 * k2_SOC + 2 * k3_SOC + k4_SOC)
        self.Tb += dt / 6.0 * (k1_Tb + 2 * k2_Tb + 2 * k3_Tb + k4_Tb)

    # =====================================================
    # 便捷接口（记录/调试）
    # =====================================================

    def get_state(self) -> dict:
        """返回当前状态与关键派生量（便于记录与可视化）"""
        return {
            "SOC": self.SOC,
            "Tb": self.Tb,
            "V_oc": self.voc(self.SOC),
            "Q_eff": self.effective_capacity(self.Tb),
            "f_T": self.temperature_factor(self.Tb),
            "f_A": self.fA,
        }
