import math

# --- Constants & thermo ---
R = 8.314
T_REF = 298.0
KC_REF = 5.9e-3
DH = 57200.0  # J per mol reaction (N2O4 -> 2 NO2)

def to_pa(P):
    return P*101325.0 if P < 2000.0 else P

def Kc_vant_hoff(T, Kc_ref=KC_REF, T_ref=T_REF, dH=DH):
    return Kc_ref * math.exp(-(dH/R) * (1.0/T - 1.0/T_ref))

def Ctot(T, P_pa):
    return P_pa / (R * T)

def y_NO2_from_eq(T, P_pa):
    Kc = Kc_vant_hoff(T)
    C = Ctot(T, P_pa)
    gamma = Kc / C
    a, b, c = 1.0, gamma, -gamma
    disc = b*b - 4*a*c
    y = (-b + math.sqrt(disc)) / (2*a)
    return max(0.0, min(1.0, y))

def Uc_free_fall(Tm, Tb, Tt, H, gamma=0.2, g=9.81):
    alpha = 1.0 / Tm
    dT = max(1e-9, Tb - Tt)
    return gamma * math.sqrt(alpha * g * dT * H)

def film_h(lambda_, H, frac):
    return lambda_ / (frac * H)

def post_compute(Tm, H, Tb, Tt, P, gamma, lambda_, frac_b, frac_t):
    P_pa = to_pa(P)
    y_b = y_NO2_from_eq(Tb, P_pa)
    y_t = y_NO2_from_eq(Tt, P_pa)
    y_m = y_NO2_from_eq(Tm, P_pa)

    C_m = Ctot(Tm, P_pa)
    U_c = Uc_free_fall(Tm, Tb, Tt, H, gamma=gamma, g=9.81)
    Ndot = C_m * U_c

    h_b = film_h(lambda_, H, frac_b)
    h_t = film_h(lambda_, H, frac_t)

    # Corrected: divide DH by 2 because y is NO2 mole fraction
    q_ab = (abs(DH)/2.0) * Ndot * max(0.0, (y_b - y_m))
    q_re = (abs(DH)/2.0) * Ndot * max(0.0, (y_m - y_t))

    q_s_b = h_b * (Tb - Tm)
    q_s_t = h_t * (Tm - Tt)

    q_in = q_ab + q_s_b
    q_out = q_re + q_s_t
    q_total = 0.5 * (q_in + q_out)
    Nu = q_total * H / (lambda_ * max(1e-9, (Tb - Tt)))

    return {
        "Tm [K]": Tm,
        "y_b": y_b, "y_m": y_m, "y_t": y_t,
        "C_tot [mol/m^3]": C_m,
        "U_c [m/s]": U_c,
        "Ndot [mol/m^2/s]": Ndot,
        "h_b [W/m^2/K]": h_b, "h_t [W/m^2/K]": h_t,
        "q_ab [W/m^2]": q_ab, "q_re [W/m^2]": q_re,
        "q_s,b [W/m^2]": q_s_b, "q_s,t [W/m^2]": q_s_t,
        "q_in [W/m^2]": q_in, "q_out [W/m^2]": q_out,
        "q_total [W/m^2]": q_total, "Nu [-]": Nu
    }

def solve(H, Tb, Tt, P, gamma, lambda_, frac_b=0.06, frac_t=0.04, tol=1e-6, max_iter=200):
    P_pa = to_pa(P)
    y_b = y_NO2_from_eq(Tb, P_pa)
    y_t = y_NO2_from_eq(Tt, P_pa)
    h_b = film_h(lambda_, H, frac_b)
    h_t = film_h(lambda_, H, frac_t)

    def F(Tm):
        y_m = y_NO2_from_eq(Tm, P_pa)
        C_m = Ctot(Tm, P_pa)
        U_c = Uc_free_fall(Tm, Tb, Tt, H, gamma=gamma)
        Ndot = C_m * U_c
        q_ab = (abs(DH)/2.0) * Ndot * max(0.0, (y_b - y_m))
        q_re = (abs(DH)/2.0) * Ndot * max(0.0, (y_m - y_t))
        q_s_b = h_b * (Tb - Tm)
        q_s_t = h_t * (Tm - Tt)
        return (q_ab + q_s_b) - (q_re + q_s_t)

    lo, hi = Tt + 1e-6, Tb - 1e-6
    f_lo, f_hi = F(lo), F(hi)
    if f_lo * f_hi > 0:
        for theta in [0.25, 0.5, 0.75]:
            mid = lo + theta*(hi - lo)
            f_mid = F(mid)
            if f_lo * f_mid <= 0:
                hi, f_hi = mid, f_mid
                break
            if f_mid * f_hi <= 0:
                lo, f_lo = mid, f_mid
                break
    for _ in range(max_iter):
        mid = 0.5*(lo + hi)
        f_mid = F(mid)
        if abs(f_mid) < tol or (hi - lo) < 1e-6:
            Tm_star = mid
            break
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    else:
        Tm_star = 0.5*(lo + hi)
    res = post_compute(Tm_star, H, Tb, Tt, P, gamma, lambda_, frac_b, frac_t)
    return res

# --- INPUTS (edit here) ---
H = 0.06088       # m
Tb = 350.0       # K
Tt = 300.0       # K
P  = 1.0         # atm (or Pa if >=2000)
gamma = 0.2      # calibration factor
lambda_ = 0.026  # W/(m K)
frac_b, frac_t = 0.06, 0.04 # normalized thickness for the bottom/top film

# --- Run ---
res = solve(H, Tb, Tt, P, gamma, lambda_, frac_b, frac_t)

print(f"********* Double-film model results: Tb = {Tb} K, P = {P} atm *********")
print(f"Nu : {res['Nu [-]']:.6g}")
