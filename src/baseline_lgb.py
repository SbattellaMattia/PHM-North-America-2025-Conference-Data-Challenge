# Dopo scaler.transform, aggiungi physics
from src.feature_physics import add_physics_features

dtr_phys = add_physics_features(dtr_std, snapshot=4)
dva_phys = add_physics_features(dva_std, snapshot=4)

# Usa SOLO physics features (no wide)
phys_cols = ['isentropic_eff', 'increment_T3_T45', 'pressure_ratio',
             'EGT_margin_proxy', 'fuel_eff', 'thrust_proxy', 'speed_ratio']

# Aggrega per ciclo (prendi media se multipli snapshot)
tr_phys_agg = dtr_phys.groupby([id_col, cyc_col]).agg({
    **{f: 'mean' for f in phys_cols},
    **{t: 'first' for t in targets}
}).reset_index()

va_phys_agg = dva_phys.groupby([id_col, cyc_col]).agg({
    **{f: 'mean' for f in phys_cols},
    **{t: 'first' for t in targets}
}).reset_index()

X_tr = tr_phys_agg[phys_cols].values
X_va = va_phys_agg[phys_cols].values
