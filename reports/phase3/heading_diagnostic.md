# Smartphone Heading Diagnostic

**Date:** 2026-09-17  
**Sessions analysed:** S1 (Driver A, stationary then drive), Vw12 (Driver E, highway)  
**Status:** Read-only diagnostic — no estimator code or defaults modified  
**Data source:** S-file CSVs only; no V-file / reference trajectory used

---

## Executive summary

| Signal | World heading? | Stationary reliability | Used during GNSS blackout? |
|---|---|---|---|
| ORIENTATION Yaw | **No** | Poor — drifts −0.20°/s (s1), −0.73°/s (vw12) | Unsafe |
| Magnetometer + gravity (tilt-compensated) | **No** — device frame only | **Excellent** — circ_std < 1.3° over 44 s (s1) | Device-frame only |
| GPS ORIENTATION (course field) | Yes (COG) | N/A — only available while moving | Yes, post-blackout |
| ORIENTATION Pitch / Roll | N/A — not physical tilt | Constant (app convention) | N/A |

**Bottom line:** No signal in the S-file provides a trustworthy, validated initial vehicle heading before the GNSS blackout.

---

## 1. ORIENTATION Yaw — not an absolute heading

### Stationary drift (s1, t ∈ [12.0, 55.7] s, 43.7 s of rest)

| Quantity | Value |
|---|---|
| Circular mean | 19.6° |
| Circular std (jitter) | 3.07° |
| Linear drift rate | −0.199 °/s |
| Total drift over 43.7 s | −8.7° |

### Drift rate changes sign under driving

| Window | d(yaw)/dt |
|---|---|
| s1 stationary [12, 55.7] s | −0.200 °/s |
| s1 driving t > 2000 s | +0.240 °/s |
| vw12 stationary [0, 4.6] s | −0.731 °/s |

Yaw drift is not a constant bias — it is a random walk / software artefact.

### Not a gyro integration identity

Comparing cumulative Δyaw against the GYROSCOPE Yaw column × dt:

| Session | resid median | resid MAD |
|---|---|---|
| s1 | 25.5° | 76.3° |
| vw12 | −12.0° | 5.6° |

The ORIENTATION Yaw is **not** a simple integral of the reported gyroscope yaw channel.

### Offset vs GPS course is highly variable

| Session | n (fix pairs) | offset mean | offset circ_std |
|---|---|---|---|
| s1 | 531 | 194.5° | **77.5°** |
| vw12 | 10 | 220.7° | 11.0° |

The s1 offset varies by nearly a full circle — yaw does not track GPS course at all during that session. vw12 is more consistent (smaller spread, gentler turns) but the 220° offset itself shows the frame is device-relative, not aligned to vehicle or world.

### Conclusion

ORIENTATION Yaw is a **modulo-360 device-relative rotation angle** with arbitrary initial reference and unstable drift. It is not safe to use as a vehicle heading initialisation in any phase of the filter.

---

## 2. ORIENTATION Pitch / Roll — not physical tilt

| Session | pitch mean | pitch std | roll mean | roll std | gravity tilt (device-Z vs vertical) |
|---|---|---|---|---|---|
| s1 | −80.3° | 2.7° | −150.0° | 82.5° | **0.126°** |
| vw12 | −85.7° | 0.7° | −142.8° | 9.4° | **0.283°** |

Gravity shows the phone is essentially flat (tilt < 0.3°) — yet pitch is −81 to −87° and roll is −127 to −153°. These Euler angles are clearly in a **fixed non-world convention** (likely device-axis rotations, or a non-standard Euler sequence used by the capture app). The roll value wraps across the full session in s1 (std 82.5°), confirming it is not a stable physical quantity.

Pitch and roll are not usable for attitude initialisation.

---

## 3. Magnetometer + gravity — stable in the device frame

### Stationary performance (s1, t ∈ [12.0, 55.7] s)

| Quantity | Value |
|---|---|
| Tilt-compensated mag heading (device frame) | **254.0°** |
| Circular std | **1.18°** |
| Sub-window stds (15 s) | 0.58 – 1.24° |
| Mean |B| | 40.1 µT (geomagnetic, UK) |

The magnetometer provides a **genuinely stable, self-consistent horizontal heading** in the phone's XY plane while stationary. This is the only reliable absolute-ish heading signal in the S-file.

### Why it cannot give a vehicle heading

1. **Device → vehicle mount offset (α) is unknown.** The 254° is the angle from device +X to magnetic north in the device horizontal plane. To convert to vehicle heading: `vehicle_heading = mag_device_heading − α + declination`. α depends on how the phone is mounted (portrait vs landscape, which edge faces forward, any rotation).

2. **α cannot be calibrated while stationary.** No motion data exists to infer it.

3. **α cannot be reliably estimated from post-motion GPS course.** Two different departure windows yield contradictory estimates:
   - [100, 400] s: GPS course 275.9° → α ≈ +19.3°
   - [500, 900] s: GPS course 292.0° → α ≈ +41.9°
   
   The 22.6° discrepancy implies either the vehicle was not heading straight at departure (perpendicular parking, turning), or significant magnetic interference corrupts the driving heading. Either way, α is unreliable.

4. **Magnetometer degrades severely under motion.** Full-session mag-device-heading vs GPS course: circ_std = **103.5°** (s1). Magnetic interference from the vehicle's electrical system, nearby metal infrastructure, and road surroundings swamps the signal during driving.

### Can it still be useful?

The magnetometer signal is excellent at holding a **constant device-frame heading bias** while stationary — it would be suitable for:
- Detecting whether the device was re-mounted between sessions (different α),
- Providing a relative rotation reference (if the car rotates, the mag heading rotates),
- Sanity-checking ORIENTATION Yaw stability.

It is **not** suitable for providing an absolute vehicle heading without an external calibration step (e.g., a drive straight segment + GPS course to solve for α).

---

## 4. GPS ORIENTATION (course field)

This is the phone's reported course-over-ground.

| Session | n (fix pairs) | offset vs inter-fix displacement course (mean ± std) |
|---|---|---|
| s1 | 531 | 359.6° ± 28.9° |
| vw12 | 10 | 0.7° ± 0.8° |

The course field is a legitimate COG signal — consistent with true displacement at low fix rates (vw12: <1° std). In s1 the larger spread (28.9°) is due to small inter-fix displacements at slow speed (2–19 m/s over ~9 s fix intervals). Only available while moving.

---

## 5. GPS SPEED field — unit misnomer

Despite the header label "Kmh", the GPS SPEED values are **m/s** (confirmed by magnitude):

| Session | mean | max | Equivalent km/h |
|---|---|---|---|
| s1 | 7.43 | 18.8 | 27 – 68 km/h |
| vw12 | 23.3–26.5 | ~26 | ~83–95 km/h |

These values were already correctly used as m/s in the Phase 3 estimator (the earlier unit-check failure was caused by 9-s-latched GPS fixes producing artificially short displacements in the speed check).

---

## 6. s1 pre-blackout stationary heading references

The vehicle is stationary from approximately t ≈ 12 s to t ≈ 55.7 s (43.7 s). During this window, only two heading-like signals are present:

| Signal | Mean | Circ. std | Drift | Verdict |
|---|---|---|---|---|
| ORIENTATION Yaw | 19.6° | 3.07° | −0.20 °/s (−8.7° total) | Unusable |
| Mag device heading (tilt-compensated) | 254.0° | 1.18° | Stable (no drift) | Stable in device frame only |

Neither can be converted to a vehicle heading without knowing the phone's mounting orientation (α), which is unobservable while the car is stationary.

---

## 7. GPS COG heading at first motion (s1)

The earliest reliable GPS course is obtained from consecutive fix displacements starting ~100 s:

| Window | GPS course | Yaw (mean) | Yaw − course | Mag_tc (mean) | Mag_tc − course |
|---|---|---|---|---|---|
| [100, 400] s | 275.9° | 359.0° | 83.1° | 256.5° | −19.4° |
| [500, 900] s | 292.0° | 352.5° | 60.5° | 250.1° | −41.9° |

The early fix-courses (t < 100 s) are unreliable due to GPS jitter at low speed — the first fix at t = 2.9 s reports course = 239° (the car is stationary). The [100, 400] s window is the earliest reasonably trustworthy displacement-based course, but note it is a multi-minute average, not the instantaneous heading at departure.

---

## 8. Cross-session consistency (device mount check)

| Session | yaw (stationary) | mag_tc (stationary) | yaw − mag_tc |
|---|---|---|---|
| s1 | 19.6° | 254.0° | −234.4° (≈ 125.6°) |
| vw12 | 248.0° | 254.7° | −6.7° |

The yaw − mag_tc offset differs by **132°** between the two sessions. This confirms:
- The phone was mounted differently in each session, OR
- ORIENTATION Yaw's arbitrary initial reference shifted between recording sessions, OR
- Both signals are unreliable and the coincidence of vw12's yaw being close to mag_tc is accidental.

Since the two sessions use different devices / mounting positions (Driver A vs Driver E), the discrepancy is expected and further confirms that neither signal alone can provide a vehicle heading without per-session calibration.

---

## 9. Implications for Phase 3 estimator

- **s1:** The Phase 3 filter used yaw = 0 with `heading_observable = False` for t < 60 s. This is the correct fallback: no reliable heading exists in the S-file data alone. The filter's own observability analysis (the "delayed heading" finding) is corroborated by this diagnostic.
- **vw12:** The filter initialised with GPS course at first fix (t ≈ 0), which is appropriate — the car is moving and GPS course is valid. The ORIENTATION yaw was not used.
- **Recommendation for any future smartphone-only pipeline:** Do not initialise heading from ORIENTATION Yaw. Use the first GPS course measurement after motion begins as the earliest trustworthy heading. The magnetometer can serve as a sanity-check or relative-rotation reference but not as an absolute vehicle heading without motion-based α calibration.

---

## Files produced

| File | Description |
|---|---|
| `heading_diag_signal_stats.csv` | Per-session summary statistics (yaw, pitch, roll, mag, tilt, speed) |
| `heading_diag_stationary.csv` | Stationary window stats (yaw drift, mag stability, pitch/roll means) |
| `heading_diag_fixes_s1.csv` | Per-fix (inter-displacement) course vs yaw vs mag for s1 |
| `heading_diag_fixes_vw12.csv` | Per-fix course vs yaw vs mag for vw12 |
| `heading_diagnostic.md` | This report |
