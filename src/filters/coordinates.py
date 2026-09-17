import numpy as np

A = 6378137.0
E_SQ = 0.00669437999014

def wgs84_to_ecef(lat_deg, lon_deg, alt_m):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N = A / np.sqrt(1.0 - E_SQ * sin_lat**2)
    
    x = (N + alt_m) * cos_lat * np.cos(lon)
    y = (N + alt_m) * cos_lat * np.sin(lon)
    z = (N * (1.0 - E_SQ) + alt_m) * sin_lat
    return np.array([x, y, z])

class LocalTangentPlane:
    def __init__(self, lat0_deg, lon0_deg, alt0_m=0.0):
        self.lat0 = np.radians(lat0_deg)
        self.lon0 = np.radians(lon0_deg)
        self.alt0 = alt0_m
        self.ecef0 = wgs84_to_ecef(lat0_deg, lon0_deg, alt0_m)
        
        sl = np.sin(self.lat0)
        cl = np.cos(self.lat0)
        slo = np.sin(self.lon0)
        clo = np.cos(self.lon0)
        
        # ECEF to ENU rotation matrix
        self.R = np.array([
            [-slo, clo, 0.0],
            [-sl * clo, -sl * slo, cl],
            [cl * clo, cl * slo, sl]
        ])

    def to_enu(self, lat_deg, lon_deg, alt_m=0.0):
        ecef = wgs84_to_ecef(lat_deg, lon_deg, alt_m)
        return self.R @ (ecef - self.ecef0)