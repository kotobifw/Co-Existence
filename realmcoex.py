import networkx as nx
import time
import operator
import numpy as np
import csv
import math
import json
from itertools import permutations, accumulate
import folium
from operator import mul
from folium import plugins
from folium.plugins import HeatMap
from folium import FeatureGroup, LayerControl
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from vincenty import *
from pprint import pprint
from shapely.ops import cascaded_union
from shapely.geometry import Polygon, Point, LinearRing, MultiPolygon
from shapely import geometry
from shapely.geometry import shape
import time
from geojson import Feature

# ----------
# Continental USA
# ----------
with open('CONUS.geojson', 'r') as f:
    ConUSA = json.load(f)
feature = ConUSA['features'][0]
coordsConUSA = geometry.Polygon(feature['geometry']['coordinates'][0])

def Haversine(lat1, lon1, lat2, lon2):
  """
  Calculate the Great Circle distance on Earth between two latitude-longitude
  points
  :param lat1 Latitude of Point 1 in degrees
  :param lon1 Longtiude of Point 1 in degrees
  :param lat2 Latitude of Point 2 in degrees
  :param lon2 Longtiude of Point 2 in degrees
  :returns Distance between the two points in kilometres
  """
  Rearth = 6371
  lat1   = np.radians(lat1)
  lon1   = np.radians(lon1)
  lat2   = np.radians(lat2)
  lon2   = np.radians(lon2)
  #Haversine formula
  dlon = lon2 - lon1
  dlat = lat2 - lat1
  a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
  c = 2 * np.arcsin(np.sqrt(a))
  return Rearth*c

def DistanceCentroid(lat,lon,geom): # Call it a different function that is clearly not polygon
  """
  Calculate the closest distance between a polygon and a latitude-longitude
  point, using only spherical considerations. Ignore edges.
  :param lat  Latitude of query point in degrees
  :param lon  Longitude of query point in degrees
  :param geom A `shapely` geometry whose points are in latitude-longitude space
  :returns: The minimum distance in kilometres between the polygon and the
            query point
  """
  # assume that geometry is stored with order (lon,lat)
  centroidObj = geom.centroid
  #p1 = (lat, lon)
  #p0 = (centroidObj.y, centroidObj.x)
  #dist = geopy.distance.vincenty(p0, p1).km
  dist = np.min(Haversine(centroidObj.y, centroidObj.x, lat, lon))
  return dist

def style_function_red(feature):
    return {
        'fillColor': 'red',
        'fillOpacity': 0.2,
        'color': 'red',
        'weight': 4,
    }
def style_function_blue(feature):
    return {
        'fillColor': 'blue',
        'fillOpacity': 0.2,
        'color': 'blue',
        'weight': 4,
    }
def style_function_green(feature):
    return {
        'fillColor': 'green',
        'fillOpacity': 0.2,
        'color': 'green',
        'weight': 2,
    }
def style_function_yellow(feature):
    return {
        'fillColor': 'yellow',
        'fillOpacity': 0.2,
        'color': 'yellow',
        'weight': 2,
    }
def style_function_turquoise(feature):
    return {
        'fillColor': 'turquoise',
        'fillOpacity': 0.2,
        'color': 'turquoise',
        'weight': 2,
    }
def style_function_grey(feature):
    return {
        'fillColor': 'grey',
        'fillOpacity': 0.2,
        'color': 'yellow',
        'weight': 2,
    }
def style_function_pink(feature):
    return {
        'fillColor': 'pink',
        'fillOpacity': 0.2,
        'color': 'pink',
        'weight': 2,
    }

#colors_fill = ['red', 'green', 'blue', 'magenta', 'orange', 'purple', 'gray', 'black', 'brown','yellow','turquoise','pink','olive','darksalmon','ivory']

cmap = cm.get_cmap('coolwarm', 11)
colors_fill = []
for i in range(cmap.N):
   rgb = cmap(i)[:3]  # will return rgba, we take only first 3 so we get rgb
   colors_fill.append(matplotlib.colors.rgb2hex(rgb))

colors_fill = [
    'red',
    'blue',
    'gray',
    'darkred',
    'lightgray',
    'orange',
    'beige',
    'black',
    'green',
    'darkgreen',
    'darkblue',
    'lightgreen',
    'cadetblue',
    'lightblue',
    'purple',
    'darkpurple',
    'pink',
    'lightgray',
    'lightred',
    'turquoise'
]

## If there are more that one realm to study put the name in the following line
otherrealmunderstudy = "Kansas City (100 MHz)"
fillOpac = 0.4
def style_function(feature):

    CNG = feature['properties']['CNG']
    CxG = feature['properties']['CxG']
    Realm = feature['properties']['Realm']

    #print("CNG=", CNG,"CxG=", CxG, "realm=", Realm)
    if CNG == 'VRZ':
        if Realm == 'NENorth':
            if CxG == 'FEDERATEDWIRELESS':
                clr = colors_fill[0]
            elif CxG == 'GOOGLE':
                clr = colors_fill[1]
            else:
                clr = colors_fill[2]
        elif Realm == otherrealmunderstudy:
            if CxG == 'FEDERATEDWIRELESS':
                clr = colors_fill[0]
            elif CxG == 'GOOGLE':
                clr = colors_fill[1]
            else:
                clr = colors_fill[2]
        else:
            clr = colors_fill[9]
    elif CNG == 'TMO':
        if Realm == 'NENorth':
            if CxG == 'FEDERATEDWIRELESS':
                clr = colors_fill[3]
            elif CxG == 'GOOGLE':
                clr = colors_fill[4]
            else:
                clr = colors_fill[5]
        elif Realm == otherrealmunderstudy:
            if CxG == 'FEDERATEDWIRELESS':
                clr = colors_fill[3]
            elif CxG == 'GOOGLE':
                clr = colors_fill[4]
            else:
                clr = colors_fill[5]
        else:
            clr = colors_fill[10]
    else:
        if Realm == 'NENorth':
            if CxG == 'FEDERATEDWIRELESS':
                clr = colors_fill[6]
            elif CxG == 'GOOGLE':
                clr = colors_fill[7]
            else:
                clr = colors_fill[8]
        elif Realm == otherrealmunderstudy:
            if CxG == 'FEDERATEDWIRELESS':
                clr = colors_fill[6]
            elif CxG == 'GOOGLE':
                clr = colors_fill[7]
            else:
                clr = colors_fill[8]
        else:
            clr = colors_fill[11]
    #print(clr)
    return {
        'fillColor': clr,
        'fillOpacity': fillOpac,
        'color': clr,
        'weight': 3,
    }

def GetStandardAntennaGains(hor_dirs, ant_azimuth=None, ant_beamwidth=None, ant_gain=0):
  """Computes the antenna gains from a standard antenna defined by beamwidth.
  See R2-SGN-20.
  This uses the standard 3GPP formula for pattern derivation from a given
  antenna 3dB cutoff beamwidth.
  Directions and azimuth are defined compared to the north in clockwise
  direction and shall be within [0..360] degrees.

  Inputs:
    hor_dirs:       Ray directions in horizontal plane (degrees).
                    Either a scalar or an iterable.
    ant_azimut:     Antenna azimuth (degrees).
    ant_beamwidth:  Antenna 3dB cutoff beamwidth (degrees).
                    If None, then antenna is isotropic (default).
    ant_gain:       Antenna gain (dBi).

  Returns:
    The CBSD antenna gains (in dB).
    Either a scalar if hor_dirs is scalar or an ndarray otherwise.
  """
  is_scalar = np.isscalar(hor_dirs)
  hor_dirs = np.atleast_1d(hor_dirs)

  if (ant_beamwidth is None or ant_azimuth is None or
      ant_beamwidth == 0 or ant_beamwidth == 360):
    gains = ant_gain * np.ones(hor_dirs.shape)
  else:
    bore_angle = hor_dirs - ant_azimuth
    bore_angle[bore_angle > 180] -= 360
    bore_angle[bore_angle < -180] += 360
    gains = -12 * (bore_angle / float(ant_beamwidth))**2
    gains[gains < -20] = -20.
    gains += ant_gain

  if is_scalar: return gains[0]
  return gains

class grant:
    def __init__(self, grantid=None, grantstate=None, cbsdid=None, cbsduserid=None, cbsdsasid=None, lat=None, lon=None,
                 antennaGain=None,
                 antennaAzimuth=None, antennaBeamwidth=None, iapEirp=None, indoor=None, contour = None, cat = None, realm = None, distance = None):
        self.grantid = grantid
        self.grantstate = grantstate
        self.cbsdid = cbsdid
        self.cbsduserid = cbsduserid
        self.cbsdsasid = cbsdsasid
        self.lon = lon
        self.lat = lat
        self.antennaGain = antennaGain
        self.antennaAzimuth = antennaAzimuth
        self.antennaBeamwidth = antennaBeamwidth
        self.iapEirp = iapEirp
        self.indoor = indoor
        self.contour = contour
        self.cat = cat
        self.realm = realm
        self.distance = distance

class realm:
    def __init__(self, name=None, lower_frequency=None, upper_frequency=None, contour = None):
        self.name = name
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.contour = contour

class myFSS:
    def __init__(self, BWtype=None, lower_frequency=None, upper_frequency=None, coords = None):
        self.BWtype = BWtype
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.coords = coords

class myEDPA:
    def __init__(self, name=None, CatAD=None, CatBD=None, geom = None, coords = None):
        self.name = name
        self.CatAD = CatAD
        self.CatBD = CatBD
        self.geom = geom
        self.coords = coords

# -----------
# Initializations
# -----------
azimuths = np.arange(0.0, 360.0)
rads = azimuths * math.pi/180.0

# Frequency settings
ALL_BW = 150000000
ChannelBW = 5000000
Mhz = 1000000
NumberOfChannels = 30

# Code options
verificationtest = 1
readjson = 1
# Graph options
show_onmap = 1
seeCBSDs = 1
seeMarkers = 1
seeGraph = 0
getincumbent = 1
other_incumbents = 1
show_graphs_connected_set = 0
perform_extra5MHz_assignment = 0
perform_channel_mapping = 1

# -----------
# GLOBAL PARAMETERS FOR CONTOUR
# -----------
# coverage criteria
THRESHOLD_PER_10MHZ = -96  # dBm/MHz
# path loss parameters
LightSpeedC = 3e8
Freq = 3625*1000000  #hz
PathLossExponent = 3.0  # select depending on scenario
Wavelength = LightSpeedC/Freq

if readjson:
    # --------------
    # realms20sep_extended.geojson:
    # --------------
    with open('realms20sep_extended.geojson', 'r') as f:
        Realms = json.load(f)
    realms = []
    for feature in Realms['features']:
        name = feature['properties']['Name']
        lower_frequency = feature['properties']['Lower Frequency']
        upper_frequency = feature['properties']['Upper Frequency']
        contour = shape(feature['geometry'])

        realms.append(realm(name, lower_frequency, upper_frequency, contour))
    # --------------
    # cbsd.grants.json:
    # --------------
    with open('modified.grantcbsddata.json', 'r') as f:
        Grants = json.load(f)
    grants = []
    for feature in Grants['features']:
        grantstate = feature['features'][0]['properties']['oper']['grantState']
        if grantstate != "TERMINATED":
            grantcbsdid = feature['features'][0]['properties']['admin']['cbsdId']
            grantid = feature['features'][0]['properties']['admin']['grantId']
            grants.append(grant(grantid, grantstate, grantcbsdid))
    print("Number of grants is =", len(grants))

    # --------------
    # cbsd.registrations.json:
    # --------------
    with open('modified.cbsddata.json', 'r') as f:
        CBSDs = json.load(f)
    pairedgrants = []
    for feature in CBSDs['features']:

        cbsdid = feature['features'][0]['properties']['admin']['cbsdId']

        for item in grants:
            if item.cbsdid == cbsdid:
                # If we need cbsdReferenceId
                #item.cbsdrefid = feature['features'][0]['properties']['oper']['cbsdReferenceId']
                #item.cbsdrefids.append(cbsdid)
                item.cbsduserid = feature['features'][0]['properties']['admin']['userId']
                item.cbsdsasid = feature['features'][0]['properties']['admin']['sasId']
                item.cat = feature['features'][0]['properties']['admin']['cbsdCategory']
                item.lon = feature['features'][0]['properties']['admin']['installationParam']['longitude']
                item.lat = feature['features'][0]['properties']['admin']['installationParam']['latitude']
                item.antennaGain = feature['features'][0]['properties']['admin']['installationParam']['antennaGain']
                item.antennaAzimuth = feature['features'][0]['properties']['admin']['installationParam']['antennaAzimuth']
                item.antennaBeamwidth = feature['features'][0]['properties']['admin']['installationParam']['antennaBeamwidth']
                item.iapEirp = feature['features'][0]['properties']['admin']['installationParam']['eirpCapability']
                item.indoor = feature['features'][0]['properties']['admin']['installationParam']['indoorDeployment']
                po = Point([item.lon , item.lat])
                for realm in realms:
                    if po.within(realm.contour):
                        item.realm= realm.name
                        break
                pairedgrants.append(item)

print("Number of realms is =", len(realms))
print("Number of CBSDs with grants is =", len(pairedgrants))

# -----------
# INCUMBENT INFORMATION
# -----------
if getincumbent:
    # ----------
    # ESC
    # ----------
    with open('modified.incumbent.federatedwireless.esc.json', 'r') as f:
        ESCs = json.load(f)
    realmESC = {}
    for feature in ESCs['features']:
        ESClon = float(feature['features'][0]['geometry']['coordinates'][0])
        ESClat = float(feature['features'][0]['geometry']['coordinates'][1])
        coords = geometry.Point(ESClat, ESClon)
        coords = [ESClat, ESClon]
        po = Point([ESClon, ESClat])
        for realm in realms:
            if po.within(realm.contour):
                ESC_realm = realm.name
                break
        if ESC_realm in realmESC.keys():
            realmESC[ESC_realm].append(coords)
        else:
            realmESC[ESC_realm] = [coords]

    # ----------
    # E DPAs
    # ----------
    with open('e-dpas.geojson', 'r') as f:
        DPAs = json.load(f)
    EDPAs = []
    realmEDPA = {}
    for feature in DPAs['features']:

        geom = feature['geometry']['type']
        CatBD = int(feature['properties']['catBNeighborhoodDistanceKm'])
        CatAD = int(feature['properties']['catANeighborhoodDistanceKm'])
        Name = feature['properties']['Name']

        if geom == "Polygon":
            coords = geometry.Polygon(feature['geometry']['coordinates'][0])
            x, y = coords.exterior.coords.xy
            if x[0] > -126:
                for realm in realms:
                    if coords.intersects(realm.contour):
                        DPA_realm = realm.name
                        break
                if DPA_realm in realmEDPA.keys():
                    realmEDPA[DPA_realm].append(myEDPA(Name, CatAD, CatBD, geom, coords))
                else:
                    realmEDPA[DPA_realm] = [myEDPA(Name, CatAD, CatBD, geom, coords)]

        elif geom == "Point":
            coords = geometry.Point(feature['geometry']['coordinates'])
            if coords.x > -126:
                for realm in realms:
                    po = Point([coords.x, coords.y])
                    if po.within(realm.contour):
                        DPA_realm = realm.name
                        break
                if DPA_realm in realmEDPA.keys():
                    realmEDPA[DPA_realm].append(myEDPA(Name, CatAD, CatBD, geom, coords))
                else:
                    realmEDPA[DPA_realm] = [myEDPA(Name, CatAD, CatBD, geom, coords)]
        else: # ignore multipolygon
            pass

    # ----------
    # P-DPAs
    # ----------
    with open('p-dpas.geojson', 'r') as f:
        DPAs = json.load(f)
    realmPDPA = {}
    for feature in DPAs['features']:

        geom = feature['geometry']['type']
        CatBD = int(feature['properties']['catBNeighborhoodDistanceKm'])
        CatAD = int(feature['properties']['catANeighborhoodDistanceKm'])
        Name = feature['properties']['Name']

        if geom == "Polygon":
            coords = geometry.Polygon(feature['geometry']['coordinates'][0])
            if x[0] > -126:
                for realm in realms:
                    if coords.intersects(realm.contour):
                        DPA_realm = realm.name
                        break
                if DPA_realm in realmPDPA.keys():
                    realmPDPA[DPA_realm].append(myEDPA(Name, CatAD, CatBD, geom, coords))
                else:
                    realmPDPA[DPA_realm] = [myEDPA(Name, CatAD, CatBD, geom, coords)]
        elif geom == "Point":
            coords = geometry.Point(feature['geometry']['coordinates'])
            if coords.x > -126:
                for realm in realms:
                    po = Point([coords.x, coords.y])
                    if po.within(realm.contour):
                        DPA_realm = realm.name
                        break
                if DPA_realm in realmPDPA.keys():
                    realmPDPA[DPA_realm].append(myEDPA(Name, CatAD, CatBD, geom, coords))
                else:
                    realmPDPA[DPA_realm] = [myEDPA(Name, CatAD, CatBD, geom, coords)]
        else: # ignore multipolygon
            pass

    # ----------
    # FSS
    # ----------
    with open('FSS_latest.json', 'r') as f:
        FSSs = json.load(f)
    results = FSSs["result"] # array of dicts
    realmFSS = {}
    for FSS in results:
        lower_freq = FSS["lower_frequency"]
        lower_freq = int(lower_freq.replace(',', ''))
        # include if not TTC FSS
        if lower_freq < 3699:
            upper_freq = FSS["upper_frequency"]
            upper_freq = int(upper_freq.replace(',', ''))
            if lower_freq == 3625:
                BWtype = 0
            else:
                # just in case something other than 3600
                BWtype = 1
            FSSlat = float(FSS['earth_station_latitude_decimal'])
            FSSlon = float(FSS['earth_station_longitude_decimal'])
            po = Point([FSSlon, FSSlat])
            coords = geometry.Point(FSSlat, FSSlon)
            for realm in realms:
                if po.within(realm.contour):
                    FSS_realm = realm.name
                    break
            if FSS_realm in realmFSS.keys():
                realmFSS[FSS_realm].append(myFSS(BWtype,lower_freq,upper_freq,coords))
            else:
                realmFSS[FSS_realm] = [myFSS(BWtype,lower_freq,upper_freq,coords)]

if other_incumbents:
    # ----------
    # GWPZs
    # ----------
    with open('Sectors3650_10-20-2017.json', 'r') as f:
        GWPZs = json.load(f)
    GWPZList = []
    for feature in GWPZs['features']:

        lower_freq = feature['properties']['u_lower_frequency']
        upper_freq = feature['properties']['u_upper_frequency']
        if upper_freq == 'Mhz':
            upper_freq = 3700
        else:
            upper_freq = int(upper_freq[0:4])
        lower_freq = int(lower_freq[0:4])

        if upper_freq == 3675 and lower_freq == 3650:
            BWtype = 0
        elif upper_freq == 3700 and lower_freq == 3675:
            BWtype = 1
        else:
            BWtype = 2

        coords = geometry.Polygon(feature['geometry']['coordinates'][0])
        GWPZList.append(Feature(geometry=coords,
                                properties={"lower_freq": lower_freq,
                                            "upper_freq": upper_freq,
                                            "BWtype": BWtype}))

    # ----------
    # EZ
    # ----------
    # 3650-3700
    with open('EZ3550_3650.geojson', 'r') as f:
        EZ3550 = json.load(f)
    with open('EZ3650_3700.geojson', 'r') as f:
        EZ3650 = json.load(f)
    # ---------------
    # Quiet Zones
    # ---------------
    # Rule - Blocked if inside
    NRAO_NRRO = Polygon([[37.5, -78.5], [39.25, -78.5], [39.25, -80.5], [37.5, -80.5], [37.5, -78.5]])
    # Rule Encumbered if Cat A within 3.8 km, Cat B within 80 km
    Table_Mountain = [40.130660, -105.244596]
    # Rule Blocked if <= 2.4km, Encumbered if Cat B and >2.4 and <=4.8
    FCC_Field_Offices = [[42.60558, -85.9556], [44.44508, -69.0823], [42.91339, -77.2661], [31.50064, -109.654],
                         [48.95567, -122.555], [40.9225, -98.4287], [60.72389, -151.338], [27.44169, -97.8836],
                         [39.16511, -76.8211], [37.72492, -121.754], [33.86233, -84.7238], [18.00525, -66.3752],
                         [27.60614, -80.6348], [21.376, -157.996]]

# -----------
# Coverage for each CBSD
# -----------
CBSDlat = 40.73  #NY
CBSDlon = -73.94 #NY
hmap = folium.Map(location=[CBSDlat, CBSDlon], zoom_start=8, control_scale=True)
contours = []
for grant in pairedgrants:

    antenna_gains = GetStandardAntennaGains(azimuths,
      grant.antennaAzimuth,
      grant.antennaBeamwidth,
      grant.antennaGain)

    show_antenna = 0
    if show_antenna:
        plt.axes(projection='polar')
        for cnt, radian in enumerate(rads):
            plt.polar(radian, antenna_gains[cnt], 'o')
        plt.show()

    Pr0max = grant.iapEirp - (10*PathLossExponent*math.log10(4*math.pi/Wavelength))
    d_g = []
    for cnt, g in enumerate(antenna_gains):
        d_g.append(10**((Pr0max - grant.antennaGain + g - THRESHOLD_PER_10MHZ)/20.0))

    show_d_g = 0
    if show_d_g:
        plt.axes(projection='polar')
        for cnt, radian in enumerate(rads):
            plt.polar(radian, d_g[cnt], 'o')
        plt.show()

    circle = []
    for cnt, az in enumerate(rads):
        lat, lon, bear = GeodesicPoint(grant.lat, grant.lon, d_g[cnt]/1000, azimuths[cnt])
        circle.append([lat, lon])
    xx = [item[0] for item in circle]  # lats
    yy = [item[1] for item in circle]  # lons
    Coverage = Polygon(list(zip(yy, xx)))
    grant.contour = Coverage

    # -----------
    # Distance Caclulation
    # -----------
    CBSD = geometry.Point(grant.lon, grant.lat)

    D1_ESC = 1.0  #3550 - 3660
    D2_ESC = 1.0  #3550 - 3680
    D_DPA = 1.0
    D_PDPA = 1.0
    D0_FSS = 1.0
    D1_FSS = 1.0

    if grant.realm in realmESC.keys():
        for ESC in realmESC[grant.realm]:
            dist = Haversine(grant.lat, grant.lon, ESC[0],
                             ESC[1])
            if grant.cat == "A":
                dist = dist / 40.0
                if dist < D1_ESC:
                    D1_ESC = dist
            elif grant.cat == "B":
                dist = dist / 80.0
                if dist < D2_ESC:
                    D2_ESC = dist

    if grant.realm in realmEDPA.keys():
        for DPAs in realmEDPA[grant.realm]:
            if coordsConUSA.contains(shape(DPAs.coords)):
                geom = DPAs.geom
                if grant.cat == 'A':
                    NeighborhoodDist = DPAs.CatAD
                else:
                    NeighborhoodDist = DPAs.CatBD
                if geom == 'Polygon':
                    DPA = shape(DPAs.coords)
                    if CBSD.within(DPA):
                        D_DPA = 0.0
                        break
                    else:
                        dist = DistanceCentroid(grant.lat, grant.lon, DPA)/NeighborhoodDist
                else:
                    dist = Haversine(grant.lat, grant.lon, DPAs.coords.y,
                                     DPAs.coords.x)
                    dist = dist/NeighborhoodDist
                    dist = dist / 20.0
                if dist < D_DPA:
                    D_DPA = dist

    if grant.realm in realmPDPA.keys():
        for DPAs in realmPDPA[grant.realm]:
            if grant.cat == 'A':
                NeighborhoodDist = DPAs.CatAD
            else:
                NeighborhoodDist = DPAs.CatBD
            if DPAs.geom == 'Polygon':
                DPA = shape(DPAs.geom)
                if CBSD.within(DPA):
                    D_PDPA = 0.0
                    break
                else:
                    dist = DistanceCentroid(grant.lat, grant.lon, DPA) / NeighborhoodDist
            else:
                dist = Haversine(grant.lat, grant.lon, DPAs.coords.y,
                                 DPAs.coords.x)
                dist = dist / NeighborhoodDist
                dist = dist / 20.0  # 7/23
            if dist < D_PDPA:
                D_PDPA = dist

    if grant.realm in realmFSS.keys():
        for FSS in realmFSS[grant.realm]:
            BWtype = FSS.BWtype
            dist = Haversine(lat, lon, FSS.coords.y,
                             FSS.coords.x)
            dist = dist / 150.0
            if BWtype == 0:
                if dist < D0_FSS:
                    D0_FSS = dist
            else:
                if dist < D0_FSS:
                    D0_FSS = dist
                if dist < D1_FSS:
                    D1_FSS = dist

    grant.distance = [D1_ESC, D2_ESC, D_DPA, D_PDPA, D0_FSS,D1_FSS]
    #print([D1_ESC, D2_ESC, D_DPA, D_PDPA, D0_FSS,D1_FSS])

# -----------
# Unique CxG^CNGs with their cascaded contour
# -----------

uniqueCxGCNGs = {}
UniqueCxGs = []
for grant in pairedgrants:
    SASID = grant.cbsdsasid
    group = grant.cbsduserid + "^" + SASID + "^" + grant.realm
    if group in uniqueCxGCNGs.keys():
        uniqueCxGCNGs[group] = cascaded_union([uniqueCxGCNGs[group], grant.contour])
    else:
        uniqueCxGCNGs[group] = grant.contour

    if seeMarkers:
        CNG = grant.cbsduserid
        CxG = SASID
        if CNG == 'VRZ':
            if CxG == 'FEDERATEDWIRELESS':
                clr = colors_fill[0]
            elif CxG == 'GOOGLE':
                clr = colors_fill[1]
            else:
                clr = colors_fill[2]
        elif CNG == 'TMO':
            if CxG == 'FEDERATEDWIRELESS':
                clr = colors_fill[3]
            elif CxG == 'GOOGLE':
                clr = colors_fill[4]
            else:
                clr = colors_fill[5]
        else:
            if CxG == 'FEDERATEDWIRELESS':
                clr = colors_fill[6]
            elif CxG == 'GOOGLE':
                clr = colors_fill[7]
            else:
                clr = colors_fill[8]
        folium.Marker([grant.lat, grant.lon], icon=folium.Icon(color=clr,icon='')).add_to(hmap)

if show_onmap:
    for k, v in uniqueCxGCNGs.items():
        klist = k.split('^')
        props = {"CNG": klist[0], "CxG": klist[1], "Realm": klist[2]}
        feat = Feature(geometry=v, properties=props)
        folium.GeoJson(feat, name=k, style_function=style_function).add_to(hmap)

    folium.LayerControl().add_to(hmap)
    hmap.save(outfile='coexcbsds.html')

# -----------
# Get CxGCNG for each Realms
# -----------
realmslist = []
realmCxGCNG = {}

for k, v in uniqueCxGCNGs.items():
    klist = k.split('^')
    realm = klist[2]
    CxGCNG = klist[0] +'^'+ klist[1]
    if realm not in realmslist:
        realmslist.append(realm)
    if realm in realmCxGCNG:
        realmCxGCNG[realm].append(CxGCNG)
    else:
        realmCxGCNG[realm] = [CxGCNG]

print("Realms are ==", realmslist)
print("CxGCNG for each realms are ==", realmCxGCNG)

G = nx.Graph()
realm_connected_set = {}

for realm in realmslist:
    G.clear()
    print("We are in the realm of",realm)

    for cnt, CxGCNG in enumerate(realmCxGCNG[realm]):
        CxG1 = CxGCNG.split("^")[1]
        group1 = CxGCNG + '^' + realm
        G.add_node(group1)
        G.add_edge(group1, group1, weight=1)
        for CxGCNG2 in realmCxGCNG[realm][cnt+1:]:
            group2 = CxGCNG2 + '^' + realm
            CxG2 = CxGCNG2.split("^")[1]
            if uniqueCxGCNGs[group1].intersects(uniqueCxGCNGs[group2]):
                G.add_edge(group1, group2, weight=1)
                if CxG1 not in UniqueCxGs:
                    UniqueCxGs.append(CxG1)
                if CxG2 not in UniqueCxGs:
                    UniqueCxGs.append(CxG2)
    if seeGraph:
        pos = nx.spring_layout(G, k=0.45, iterations=20)
        nx.draw(G, pos)
        nx.draw_networkx_labels(G, pos, font_size=5)
        plt.show()

    # --------------
    # Connected Set and Dictionary for each realm
    # --------------
    print("CxG^CNG graph has %d connected sets" % nx.number_connected_components(G))
    connectSets = []
    for cs in nx.connected_components(G):
        print("operating on connected set: ", cs, "on the realm of", realm)
        CxGColors = []
        connected_set_total_colors = 0
        connected_set_allocation = {}
        for CxG in UniqueCxGs:
            CxGsubgraphs = []
            print("operating on CxG: ", CxG)
            # --------------
            # FORM CxG SUBGRAPH
            # --------------
            CxGG = nx.Graph()
            CxGG.add_nodes_from((n, G.nodes[n]) for n in cs if n.split('^')[1] == CxG)
            CxGG.add_edges_from((n, nbr, d)
                                for n, nbrs in G.adj.items() if n in cs and
                                n.split('^')[1] == CxG
                                for nbr, d in nbrs.items() if nbr in cs and
                                nbr.split('^')[1] == CxG)

            # Mapping of CNG->color in each CxG subgraph connected set
            # For each connected component of CxGG, perform graph coloring
            cxg_projected_set_maximum_number_colors = 0
            for cs_cnt, cs_CxG in enumerate(nx.connected_components(CxGG)):

                # subgraph for connected set
                cs_CxGG = nx.Graph()
                cs_CxGG.add_nodes_from((n, CxGG.nodes[n]) for n in cs_CxG)
                cs_CxGG.add_edges_from((n, nbr, d)
                                       for n, nbrs in CxGG.adj.items() if n in cs_CxG
                                       for nbr, d in nbrs.items() if nbr in cs_CxG)
                # --------------
                # COLOR
                # --------------
                start = time.time()
                d = nx.coloring.greedy_color(cs_CxGG, strategy=nx.coloring.strategy_saturation_largest_first)
                end = time.time()
                print("Coloring using DSATUR colors: %s seconds elapsed" % (end - start))
                CxGsubgraphs.append(d)
                numcolors = max(d.items(), key=operator.itemgetter(1))[1]
                # Convenient to have sorted in order of color
                sorted_by_value = sorted(d.items(), key=lambda kv: kv[1])
                # Determine whether this component results in largest number of colors
                if numcolors + 1 > cxg_projected_set_maximum_number_colors:
                    cxg_projected_set_maximum_number_colors = numcolors + 1

                if show_graphs_connected_set:
                    cmap = cm.get_cmap('coolwarm', numcolors + 1)
                    colors = []
                    for i in range(cmap.N):
                        rgb = cmap(i)[:3]  # will return rgba, we take only first 3 so we get rgb
                        colors.append(cm.colors.rgb2hex(rgb))
                    for color in range(0, numcolors + 1):
                        k0 = [k for k, v in d.items() if v == color]
                        nx.draw_networkx_nodes(cs_CxGG, pos,
                                               nodelist=k0,
                                               node_color=colors[color],
                                               node_size=500,
                                               alpha=0.8, with_labels=True)
                    nx.draw_networkx_edges(cs_CxGG, pos, with_labels=True)
                    nx.draw_networkx_labels(cs_CxGG, pos, font_size=8)
                    print("Khashayar test for Graph for cs=%d in CxG=%s has Nodes=%d, Edges=%d, Colors=%d"
                          % (cs_cnt, CxG, nx.number_of_nodes(cs_CxGG), nx.number_of_edges(cs_CxGG), numcolors + 1))
                    plt.title('Graph Coloring for ConnSet/CxG=%d/%s' % (cs_cnt, CxG))
                    plt.show()

            print("maximum colors in CxG = %s cs = %s" % (CxG, cxg_projected_set_maximum_number_colors))

            # --------------
            # Dictionary
            # --------------
            if CxG == 'FEDERATEDWIRELESS':
                FWCxGNumColor = cxg_projected_set_maximum_number_colors
                print("FWCxGNumColor",FWCxGNumColor)

            CxGColors.append(cxg_projected_set_maximum_number_colors)
            connected_set_total_colors += cxg_projected_set_maximum_number_colors
            connected_set_allocation.update({CxG: {"numberColors": cxg_projected_set_maximum_number_colors,
                                                   "subgraphs": CxGsubgraphs
                                                   }})

        myrealm = next((i for i in realms if i.name == realm), None)
        realmBW = myrealm.upper_frequency - myrealm.lower_frequency
        connected_set_bandwidth_per_color = ChannelBW * \
                                            math.floor(
                                                (realmBW * Mhz / connected_set_total_colors) / ChannelBW)# THIS IS AN ISSUE WHEN FW AND OTHER SASs HAS SAME NUM COLOR
        connected_set_allocation['bandWidthperColor'] = connected_set_bandwidth_per_color
        for CxG in UniqueCxGs:
            connected_set_allocation[CxG]['frequencyRange'] = {}
            if CxG == 'FEDERATEDWIRELESS':
                connected_set_allocation[CxG]['frequencyRange']['lowFrequency'] = myrealm.lower_frequency * Mhz
                connected_set_allocation[CxG]['frequencyRange']['highFrequency'] = (myrealm.lower_frequency * Mhz)+ connected_set_bandwidth_per_color * connected_set_allocation[CxG]['numberColors']
        connectSets.append(connected_set_allocation)
        print("coloring across CxGs")
        print(CxGColors)
    realm_connected_set[realm] = connectSets
print('The realm cs list is ==', realm_connected_set)

# -----------
# CHANNEL MAPPING FOR EACH REALM
# -----------
for realm in realmslist:
    for ConnectedSet in realm_connected_set[realm]:
        FWCxGInfo = ConnectedSet['FEDERATEDWIRELESS']['subgraphs']
        if len(FWCxGInfo) > 0:
            print("doing channel mapping for FW CxG\n")
            # Determining the larges connected component to determine the allocated BW for FW:
            FWCxGNumColor = ConnectedSet['FEDERATEDWIRELESS']['numberColors']
            try:
                sumCxGColors = FWCxGNumColor + ConnectedSet['GOOGLE']['numberColors'] + \
                               ConnectedSet['COMM']['numberColors']
            except KeyError:
                sumCxGColors = FWCxGNumColor
                print("Only FW is accepted as a CxG")
            allBW = ConnectedSet['FEDERATEDWIRELESS']['frequencyRange']['highFrequency'] - ConnectedSet['FEDERATEDWIRELESS']['frequencyRange']['lowFrequency']
            BWPerColorFW = int((allBW / Mhz) * FWCxGNumColor / sumCxGColors) / FWCxGNumColor  # THIS IS AN ISSUE WHEN FW AND OTHER SASs HAS SAME NUM COLOR

            for FWConnComp in FWCxGInfo:
                FWConnCompColors = max(FWConnComp.values()) + 1
                # find CBSDs in each color
                CBSDsInColor = [[] for i in range(FWConnCompColors)]
                distances = [[] for i in range(FWConnCompColors)]

                for cnt, cbsd in enumerate(pairedgrants):
                    if cbsd.realm == realm:
                        cbsdId = cbsd.cbsdid
                        CxG = cbsd.cbsdsasid
                        CNG = cbsd.cbsduserid

                        if CxG == 'FEDERATEDWIRELESS':
                            MyCxGCNG = CNG +'^FEDERATEDWIRELESS' + '^' + realm
                            color = FWConnComp.get(MyCxGCNG)
                            if color != None:
                                CBSDsInColor[color].append(cbsdId)
                                distances[color].append(cbsd.distance)
                distancenp = []
                #[D1_ESC, D2_ESC, D_DPA, D_PDPA, D0_FSS,D1_FSS]

                for distancec in distances:
                    b6alt2 = np.zeros((len(distancec), NumberOfChannels))
                    for dcnt, row in enumerate(distancec):
                        D1_ESC = row[0]
                        D2_ESC = row[1]
                        D_DPA = row[2]
                        D_PDPA = row[3]
                        D0_FSS = row[4]
                        D1_FSS = row[5]
                        # compute per channel D's
                        # 3550-3600
                        block1 = D_DPA * D1_ESC * D2_ESC
                        # 3600-3625
                        block2 = D_DPA * D0_FSS * D_PDPA * D1_ESC * D2_ESC
                        # 3625-3650
                        block3 = D_PDPA * D_DPA * D1_ESC * D1_FSS * D2_ESC
                        # 3650-3660
                        block4 = D1_FSS * D1_ESC * D2_ESC
                        # 3660-3680
                        block5 = D1_FSS * D2_ESC
                        # 3680-3700
                        block6 = D1_FSS

                        # Create for all 30 channels
                        b1 = np.array([block1] * 10)
                        b2 = np.array([block2] * 5)
                        b3 = np.array([block3] * 5)
                        b4 = np.array([block4] * 2)
                        b5 = np.array([block5] * 4)
                        b6 = np.array([block6] * 4)
                        b7 = np.concatenate((b1, b2, b3, b4, b5, b6), axis=0)
                        b8 = np.transpose(b7)
                        b6alt2[dcnt, :] = b8
                    distancenp.append(b6alt2)

                # Determine channels/color - allocate guard band to colors with largest # CBSDs
                chans_per_color = int(math.floor((BWPerColorFW * FWCxGNumColor / FWConnCompColors) / (ChannelBW / Mhz)))
                print("Starting chans_per_color=%d" % chans_per_color)
                Starting_chans = [chans_per_color] * FWConnCompColors
                # --------------
                # PERFORM EXTRA 5MHz ASSIGNMENT
                # --------------
                if perform_extra5MHz_assignment:
                    print("CBSDs in each color:")
                    NumNodesPerColor = []
                    for colorlist in CBSDsInColor:
                        NumNodesPerColor.append(len(colorlist))
                    NN = np.argsort(NumNodesPerColor)[::-1]  # sort the arg indice based on the arg attr
                    print("Colors sorted in order of #CBSDs")
                    print(NN)
                    colors_w_extra5 = int(
                        (BWPerColorFW * FWCxGNumColor - (ChannelBW / Mhz) * chans_per_color * FWConnCompColors) / (
                                    ChannelBW / Mhz))
                    print("colors with Extra 5 MHz=%d" % colors_w_extra5)

                    for color in range(0, colors_w_extra5):
                        Starting_chans[NN[color]] += 1
                    print("channels for each color after extra BW added")
                    print(Starting_chans)

                if perform_channel_mapping:
                    start = time.time()
                    # perform channel mapping algorithm
                    LL = [i for i in range(0, NumberOfChannels)]
                    bestnonzero = math.inf
                    bestscore = - math.inf
                    print('FWConnCompColors',FWConnCompColors)
                    for mapping in list(permutations(range(0, FWConnCompColors))):
                        T = [Starting_chans[i] for i in mapping]
                        U = list(accumulate(T))
                        U.insert(0, 0)
                        W = [LL[U[i]:U[i + 1]] for i in range(len(U) - 1)]
                        # assignment of colors to channels: Z
                        Z = {}
                        for c, colors in enumerate(mapping):
                            Z[colors] = W[c]

                        b10 = 1
                        b12 = 0
                        for c, colors in enumerate(mapping):

                            # all distances for nodes in color
                            b6 = distancenp[colors].T
                            # subset of distances related to channels given to mapping
                            #print("b6", b6)
                            b7 = b6[W[c], :]  # should be mostly ones, look at this and confirm for the winning mapping (combination)
                            #print("b7", b7)
                            # minimum in each row
                            b8 = np.amin(b7, axis=0)
                            # product of terms
                            b9 = np.prod(b8)  # when it is zero do not put it product
                            # aggregate product
                            b10 = b10 * b9
                            # number of zeros
                            b11 = b7.size - np.count_nonzero(b7)
                            # cumulative number of zeros
                            b12 += b11

                            show_score_calculation = 0
                            if (show_score_calculation):
                                print("\nfor c, colors={%d, %d}" % (c, colors))
                                print("b7:")
                                print(b7.T)
                                print("b8:")
                                print(b8)
                                print('b9=%f' % b9)
                                print('b11=%d zeros' % b11)

                        if b10 > bestscore:
                            bestscore = b10
                            bestmapping = mapping
                            bestZ = Z
                        if b12 < bestnonzero:
                            bestnonzero = b12
                            bestnonzeromapping = mapping
                            bestZnonzero = Z

                        show_mapping_progress = 0
                        if (show_mapping_progress):
                            print("\n{score, zeros} = {%.6E, %d} for color mapping:" % (b10, b12))
                            print(mapping)

                    if bestscore > 0.0:
                        print('have nonzero result, best mapping with score %.6E is' % bestscore)
                        print(bestmapping)
                        print('yielding assignment to channels of:')
                        print(bestZ)
                    else:
                        print('zero result, best mapping with %d zeros is' % bestnonzero)
                        print(bestnonzeromapping)
                        print('yielding assignment to channels of:')
                        print(bestZnonzero)

                    end = time.time()
                    print("Channel Mapping: %s seconds elapsed" % (end - start))

    print('final test', connectSets)
