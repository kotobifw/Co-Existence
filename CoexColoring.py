import pymongo
from shapely.geometry import Polygon, Point, LinearRing, MultiPolygon
import geojson
from geojson import Point, Feature, FeatureCollection, dump
from shapely.ops import cascaded_union
import shelve
import branca
from pulp import *
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import operator
import numpy as np
import csv
import math
from itertools import permutations, accumulate

# Frequency settings
ALL_BW = 150000000
ChannelBW = 5000000
Mhz = 1000000
NumberOfChannels = 30

# Random settings
np.random.seed(5)
thresh1 = 0.98
thresh2 = 0.02

# Simulation options
MakeAdjList = 1
IncludePeerSAS = 0
show_graphs = 1
perform_channel_mapping = 0
perform_extra5MHz_assignment = 0
IPexactColoring = 1

# Graph options
GraphGridOn = 1
num_init = 64# used for locused = np.zeros(), 16 for 32, NEEDS TO BE FIXED!


# Mongo setting details
MongoEngineering = 0 # Original test, if you want this be careful about the number of CBSDs in the tests
MongoVerificationTest = 1 #Pirouz tests

# Query setting details
#myquery = {"type":"FeatureCollection"}
myquery = {"features.properties.hasGaa": True}

if MongoEngineering:
    username = "coex-user-01"
    password = "4fUY1uZ6ldUZzWcv"
    myclient = pymongo.MongoClient("mongodb://ds155839-a0.mqg72.fleet.mlab.com:55839/")
    db = myclient["load-co-ex"]
    db.authenticate(username, password)
    db.list_collection_names()
    TestcbsdCollection = db["CoExCbsdCollection"]  # we need populate DB and we need to use this
    cbsdDistance = db["cbsdDistanceCollection"]  # we need to create this based on the previous DB

if MongoVerificationTest:
    username = "dbtest"
    password = "Federated123"
    mongodb_url = "mongodb+srv://v16-jf65q.mongodb.net/coex"
    database_name = "CoexTest"
    auth_source = "admin"
    myclient = pymongo.MongoClient(mongodb_url, username= username, password=password, ssl=True, authSource=auth_source)
    db = myclient[database_name]
    TestcbsdCollection = db["Testcbsd7aDistanceCollection"]
    TestcbsdCollection = db["pirouzCoExCbsdCollection"]
    #TestcbsdCollection = db["Test11CoExCbsdCollection"]
    #TestcbsdCollection = db["Test10CoExCbsdCollection"]
    cbsdDistance = db["test_05_2distanceCoExCbsdCollection"]
    #cbsdDistance = db["Testcbsd5bDistanceCollection"]
    #cbsdDistance = db["Testcbsd5cDistanceCollection"]
    #cbsdDistance = db["Testcbsd5dDistanceCollection"]

# Test settings
testlimit = TestcbsdCollection.count_documents({}) #It can be big. Check based on DB
print(testlimit)
NumberofBatch = 4  # Number of batches
skipCBSD = int(testlimit/NumberofBatch)


if MakeAdjList:

    for NumBatch in range(NumberofBatch):

        print("inside batch=%d" % NumBatch)

        cbsds = TestcbsdCollection.find(myquery).sort('_id').limit(skipCBSD).skip( NumBatch * skipCBSD)

        Path1 = str(testlimit) + "adjlistOsamaFW" + str(NumBatch) + "Batch" + ".txt"
        Path2 = str(testlimit) + "adjlistOsamaGoogle" + str(NumBatch) + "Batch" + ".txt"
        Path3 = str(testlimit) + "adjlistOsamaCommscope" + str(NumBatch) + "Batch" + ".txt"

        try:
            with open(Path1, 'w') as f1, open(Path2, 'w') as f2, open(Path3, 'w') as f3:

                for cnt1, cbsd in enumerate(cbsds):
                    #print('khashayar cbsd', cbsd)
                    CBSDRefID = \
                        cbsd['features'][0]['properties']['cbsdReferenceId']
                    coords = \
                        Polygon(cbsd['features'][0]['geometry']['coordinates'][0])
                    CxG = \
                        cbsd['features'][0]['properties']['groupingParam'][0]['groupId'] #IT MAY NOT BE CxG
                    NEG = \
                        cbsd['features'][0]['properties']['groupingParam'][1]['groupId']
                    CNG = \
                        cbsd['features'][0]['properties']['groupingParam'][2]['groupId']
                    LAT1 = \
                        cbsd['features'][0]['properties']['installationParam']['latitude']
                    lON1 = \
                        cbsd['features'][0]['properties']['installationParam']['longitude']
                    cbsds2 = TestcbsdCollection.find(myquery).sort('_id').limit(testlimit)

                    for cnt2, cbsd2 in enumerate(cbsds2):
                        CBSDRefID2 = \
                            cbsd2['features'][0]['properties']['cbsdReferenceId']
                        coords2 = \
                            Polygon(cbsd2['features'][0]['geometry']['coordinates'][0])
                        CxG2 = \
                            cbsd2['features'][0]['properties']['groupingParam'][0]['groupId']
                        NEG2 = \
                            cbsd2['features'][0]['properties']['groupingParam'][1]['groupId']
                        CNG2 = \
                            cbsd2['features'][0]['properties']['groupingParam'][2]['groupId']
                        LAT2 = \
                            cbsd2['features'][0]['properties']['installationParam']['latitude']
                        #print('pirouz', LAT2, type(LAT2), LAT2 * 2)
                        lON2 = \
                            cbsd2['features'][0]['properties']['installationParam']['longitude']

                        if NEG == 'DISABLE_NEG_MECH' or CxG != CxG2 or NEG != NEG2 or CNG == CNG2:
                            if coords.intersects(coords2):
                                edgeOsama = CBSDRefID+','+CBSDRefID2+','+str(1)+','+CxG+\
                                            "^"+CNG+','+CxG +"^"+ NEG+','+CxG2 +"^"+ CNG2+\
                                            ','+CxG2 + "^" + NEG2 + ','+str(LAT1)+"^"+str(lON1)+','+str(LAT2)+"^"+str(lON2)

                                f1.write(edgeOsama)
                                f1.write('\n')
                                if IncludePeerSAS:
                                    # google
                                    if np.random.uniform() < thresh1 or CBSDRefID2 == CBSDRefID:
                                        f2.write(edgeOsama)
                                        f2.write('\n')
                                    # commscope
                                    if np.random.uniform() < thresh1 or CBSDRefID2 == CBSDRefID:
                                        f3.write(edgeOsama)
                                        f3.write('\n')
                            else:

                                if IncludePeerSAS:
                                    # google
                                    if np.random.uniform() < thresh2:
                                        edgeOsama = CBSDRefID + ',' + CBSDRefID2 + ',' + str(1) \
                                                    + ',' + CxG + "^" + CNG + ',' + CxG + "^" + \
                                                    NEG + ',' + CxG2 + "^" + CNG2 + ',' + CxG2 + "^" + NEG2
                                        f2.write(edgeOsama)
                                        f2.write('\n')
                                    # commscope
                                    if np.random.uniform() < thresh2:
                                        edgeOsama = CBSDRefID + ',' + CBSDRefID2 + ',' + str(1) \
                                                    + ',' + CxG + "^" + CNG + ',' + CxG + "^" + \
                                                    NEG + ',' + CxG2 + "^" + CNG2 + ',' + CxG2 + "^" + NEG2
                                        f3.write(edgeOsama)
                                        f3.write('\n')
        except:
            print("file error")

G = nx.Graph()
UniqueCxGs = []

if GraphGridOn:
    srccnt=0
    uniquepos=[]
    locused = np.zeros(num_init)

for NumBatch in range(NumberofBatch):

    # --------------
    # READ FILES
    # --------------
    Path1 = str(testlimit) + "adjlistOsamaFW" + str(NumBatch) + \
            "Batch" + ".txt"
    Path2 = str(testlimit) + "adjlistOsamaGoogle" + str(NumBatch) + \
            "Batch" + ".txt"
    Path3 = str(testlimit) + "adjlistOsamaCommscope" + str(NumBatch) + \
            "Batch" + ".txt"

    try:
        with open(Path1, 'r') as f1, open(Path2, 'r') as f2, \
                open(Path3, 'r') as f3:

            readCSV = csv.reader(f1, delimiter=',')
            Data1 = []
            for row in readCSV:
                if row:
                    Data1.append(row)
            readCSV = csv.reader(f2, delimiter=',')
            Data2 = []
            for row in readCSV:
                if row:
                    Data2.append(row)
            readCSV = csv.reader(f3, delimiter=',')
            Data3 = []
            for row in readCSV:
                if row:
                    Data3.append(row)
    except:
        print("file error")

    # --------------
    # RECONCILIATION
    # --------------
    SRCs = set([row[0] for row in Data1])
    nodecnt=0
    for cnt, SRC in enumerate(SRCs):
        D1rows = [row for row in Data1 if row[0] == SRC]
        if IncludePeerSAS:
            Dests = [row[1] for row in D1rows]
            D2rows = [row for row in Data2 if row[0] == SRC and row[1] not in Dests]
            Dests2 = [row[1] for row in D2rows]
            Dests = Dests + Dests2
            D3rows = [row for row in Data3 if row[0] == SRC and row[1] not in Dests]
            Dests3 = [row[1] for row in D3rows]
            OverallD = D1rows+D2rows+D3rows
        else:
            OverallD = D1rows

        for destcnt, row in enumerate(OverallD):

            CxGCNG1 = row[3]
            CxGCNG2 = row[5]
            NEG1 = row[4].split('^')[1]
            NEG2 = row[6].split('^')[1]
            CxG1 = row[3].split('^')[0]
            CxG2 = row[6].split('^')[0]
            LAT1 = float(row[7].split('^')[0])
            LON1 = float(row[7].split('^')[1])

            if destcnt == 0 and GraphGridOn:
                row = round((LAT1-41.2)/0.2); col = round((LON1+75.8)/0.2)
                entry = int(4.0*row+col)
                locused[entry] += 1
                print("locused[%d]=%d"%(entry, locused[entry]))
                print("LAT1=%f, LON1=%f -> row/col/entry=%d,%d,%d" % (LAT1, LON1, row, col, entry))

                if 1:
                    if locused[entry]%4 == 1:
                        LAT1 += 0.02
                        LON1 += 0.02
                    elif locused[entry]%4 == 2:
                        LAT1 += 0.02
                        LON1 -= 0.02
                    elif locused[entry]%4 == 3:
                        LAT1 -= 0.02
                        LON1 += 0.02
                    else:
                        LAT1 -= 0.02
                        LON1 -= 0.02

                G.add_node(CxGCNG1, pos=(LAT1, LON1))  # Position test 6/4
                G.add_edge(CxGCNG1, CxGCNG1, weight=1)
                srccnt += 1

            if NEG1 == 'DISABLE_NEG_MECH' or CxG1 != CxG2 or NEG1 != NEG2 or CxGCNG1 == CxGCNG2:
                # print("for CxGCNG1=",CxGCNG1, "for CxGCNG2=",CxGCNG2, "LAT1=", LAT1, "LON1=", LON1)
                G.add_edge(CxGCNG1, CxGCNG2, weight=1)
                if CxG1 not in UniqueCxGs:
                    UniqueCxGs.append(CxG1)
                if CxG2 not in UniqueCxGs:
                    UniqueCxGs.append(CxG2)
                #print('Khashayar CNG Test', cnt, 'CxGCNG1==',CxGCNG1,'CxGCNG2==',CxGCNG1)


# --------------
# CxG CNG Graph
# --------------
if show_graphs:
    #pos = nx.spring_layout(G, k=0.45, iterations=20) #Position test 6/4 old one
    if GraphGridOn:
        pos = nx.get_node_attributes(G, 'pos')
    else:
        pos = nx.spring_layout(G, k=0.45, iterations=20)
        #pos = nx.spring_layout(G)

    print("CxG^CNG Graph has Nodes=%d, Edges=%d"%(nx.number_of_nodes(G), nx.number_of_edges(G)))
    nx.draw(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=5)
    plt.show()

    for v in nx.nodes(G):
        print('Node:%s Degree:%d' % (v, nx.degree(G, v)))
    print("UniqueCxGs=", UniqueCxGs)

print("CxG^CNG graph has %d connected sets"%nx.number_connected_components(G))

# --------------
# Connected Set and Dictionary
# --------------
Gcliques = nx.graph_clique_number(G, cliques=None)
print("All Graph cliques ", Gcliques)
connectSets = []

for cs in nx.connected_components(G):

    print("operating on connected set: ", cs)

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
        CxGG.add_nodes_from((n, G.nodes[n]) for n in cs if n.split('^')[0] == CxG)
        CxGG.add_edges_from((n, nbr, d)
                            for n, nbrs in G.adj.items() if n in cs and
                            n.split('^')[0] == CxG
                            for nbr, d in nbrs.items() if nbr in cs and
                            nbr.split('^')[0] == CxG)

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

            if show_graphs:
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
                plt.title('Graph Coloring for ConnSet/CxG=%d/%s' % (cs_cnt,CxG))

                plt.show()
                #plt.savefig('test2.png')
            # -----------
            # Exact Coloring Using Integer Programming
            # -----------

            if IPexactColoring:

                n = 5 #guess the number of colors here you can use the networkx number of colors:
                #n = cxg_projected_set_maximum_number_colors
                # Adjacency Matrix
                B = nx.to_numpy_matrix(cs_CxGG)
                np.fill_diagonal(B, 0)
                C = B.tolist()

                g = [[int(c) for c in row] for row in C]
                nodes = range(len(g))
                # print('diag test g[4][4]',g[4][4])
                # print('g.graph=')
                # print(g.graph)

                # solution with same number of colors as DSATUR
                start = time.time()

                y = range(n)
                # initializes lp problem
                lp = LpProblem("Coloring Problem", LpMinimize)
                # The problem variables are created
                # variables x_ij to indicate whether node i is colored by color j;
                xij = LpVariable.dicts("x", (nodes, y), 0, 1, LpInteger)
                # variables yj to indicate whether color j was used
                yj = LpVariable.dicts("y", y, 0, 1, LpInteger)

                # objective is the sum of yj over all j
                obj = lpSum(yj[j] for j in y)
                lp += obj, "Objective Function"

                # constraint s.t. each node uses exactly 1 color
                for r in nodes:
                    jsum = 0.0
                    for j in y:
                        jsum += xij[r][j]
                    lp += jsum == 1, ""

                # constraint s.t. adjacent nodes do not have the same color
                for row in range(0, len(g)):
                    for col in range(0, len(g)):
                        if g[row][col] == 1:
                            for j in y:
                                lp += xij[row][j] + xij[col][j] <= 1, ""

                # constraint s.t. if node i is assigned color k, color k is used
                for i in nodes:
                    for j in y:
                        lp += xij[i][j] <= yj[j], ""

                # constrinat for upper bound on # of colors used
                lp += lpSum(yj[j] for j in y) <= n, ""

                # solves lp and prints optimal solution/objective value
                lp.solve()
                status = str(LpStatus[lp.status])
                print("Solution: " + status)

                print("Optimal Solution:")
                print("Xij=1 values:")
                for i in nodes:
                    for j in y:
                        if xij[i][j].value() == 1:
                            print(xij[i][j])

                print("Yj values:")
                for j in y:
                    print(yj[j], yj[j].value())
                print("Chromatic Number: ", value(lp.objective))
                end = time.time()
                print("IP with DSATUR-1 colors: %s seconds elapsed" % (end - start))

        print("maximum colors in CxG = %s cs = %s" % (CxG, cxg_projected_set_maximum_number_colors))

        # --------------
        # Dictionary
        # --------------
        if CxG == 'FDRTDWRLSS_SAS_0':
            FWCxGNumColor = cxg_projected_set_maximum_number_colors

        CxGColors.append(cxg_projected_set_maximum_number_colors)
        connected_set_total_colors += cxg_projected_set_maximum_number_colors
        connected_set_allocation.update({CxG: {"numberColors": cxg_projected_set_maximum_number_colors,
                                               "subgraphs": CxGsubgraphs
                                               }})

    connected_set_bandwidth_per_color = ChannelBW * \
                                       math.floor(
                                           (ALL_BW / connected_set_total_colors) / ChannelBW)
    connected_set_allocation['bandWidthperColor'] = connected_set_bandwidth_per_color
    for CxG in UniqueCxGs:
        connected_set_allocation[CxG]['frequencyRange'] = {}
        if CxG == 'FDRTDWRLSS_SAS_0':
            connected_set_allocation[CxG]['frequencyRange']['lowFrequency'] = 3550000000
            connected_set_allocation[CxG]['frequencyRange']['highFrequency'] = 3550000000 + \
                                                                               connected_set_bandwidth_per_color * \
                                                                               connected_set_allocation[CxG][
                                                                                   'numberColors']
    connectSets.append(connected_set_allocation)

    print("coloring across CxGs")
    print(CxGColors)

# --------------
# CHANNEL MAPPING
# --------------
for ConnectedSet in connectSets:
    FWCxGInfo = ConnectedSet['FDRTDWRLSS_SAS_0']['subgraphs']
    if len(FWCxGInfo) > 0:
        print("doing channel mapping for FW CxG\n")
        # Determining the larges connected component to determine the allocated BW for FW:
        FWCxGNumColor = ConnectedSet['FDRTDWRLSS_SAS_0']['numberColors']

        try:
            sumCxGColors = FWCxGNumColor + ConnectedSet['GOOGLE_SAS_0']['numberColors'] + ConnectedSet['COMMSCOPE_SAS_0']['numberColors']
        except KeyError:
            sumCxGColors = FWCxGNumColor
            print("Only FW is accepted as a CxG")

        BWPerColorFW = int((ALL_BW / Mhz) * FWCxGNumColor / sumCxGColors) / FWCxGNumColor

        for FWConnComp in FWCxGInfo:
            FWConnCompColors = max(FWConnComp.values())+1
            # find CBSDs in each color
            CBSDsInColor = [[] for i in range(FWConnCompColors)]
            distances = [[] for i in range(FWConnCompColors)]

            #cbsds = cbsdCollection.find(myquery).sort('_id').limit(testlimit)
            cbsds = TestcbsdCollection.find(myquery).sort('_id').limit(testlimit)
            for cnt, cbsd in enumerate(cbsds):
                cbsdReferenceId = \
                    cbsd['features'][0]['properties']['cbsdReferenceId']
                cbsdId = \
                    cbsd['features'][0]['properties']['cbsdId']
                groupingParam = cbsd['features'][0]['properties']['groupingParam']
                for group in groupingParam:
                    if group['groupType'] == 'CXG':
                        CxG = group['groupId']
                    if group['groupType'] == 'CNG':
                        CNG = group['groupId']
                        #print('Pirouz = ',CNG)

                if CxG == 'FDRTDWRLSS_SAS_0':
                    MyCxGCNG = 'FDRTDWRLSS_SAS_0^'+CNG
                    color = FWConnComp.get(MyCxGCNG)
                    if color != None:
                        CBSDsInColor[color].append(cbsdReferenceId)
                        # NEW = Extract distance from distance collection
                        query = {"cbsdReferenceId": cbsdReferenceId}
                        distancequery = cbsdDistance.find(query)
                        for distance in distancequery:
                            distances[color].append(distance)

            # Form distance numpy arrays
            distancenp = []
            for distancec in distances:
                b6alt2 = np.zeros((len(distancec), NumberOfChannels))
                for dcnt, distance in enumerate(distancec):
                    ppaDist = distance['ppaDist']
                    gwpzDist = distance['gwpzDist']
                    ezDist = distance['ezDist']
                    dpaDist = distance['dpaDist']
                    fssDist = distance['fssDist']
                    escDist = distance['escDist']
                    intDist = [a * b for a, b in zip(ppaDist, gwpzDist)]
                    intDist = [a * b for a, b in zip(ezDist, intDist)]
                    intDist = [a * b for a, b in zip(dpaDist, intDist)]
                    intDist = [a * b for a, b in zip(fssDist, intDist)]
                    intDist = [a * b for a, b in zip(escDist, intDist)]
                    finDist = np.array(intDist)
                    b6alt2[dcnt, :] = finDist
                distancenp.append(b6alt2)

            # Determine channels/color - allocate guard band to colors with largest # CBSDs
            chans_per_color = int(math.floor((BWPerColorFW*FWCxGNumColor/FWConnCompColors)/(ChannelBW/Mhz)))
            print("Starting chans_per_color=%d" % chans_per_color)
            Starting_chans = [chans_per_color] * FWConnCompColors
            print(Starting_chans)
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
                colors_w_extra5 = int((BWPerColorFW*FWCxGNumColor-(ChannelBW/Mhz)*chans_per_color*FWConnCompColors)/(ChannelBW/Mhz))
                print("colors with Extra 5 MHz=%d" % colors_w_extra5)

                for color in range(0,colors_w_extra5):
                    Starting_chans[NN[color]] += 1
                print("channels for each color after extra BW added")
                print(Starting_chans)

            if perform_channel_mapping:
                start = time.time()
                # perform channel mapping algorithm
                LL = [i for i in range(0, NumberOfChannels)]
                bestnonzero = math.inf
                bestscore = - math.inf
                for mapping in list(permutations(range(0, FWConnCompColors))):
                    T = [Starting_chans[i] for i in mapping]
                    U = list(accumulate(T))
                    U.insert(0, 0)
                    W = [LL[U[i]:U[i+1]] for i in range(len(U)-1)]
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
                        b7 = b6[W[c], :]  # should be mostly ones, look at this and confirm for the winning mapping (combination)
                        # minimum in each row
                        b8 = np.amin(b7, axis=0)
                        # product of terms
                        b9 = np.prod(b8) #when it is zero do not put it product

                        # aggregate product
                        b10 = b10*b9
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


print('final test',connectSets)