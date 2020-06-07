"""
Created on Wed Oct 30 20:34:54 2019

Author: Gabriel Fuentes Lezcano
"""

from sklearn.cluster import DBSCAN
import numpy as np
import geopandas as gpd
from shapely.geometry import *
import pandas as pd
import traj_dist.distance as tdist
from shapely.ops import split
from datetime import timedelta
from mpi4py import MPI

def chunk(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out
##MPI machinery
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name=MPI.Get_processor_name()


if rank==0:    
    df_base=pd.DataFrame(columns=["imo","mmsi","transit_bound","initial_anchoring_area","time_anchoring_in","time_anchoring_out","draught_in","draught_out","access","time_at_entrance",
                                  "first_time_out_canal","direct_transit_boolean","comment_on_stoppage"]).to_csv("suez_canal_transits2.csv",mode="a",index=False)
    df_failure=pd.DataFrame(columns=["imo","fail","time"]).to_csv("suez_canal_transits_failures2.csv",mode="a",index=False)
    suez_poly=gpd.read_file("suez_polygons.geojson")
    vessels_all=pd.read_csv("vessels_all_specs_no_duplicates.csv")
    terrestrial=pd.read_csv("vessel_list_terrestrial.csv").imo.tolist()
    terrestrial=[int(i) for i in terrestrial]
    mask_imo_mmsi=vessels_all.imo.isnull()
    imo=vessels_all[~mask_imo_mmsi].imo.unique().tolist()
    imo=[int(i) for i in imo]
    for i in imo:
        if i in terrestrial:
            imo.remove(i)
    imo=chunk(imo,size)
    terrestrial=chunk(terrestrial,size)
    mmsi=vessels_all[mask_imo_mmsi].mmsi.unique().tolist()
    mmsi=chunk(mmsi,size)
    routes=gpd.read_file("suez_routes.geojson")
    said_access_pol=LineString([(32.3772,31.3214),(32.3697,31.2732),(32.3492,31.1994),(32.3356,31.1942)])
    said_access=gpd.GeoDataFrame([["Suez Container Terminal Access",np.nan,said_access_pol]],geometry=2)
    said_access.columns=routes.columns
    routes=pd.concat([routes,said_access]).reset_index(drop=True)
    access_routes=gpd.read_file("access_routes.geojson")

else:
    imo=None
    terrestrial=None
    mmsi=None
    suez_poly=None
    routes=None
    access_routes=None

imo=comm.scatter(imo,root=0)
terrestrial=comm.scatter(terrestrial,root=0)
mmsi=comm.scatter(mmsi,root=0)
suez_poly=comm.bcast(suez_poly,root=0)
routes=comm.bcast(routes,root=0)
access_routes=comm.bcast(access_routes,root=0)

def iteration_clustered(iterator):
    if iterator==imo:
        nan=0
    elif iterator==mmsi:
        nan=1
    elif iterator==terrestrial:
        nan=2
    df_values=[]
    fail_df=[]
    for ship in iterator:
        if nan==0:
            positions=pd.read_csv("/home/gabriel/Ships_Position_Med/{}.csv".format(ship),usecols=["imo","mmsi","timestamp_position","lon","lat","speed","draught"])
        elif nan==1:
            positions=pd.read_csv("/home/gabriel/Ships_Position_Med/mmsi/m{}.csv".format(ship),usecols=["imo","mmsi","timestamp_position","lon","lat","speed","draught"])
        elif nan==2:
            positions=pd.read_csv("/home/gabriel/Ships_Position_Med/terrestrial_merge/a{}.csv".format(ship),usecols=["imo","mmsi","timestamp_position","lon","lat","speed","draught"])
            positions=positions.assign(draught=np.where(positions.draught.isnull(),0.1,positions.draught))
        positions["timestamp_position"]=pd.to_datetime(positions["timestamp_position"],format="%Y-%m-%d %H:%M:%S")
        ##To assute values are sorted by date
        positions.sort_values(by='timestamp_position',inplace=True)
        positions.reset_index(drop=True, inplace=True)
        ##Geodataframe of csv with lon and lat as Points
        positions_gdf=gpd.GeoDataFrame(positions,geometry=[Point(x,y) for x,y in zip(positions.lon,positions.lat)])
        positions_gdf.crs = {'init' :'epsg:4326'}
        ##Spatial join of vessel to suez polygons
        port_stops=gpd.sjoin(positions_gdf,suez_poly,how="left")
        ##Remvoing overlaping positions in two polygons. Keep 1
        port_stops=port_stops.drop_duplicates(subset=["imo","mmsi","timestamp_position","lon","lat","speed","draught","geometry"])   
        
        if port_stops.name.notnull().any()==True:
            ###Segmented anchorage and full south anchorage
            port_stops_segmented_anch=port_stops.drop_duplicates(subset=["timestamp_position"])
            port_stops_in=port_stops_segmented_anch.replace(value="South Anchorage",to_replace=["1C-5C S Anchorage","Green Island S Anchorage","E13-E21 S Anchorage",
                                           "Big vessels anchorage","1H-2H S Anchorage","E1-E12 S Anchorage","W1-W14 S Anchorage"])
        
            continuous_call=port_stops_in['name'] != port_stops_in['name'].shift(1)
            port_stops_in['subgroup'] = (continuous_call).cumsum()
            #Removal positions with no record of visiting a port
            port_stops_in=port_stops_in[~port_stops_in["name"].isnull()]
            #Separate continuos visits but in different times. Ships turning off AIS after Suez and coming back to Suez
            time_subgroup=(port_stops_in.timestamp_position-port_stops_in.timestamp_position.shift(1))/np.timedelta64(1,'h')>72
            port_stops_in['subgroup2'] = (time_subgroup).cumsum()
            ##Create a subgroup only when subgroup or subgroup are not sequence of the previous
            mask_duplicates=port_stops_in.duplicated(subset=["subgroup","subgroup2"])
            port_stops_in["subgroup"]=(~mask_duplicates).cumsum()
            ##Subgroup 2 column no longer needed
            ###Segmented anchorage and full south anchorage
            port_stops_in.drop(columns=["subgroup2"],inplace=True)
            port_stops_in.reset_index(inplace=True)         
            mask_continuity_first=port_stops_in['subgroup'] != port_stops_in['subgroup'].shift(1)
            check_points=port_stops_in[mask_continuity_first]
            #Check for start and end of transits
            range_indices=check_points[(check_points["name"]=="North Anchorage")|(check_points["name"]=="South Anchorage")]
            range_indices=range_indices.assign(same=(~(range_indices.name==range_indices.name.shift(1))).cumsum())
            range_indices=range_indices.assign(time_f=range_indices.groupby("same")["timestamp_position"].apply(lambda x: x-x.shift(1)))
            ##Test for sequence of same polygon less than 72 hours. Check if it went to a port or if shifted.
            ##It will leave higher than 72 and null. Removes 1st if a port visit in between. OW remove second
            #modelling that the anchors merged
            refine_indices_fr=range_indices.index[range_indices.time_f.shift(-1)<timedelta(hours=72)].tolist()
            refine_indices_to=range_indices.index[range_indices.time_f<timedelta(hours=72)].tolist()
            test_refine=[port_stops_in.loc[refine_indices_fr[x]:refine_indices_to[x],"name"].unique().tolist() for x in range(len(refine_indices_fr))]
            suez_ports=["Port Said","Port Fuad","Said Container Terminal","Port Suez","Port Taofik"]
            test_refine=[any (e in suez_ports for e in test_refine[x]) for x in range(len(test_refine))]
            remove_indices=[refine_indices_fr[x] if test_refine[x]==True else refine_indices_to[x] for x in range(len(test_refine))]
            range_indices.drop(index=remove_indices,inplace=True)
            #Checks if value in row is equal to the previous, that cuts a line as where does a transit starts
            range_indices=range_indices.assign(equal_to_previous=(range_indices.name==range_indices.name.shift(1)).cumsum())
            range_indices.reset_index(inplace=True)
            range_indices.rename(columns={"level_0":"i_port_stops_in","index":"i_port_stops"},inplace=True)
            range_indices.drop(columns=["subgroup","time_f","same"],inplace=True)
           ##Filter out, solo groups
            range_indices_not_defined=range_indices.groupby('equal_to_previous').filter(lambda x : len(x)>=3)
            #Get the ok values based in origin destination being correctly identified 
            range_indices=range_indices.groupby('equal_to_previous').filter(lambda x : len(x)==2)
            ##Avoid passing empty dataframes with condition of more than 2 observations, filter them first.
            
    ##Testing if combinations of two are valid by time difference
            if range_indices.shape[0]!=0:
                ###Get the last position of the same subgroup to carry the test from last at 
                ###anchorage and first in the other.
                range_indices.reset_index(inplace=True)
                equal_to_previous_series=range_indices.copy().equal_to_previous.tolist()
                replace_index=range_indices.groupby('equal_to_previous').first().set_index("index")
                subgroups=[port_stops_in.loc[x,"subgroup"] for x in replace_index["i_port_stops_in"].tolist()]
                last_of_subgroups=port_stops_in.copy()[port_stops_in.subgroup.isin(subgroups)].reset_index().groupby("subgroup").last().set_index([replace_index.index.tolist()],drop=True)
                last_of_subgroups.rename(columns={"level_0":"i_port_stops_in","index":"i_port_stops"},inplace=True)
                last_of_subgroups=last_of_subgroups.assign(equal_to_previous=np.nan)
                range_indices=range_indices[~range_indices["index"].isin(replace_index.index.tolist())].set_index(["index"])
                last_of_subgroups.columns=range_indices.columns
                range_indices=pd.concat([range_indices,last_of_subgroups]).sort_index()
                range_indices["equal_to_previous"]=equal_to_previous_series
                
                range_indices=range_indices.assign(differences=range_indices.groupby("equal_to_previous")["timestamp_position"].diff(periods=1)/np.timedelta64(1,'h')) 
                range_indices=range_indices[~(range_indices.differences>96)]##Accepts null and lower than 96 ###NOT DECLARED. REMOVE
                subgroup_remove=range_indices.equal_to_previous.value_counts()<=1
                list_of_valid_groups=subgroup_remove.index[range_indices.equal_to_previous.value_counts()>1].tolist()
                range_indices=range_indices[range_indices.equal_to_previous.isin(list_of_valid_groups)]
                range_indices.drop(columns=["differences"],inplace=True) 
            
            if range_indices_not_defined.shape[0]!=0:
                ##Calculate differences in time and the biggest time difference sets the index that has to be removed
                range_indices_not_defined=range_indices_not_defined.assign(differences=range_indices_not_defined.groupby("equal_to_previous")["timestamp_position"].diff(periods=1)/np.timedelta64(1,'h')) 
                ##Periods with 96 hours between them, index for start and end of sequence. Filters out individual call with no full transit info
                start_not_defined=range_indices_not_defined.loc[range_indices_not_defined.differences.shift(-1)<96]
                end_not_defined=range_indices_not_defined[range_indices_not_defined.differences<96]
                ##Filter could lead 
                if start_not_defined.shape[0]!=0:
                    ##Define initial and end of transit
                    start_index_not_defined=start_not_defined.index.tolist()
                    end_index_not_defined=end_not_defined.index.tolist()
                    range_indices_not_defined=pd.concat([start_not_defined,end_not_defined])
                    range_indices_not_defined.sort_index(inplace=True)            
                    
                    for x in range(len(start_not_defined)):
                        range_indices_not_defined.loc[start_index_not_defined[x],"equal_to_previous"]= x
                        range_indices_not_defined.loc[end_index_not_defined[x],"equal_to_previous"]=  x                    
                    range_indices_not_defined.drop(columns=["differences"],inplace=True)
                    ##Merge all the range indices, now assured that pairs are alligned
                    range_indices=pd.concat([range_indices_not_defined,range_indices]).sort_index()
                    ##equal to previous redo
                    range_indices['equal_to_previous'] = (range_indices.equal_to_previous!=range_indices.equal_to_previous.shift(1)).cumsum()
                    ##Just in the case of a subgroup with more than 2 values
                    if any(range_indices.groupby('equal_to_previous').apply(lambda x : len(x)>=3))==True:          
                        range_indices=range_indices.assign(differences=range_indices["timestamp_position"].diff(periods=1)/np.timedelta64(1,'h'))
                        range_indices=range_indices.assign(equal_to_previous=(range_indices.differences>96).cumsum())
                        range_indices.drop(columns=["differences"],inplace=True)
                        ###Too complicated, break the loop
                        if any(range_indices.groupby('equal_to_previous').apply(lambda x : len(x)>=3))==True:
                            break           
            ##Test, if non value returned in range_indices it means no complete transit is observed, go to next iterated value
            if range_indices.shape[0]>1:
                ##Readjust range_indices from first of first index
                range_indices.reset_index(inplace=True)
                equal_to_previous_series=range_indices.copy().equal_to_previous.tolist()
                replace_index=range_indices.groupby('equal_to_previous').first().set_index("index")
                subgroups=[port_stops_in.loc[x,"subgroup"] for x in replace_index["i_port_stops_in"].tolist()]
                first_of_subgroups=port_stops_in.copy()[port_stops_in.subgroup.isin(subgroups)].reset_index().groupby("subgroup").first().set_index([replace_index.index.tolist()],drop=True)
                first_of_subgroups.rename(columns={"level_0":"i_port_stops_in","index":"i_port_stops"},inplace=True)
                first_of_subgroups=first_of_subgroups.assign(equal_to_previous=np.nan)
                range_indices=range_indices[~range_indices["index"].isin(replace_index.index.tolist())].set_index(["index"])
                first_of_subgroups.columns=range_indices.columns
                range_indices=pd.concat([range_indices,first_of_subgroups]).sort_index()
                range_indices["equal_to_previous"]=equal_to_previous_series         
                
                indices_list=pd.DataFrame(range_indices.groupby("equal_to_previous")["i_port_stops_in"].apply(list).tolist(),columns=["start","end"],dtype="object")
                while indices_list.end.iloc[-1]==None:
                    indices_list.drop(indices_list.tail(1).index,inplace=True)
                if indices_list.shape[0]!=0:
                    ##Subgroup of last range
                    value_of_last=port_stops_in.loc[indices_list.end.iloc[-1],"subgroup"]
                    ##Max value of the subgroup of last value
                    max_df=port_stops_in[port_stops_in.subgroup==value_of_last]
                    if max_df.shape[0]==1:
                        max_index=max_df.index[0]
                    else:
                        max_index=max_df.index[-1]
                    transit_test=pd.DataFrame(range_indices.groupby("equal_to_previous")["name"].apply(list).tolist(),columns=["start","end"])
                    #Test if values inside every list, testing a full transit
                    transit_test=((transit_test.start=="North Anchorage")&(transit_test.end=="South Anchorage"))|((transit_test.start=="South Anchorage")&(transit_test.end=="North Anchorage"))
                    transit_df=pd.concat([indices_list,transit_test],axis=1)
                    transit_df.columns=["start","end","full_transit"]
                    ##As the ranges are incomplete in the last sequence, the maximum value of the original df index is replaced with the last index absorbed in the subsequent tests
                    if transit_df.shape[0]==1:  
                        transit_df.replace(transit_df.end.iloc[0],max_index,inplace=True)
                    else:
                        transit_df.replace(transit_df.end.iloc[-1],max_index,inplace=True)
                    transit_df["transit_number"]=transit_df["full_transit"].astype(int)
                    transit_df.transit_number.replace(0,np.nan,inplace=True)
                    transit_df["transit_number"]=transit_df["transit_number"].cumsum().astype("Int64")
                    
                    ##Fix last value of range
                    transit_df["end"]=transit_df["end"].apply(lambda x: port_stops_in.groupby("subgroup").get_group(port_stops_in.loc[x,"subgroup"]).index[-1])
                    ##Creates empty columns to append with values of the test of transit and assign a transit number         
                    port_stops_in=port_stops_in.assign(full_transit=np.nan)
                    port_stops_in=port_stops_in.assign(transit_number=np.nan)
                    ##Fill with values of full transit and count the transit number based in the predefined combinations of ranges and full transit cumulative
                    for x in transit_df.index:
                        port_stops_in.loc[transit_df.start.loc[x]:transit_df.end.loc[x],"full_transit"]=transit_df.full_transit.loc[x]
                        port_stops_in.loc[transit_df.start.loc[x]:transit_df.end.loc[x],"transit_number"]=transit_df.transit_number.loc[x]                  
                    ##A complete transit is the one that has being assigned a cummulative number, the others means no case for our study
                    complete_transit=port_stops_in[port_stops_in.transit_number.notnull()]
                    ##Groupby every transit
                    transit_index=complete_transit.transit_number.unique().tolist()
                    complete_t_group=complete_transit.groupby("transit_number")
                    ##Revise by transit number               
                    for transit in transit_index:
                        group_of_complete=complete_t_group.get_group(transit)
                        ##Get the ordered transit polygons visit
                        list_of_polygon_visit=group_of_complete.name.unique().tolist()
                        #Creates a new list with the former filter and checks if the transit has all the points
                        valid_transit=["South Anchorage","South Access","North Access West"]
                        valid_transit_n=["North Anchorage","North Access West","South Access"]
                        valid_transit_alt=["South Anchorage","South Access","North Access East"]
                        valid_transit_altn=["North Anchorage","North Access East","South Access"]
                        valid_transit_alt2=["South Anchorage","South Access","Said Container Terminal Access"]
                        valid_transit_altn2=["North Anchorage","Said Container Terminal Access","South Access"]
                        ##Check if all the correct transit polygons are touched by the transit groupby after the filter
                        ##Check if all the correct transit polygons are touched by the transit groupby after the filter
                        test_valid_transit=all(x in list_of_polygon_visit for x in valid_transit)
                        test_valid_transit_n=all(x in list_of_polygon_visit for x in valid_transit_n)
                        test_valid_transit_alt=all(x in list_of_polygon_visit for x in valid_transit_alt)
                        test_valid_transit_altn=all(x in list_of_polygon_visit for x in valid_transit_altn)
                        test_valid_transit_alt2=all(x in list_of_polygon_visit for x in valid_transit_alt2)
                        test_valid_transit_altn2=all(x in list_of_polygon_visit for x in valid_transit_altn2)
                        ##Values as input for new dataframe appended to an empty list
                        
                        if test_valid_transit==True or test_valid_transit_alt==True or test_valid_transit_alt2==True or test_valid_transit_n==True or test_valid_transit_altn==True or test_valid_transit_altn2==True :
                            anchorage_area_name=group_of_complete.name.iloc[0]
                            if anchorage_area_name=="South Anchorage":
                                tidal_stream=0.0061 ##2.2 knots
                            elif anchorage_area_name=="North Anchorage":
                                tidal_stream=0.0044 ##1.6 knots
                            else:
                                break
                        #Geometry to array to insert into dbscan
                        #Find anchoraing spots as no advancement of more than 0.1 miles (knots/hour) every 30 min. transformation to degree=0.005
                            clustering=DBSCAN(algorithm='auto', eps=tidal_stream, leaf_size=30, metric='euclidean',metric_params=None, min_samples=3, n_jobs=None, p=None) 
                            def anchor_cluster(lon):
                            ##DBScan for recognizing static vessels
                            ##Geometry to array to insert into dbscan
                                X = [np.array([port_stops_in.lon.loc[x],port_stops_in.lat.loc[x]]) for x in lon.index]
                                clustering.fit(X)
                                ##Labels of clustering areas, -1 indicates outlier
                                return clustering.labels_
                            ##Do the dbscan to every subgroup(stops)
                            group_of_complete=group_of_complete.assign(anchoring_cluster=group_of_complete.groupby("subgroup")["lon"].transform(anchor_cluster))
                            #Replace the non clusters spots with nan values
                            group_of_complete.anchoring_cluster.replace(-1,np.nan,inplace=True)
                            ##Group by anchoring clusters
                            group_of_complete["anchoring_cluster"]=(group_of_complete["anchoring_cluster"] != group_of_complete["anchoring_cluster"].shift(1)).cumsum()
                            #Check whcih of the cluster was worngly counted, with a test that a cluster was defined as groups of 3 points or more
                            mask_cluster=group_of_complete.anchoring_cluster.value_counts()[group_of_complete.anchoring_cluster.value_counts()<3].index.tolist()
                            ##Replaces all the cluster less than sample with nan values
                            group_of_complete.loc[group_of_complete["anchoring_cluster"].isin(mask_cluster),"anchoring_cluster"]=np.nan
                            
                            #If anchored in designated anchorage. Needed for waiting time.
                            if len(group_of_complete[(group_of_complete.name==group_of_complete.name.iloc[0])&(group_of_complete.anchoring_cluster.notnull())].anchoring_cluster.unique().tolist())>=1:
                                #Joins anchoring shifting and recognize it as one
                                sequence=group_of_complete.groupby("subgroup").apply(lambda x: x.iloc[0])
                                anchor_visit=sequence["index"][sequence.name == group_of_complete.name.iloc[0]].tolist()
                                anchor_visit_not=sequence[(sequence["index"]>=anchor_visit[0])&(sequence["index"]<=anchor_visit[-1])&(sequence.name!=group_of_complete.name.iloc[0])]
                                test_port_visit=anchor_visit_not.name.tolist()
                                
                                ##Remove outliers
                                mean_anch=np.mean(group_of_complete["speed"][group_of_complete.anchoring_cluster==1])
                                st_anch=np.std(group_of_complete["speed"][group_of_complete.anchoring_cluster==1])
        
                                th=mean_anch+st_anch
        
                                group_of_complete["anchoring_cluster"]=np.where((group_of_complete.anchoring_cluster==1)&(group_of_complete.speed>th),-1,group_of_complete.anchoring_cluster)
                                                           
                                ##Models the event of going to a port and returning to anchorage
                                if "Port Said" in test_port_visit or "Port Fuad" in test_port_visit or "Said Container Terminal" in test_port_visit\
                                or "Port Suez" in test_port_visit or "Port Taofik" in test_port_visit:
                                    first_at_anch=group_of_complete.index[group_of_complete["index"]==anchor_visit[-1]][0]
                                else:
                                    first_at_anch=group_of_complete[group_of_complete.name==group_of_complete.name.iloc[0]].anchoring_cluster.first_valid_index()
                                last_at_anch=group_of_complete[group_of_complete.name==group_of_complete.name.iloc[0]].anchoring_cluster.last_valid_index()
                        
                                group_of_complete.loc[first_at_anch:last_at_anch,"anchoring_cluster"]=1
                                group_of_complete.loc[:first_at_anch,"anchoring_cluster"]=np.nan
                                first_at_anch=group_of_complete.loc[first_at_anch,"index"]
                                last_at_anch=group_of_complete.loc[last_at_anch,"index"]                            
                                ####Test if anchorage is complete
                                ##Get first pos before and after anchorage. The approx and leaving pos validates full anch.
                                if first_at_anch==group_of_complete["index"].iloc[0] or last_at_anch==positions.shape[0]-1:
                                    break
                                else:
                                    bf_anch=positions.loc[first_at_anch,"timestamp_position"]-positions.loc[first_at_anch-1,"timestamp_position"]
                                    af_anch=positions.loc[last_at_anch+1,"timestamp_position"]-positions.loc[last_at_anch,"timestamp_position"]                      
                                ###Test if approach and leave anch positions exists
                                if bf_anch <= timedelta(minutes=30) and af_anch <= timedelta(minutes=30):
                                    group_of_complete=group_of_complete.assign(subgroup=(group_of_complete.name!=group_of_complete.name.shift(1)).cumsum())
                                    ##Identify the name of the anchoring area
                                    anchorage_area_sub=group_of_complete.subgroup.iloc[0]
                                    last_pos_in_anch=port_stops.geometry.loc[last_at_anch]
                                    
                                    if anchorage_area_name=="South Anchorage":
                                        first_in_canal_base=group_of_complete["index"][(group_of_complete.name=="South Access")&(group_of_complete.lat>=29.93)]##Gets the first position inside the canal after the entrance
                                        first_in_canal_base=first_in_canal_base[first_in_canal_base>last_at_anch]
                                        if first_in_canal_base.shape[0]!=0:
                                            first_in_canal_base=first_in_canal_base.iloc[0]
                                            first_in_canal=first_in_canal_base
                                            first_pos_in_canal=port_stops.geometry.loc[first_in_canal]
                                            accessed_canal_name=port_stops.name.loc[first_in_canal]
                                            canal_linestring=list(routes.geometry.iloc[1].coords)
                                            canal_linestring=routes.geometry.iloc[1]
                                            access_anch=port_stops.name.loc[last_at_anch]
                                            access_name="South Access"
                                            ##Info of route that could serve to estimate speed                                
                                            incompl_route_full=port_stops.loc[last_at_anch:first_in_canal,:]
                                            ## Generate time difference between positions
                                            incompl_route_full=incompl_route_full.assign(diff=incompl_route_full.timestamp_position-incompl_route_full.timestamp_position.shift(1))
                                            incompl_route=LineString(incompl_route_full.geometry.to_list())
                                            #Draught in
                                            draught_in=incompl_route_full.draught.mode().iloc[0]
                                            ##Acces point to Canal
                                            lon=32.5624
                                            lat=29.9318
                                            s_point_access=Point(lon,lat)
                                            ##Best scenario. Complete route from anchorage to after checkpoint and route merging regular route
                                            if any(incompl_route_full["diff"]>timedelta(minutes=20))== False and incompl_route.intersects(canal_linestring)==True:
                                                inter=[incompl_route.intersection(canal_linestring)][0].buffer(0.0000000001) ## Buffer to cut line by very small polygon
                                                incompl_route=list(split(incompl_route,inter)[0].coords)
                                                canal_linestring=list(split(canal_linestring,inter)[2].coords)
                                                merged_lines=LineString(incompl_route+canal_linestring)
                                                
                                                ##Calculate distance
                                                incompl_route_full=incompl_route_full.assign(miles_travelled=incompl_route_full.geometry.apply(lambda x:merged_lines.project(x)*60))
                                                columns=incompl_route_full.columns.tolist()
                                                
                                            else:   
                                                incompl_route=[np.array(incompl_route.coords)]
                                                canal_linestring=list(routes.geometry.iloc[1].coords)
                                                ##Absorbs access canals and transform it to geometry
                                                
                                                if access_anch == "W1-W14 S Anchorage":
                                                    access_linestring_base=access_routes[access_routes.access_number==5]
                                                elif access_anch == "Big vessels anchorage":
                                                    access_linestring_base=access_routes[access_routes.access_number==1]
                                                elif access_anch == "E13-E21 S Anchorage":
                                                    access_linestring_base=access_routes[access_routes.access_number==3]
                                                elif access_anch == "E1-E12 S Anchorage":
                                                    access_linestring_base=access_routes[access_routes.access_number==2]
                                                elif access_anch == "1C-5C S Anchorage":
                                                    access_linestring_base=access_routes[access_routes.access_number==0]
                                                elif access_anch == "Green Island S Anchorage":
                                                    access_linestring_base=access_routes[access_routes.access_number==4]                                 
                                                    ##Filter out access with lower draft than vessel
                                                access_linestring_df=access_linestring_base[access_linestring_base.draught>=draught_in]
                                                if access_linestring_df.shape[0]<10:## If the access linestring is empty because there is no line with minimum draught, 
                                                    ##then use the 20 largest lines as reference
                                                    access_linestring_df=access_linestring_base.nlargest(20,"draught")                      
                                                access_linestring=access_linestring_df.geometry.tolist()
                                            #Creates a list of array to calculate distances
                                                access_linestring=[ np.array((geom.xy[0], geom.xy[1])).transpose() for geom in access_linestring ]
                                            ##My incomplete route is assesed distance with all the exisiting access and returns the closest access                            
                                                closest_line = tdist.cdist(incompl_route,access_linestring, metric="sspd").transpose().argmin()
                                                unique_line_access=list(access_linestring_df.geometry.iloc[closest_line].coords)
                                            ##Joins the canal route and the access route
                                                merged_lines=LineString(unique_line_access+canal_linestring)
                                                closest_point_in_line=merged_lines.interpolate(merged_lines.project(last_pos_in_anch))
                                                intersected_line=split(merged_lines,closest_point_in_line.buffer(0.000000001))
                                            ##Cuts the line from the new starting point to the whole access route
                                                if len(intersected_line)<=2:
                                                    merged_lines=LineString(list(intersected_line[-2].coords)+list(intersected_line[-1].coords))
                                                else:
                                                    merged_lines=LineString(list(intersected_line[1].coords)+list(intersected_line[2].coords))
                                                distance_old_new=closest_point_in_line.distance(last_pos_in_anch)*60
                                             #Ready the linestring cutted and the distance from old and new points. Need to identify the position inside the canal
                                                entrance_canal_distance=merged_lines.project(s_point_access)*60
                                                dist_first_pos_in_canal=merged_lines.project(first_pos_in_canal)*60
                                                
                                                ##Calculate distance of every point
                                                incompl_route_full=incompl_route_full.assign(miles_travelled=incompl_route_full.geometry.apply(lambda x:merged_lines.project(x)*60))
                                                columns=incompl_route_full.columns.tolist()
                                                incompl_route_full=incompl_route_full.assign(miles_travelled=incompl_route_full.miles_travelled+distance_old_new)
                                            
                                            entrance_info=pd.DataFrame([incompl_route_full.imo.iloc[-1],incompl_route_full.mmsi.iloc[-1],
                                                                        np.nan,lon,lat,np.nan,incompl_route_full.draught.max(),s_point_access,np.nan,
                                                                        "Entrance",np.nan,np.nan,merged_lines.project(s_point_access)*60]).T
                                            
                                            entrance_info.columns=incompl_route_full.columns
                                            incompl_route_full=incompl_route_full.assign(speed=(incompl_route_full.miles_travelled-incompl_route_full.miles_travelled.shift(1))/((incompl_route_full.timestamp_position-incompl_route_full.timestamp_position.shift(1))/np.timedelta64(1,'h')))
                                            incompl_route_full=pd.concat([incompl_route_full,entrance_info]).sort_values(by="miles_travelled").reset_index(drop=True)
                                            entrance_index=incompl_route_full.index[incompl_route_full.name=="Entrance"].tolist()[0]
                                            distances_x=[incompl_route_full.loc[entrance_index-1:entrance_index+1,"miles_travelled"]]
                                            if len(distances_x[0])==3:
                                                ##Interpolate. Full sequence
                                                distances_x=[incompl_route_full.loc[entrance_index-1,"miles_travelled"],incompl_route_full.loc[entrance_index+1,"miles_travelled"]]
                                                time_x=[0,(incompl_route_full.loc[entrance_index+1,"timestamp_position"]-incompl_route_full.loc[entrance_index-1,"timestamp_position"])/np.timedelta64(1,"h")]
                                                interp_time=np.interp(incompl_route_full.loc[entrance_index,"miles_travelled"],distances_x,time_x)
                                                incompl_route_full.loc[entrance_index,"timestamp_position"]=incompl_route_full.loc[entrance_index-1,"timestamp_position"]+timedelta(hours=interp_time)
                                            elif len(distances_x[0])==2:  ## Having two means that the last value is entrance. 
                                                #Since we requested to have first positions higher than check point, 
                                                #it means that the entrance positions is equal to the checkpoint.
                                                incompl_route_full.loc[entrance_index-1,"name"]="Entrance"
                                                incompl_route_full.drop(incompl_route_full.index[entrance_index],inplace=True)
                                            else:
                                                break                                                 
                                            ##Interpolate exit at the other side
                                            ##Gets first position out of canal after checkpoint. Calucalte average distance and interpolate position time
                                            first_position_out_filter=(group_of_complete.lat>31.3214).idxmax()
                                            bef_exit_filter=(group_of_complete.lat<=31.3214)[::-1].idxmax()                                 
                                            average_exit_speed=(group_of_complete.speed[first_position_out_filter]+group_of_complete.speed[bef_exit_filter])/2        
                                            if group_of_complete.name.loc[bef_exit_filter]== "North Access East":
                                                exit_point=Point(32.3725,31.3214)
                                                exit_linestring=LineString(list(routes.geometry.iloc[2].coords)[::-1])
                                            elif group_of_complete.name.loc[bef_exit_filter]== "Suez Container Terminal Access":
                                                exit_point=Point(32.3772,31.3214)
                                                exit_linestring=LineString(list(routes.geometry.iloc[6].coords)[::-1])
                                            else:
                                                exit_point=Point(32.3653,31.3214)
                                                exit_linestring=LineString(list(routes.geometry.iloc[0].coords)[::-1])
                                            dist_thres=(exit_linestring.project(exit_point)-exit_linestring.project(group_of_complete.geometry.loc[bef_exit_filter]))*60
                                            if average_exit_speed==0:
                                                break
                                            else:
                                                time_thres=timedelta(seconds=(dist_thres/average_exit_speed)*3600)
    #                                        ##Construction of missing values for df
                                            calculated_time_exit=group_of_complete.timestamp_position.loc[bef_exit_filter]+time_thres
                                            draught_out=group_of_complete.draught[first_position_out_filter]
                                            ship_mmsi=group_of_complete.mmsi[first_position_out_filter]
                                            time_in=incompl_route_full.timestamp_position[incompl_route_full.name=="Entrance"].tolist()[0]
                                            ##Creates a dataframe filtered with cluster positions(stoppage) in any port or anchorage inside suez canal
                                            direct_transit=group_of_complete[(group_of_complete.anchoring_cluster.notnull())&((group_of_complete.name=="Great Bitter Lake")|
                                                    (group_of_complete.name=="Port Taofik")|(group_of_complete.name=="Port Fuad")|(group_of_complete.name=="Port Said")|
                                                    (group_of_complete.name=="Said Container Terminal")|(group_of_complete.name=="Ismalia")|(group_of_complete.name=="Port Suez")|
                                                    (group_of_complete.name=="Port Aldabiya"))]
                                            if direct_transit.shape[0]==0 and ((calculated_time_exit-time_in)/np.timedelta64(1,"h"))<=24:
                                                direct_transit_boolean=True
                                                comment_on_stoppage=np.nan
                                            else:
                                                direct_transit_boolean=False
                                                stoppages=direct_transit.name.unique().tolist()
                                                ##List of clustered (stops) times inside the canal
                                                mask_stops=[]
                                                for stop in stoppages:
                                                    first_of_stop=(direct_transit.name == stop).idxmax()
                                                    last_of_stop=(direct_transit.name == stop)[::-1].idxmax()
                                                    description="{}-{}-{}".format(stop,direct_transit.timestamp_position[first_of_stop],direct_transit.timestamp_position[last_of_stop])
                                                    mask_stops.append(description)     
                                                comment_on_stoppage="/".join(mask_stops)
                                         
                                            ##From the filtered dataframe absorb the first and last position time of anchoring as declared by anchoring cluster
                                            anchoring_in=group_of_complete.timestamp_position[((group_of_complete.anchoring_cluster.notnull())&(group_of_complete.name=="South Anchorage")).idxmax()]
                                            anchoring_out=incompl_route_full.timestamp_position.iloc[0]
                                            
                                            if nan==0 or nan==2:       
                                                df_values.append([ship,ship_mmsi,"Northbound",access_anch,anchoring_in,anchoring_out,draught_in,draught_out,access_name,time_in,
                                                                  calculated_time_exit,direct_transit_boolean,comment_on_stoppage])###The case of IMO number
                                            elif nan==1:
                                                df_values.append([np.nan,ship,"Northbound",access_anch,anchoring_in,anchoring_out,draught_in,draught_out,access_name,time_in,
                                                                  calculated_time_exit,direct_transit_boolean,comment_on_stoppage])###The case of IMO number
                                    else:
                                        
                                        first_in_canal_base=group_of_complete["index"][((group_of_complete.name=="North Access East")|(group_of_complete.name=="North Access West")|(group_of_complete.name=="Said Container Terminal Access"))&(group_of_complete.lat>31.12)]
                                        first_in_canal_base=first_in_canal_base[first_in_canal_base>last_at_anch]
                                        if first_in_canal_base.shape[0]!=0:
                                            first_in_canal=first_in_canal_base.iloc[0]
                                            first_pos_in_canal=port_stops.geometry.loc[first_in_canal]
                                            accessed_canal_name=port_stops.name.loc[first_in_canal]
                                            access_anch=port_stops.name.loc[last_at_anch]
                                            
                                            ##Info of route that could serve to estimate speed                                
                                            incompl_route_full=port_stops.loc[last_at_anch:first_in_canal,:]
                                            ## Generate time difference between positions
                                            incompl_route_full=incompl_route_full.assign(diff=incompl_route_full.timestamp_position-incompl_route_full.timestamp_position.shift(1))
                                            incompl_route=LineString(incompl_route_full.geometry.to_list())
                                            incompl_route=[np.array(incompl_route.coords)]
                                            #Draught in
                                            draught_in=incompl_route_full.draught.mode().iloc[0]   
                                            
                                            if accessed_canal_name == "North Access East":
                                                #Entry point North Access East
                                                lon=32.3725
                                                lat=31.3214
                                                e_point_access=Point(lon,lat)
                                                ##Absorbs the canal route
                                                canal_linestring=list(routes.geometry.iloc[2].coords)
                                                ##Absorbs access canals and transform it to geometry
                                                access_linestring_df=access_routes[access_routes.access_number==6]
                                                access_name="North Access East"
                                            else:
                                                ##Entry point North Access West
                                                lon=32.3653
                                                lat=31.3214
                                                e_point_access=Point(lon,lat)
                                                ##Absorbs the canal route
                                                canal_linestring=list(routes.geometry.iloc[0].coords)
                                                ##Absorbs access canals and transform it to geometry
                                                access_linestring_df=access_routes[access_routes.access_number==7]                        
                                                access_name="North Access West"
                                            ##Best scenario. Complete route from anchorage to after checkpoint and route merging regular route
                                            if any(incompl_route_full["diff"]>timedelta(minutes=20))== False:
                                                access_linestring=access_linestring_df.geometry.tolist()
                                                ##Creates a list of array to calculate distances
                                                access_linestring=[ np.array((geom.xy[0], geom.xy[1])).transpose() for geom in access_linestring ]
                                                ##My incomplete route is assesed distance with all the exisiting access and returns the closest access
                                                closest_line = tdist.cdist(incompl_route,access_linestring, metric="sspd").transpose().argmin()
                                                unique_line_access=list(access_linestring_df.geometry.iloc[closest_line].coords)
                                                ##Joins the canal route and the access route
                                                merged_lines=LineString(unique_line_access+canal_linestring)
                                                closest_point_in_line=merged_lines.interpolate(merged_lines.project(last_pos_in_anch))
                                                intersected_line=split(merged_lines,closest_point_in_line.buffer(0.000000001))
                                                ##Cuts the line from the new starting point to the whole access route
                                                if len(intersected_line)<=2:
                                                    merged_lines=LineString(list(intersected_line[-2].coords)+list(intersected_line[-1].coords))
                                                else:
                                                    merged_lines=LineString(list(intersected_line[1].coords)+list(intersected_line[2].coords))
                                                distance_old_new=closest_point_in_line.distance(last_pos_in_anch)
                                                ##Ready the linestring cutted and the distance from old and new points. Need to identify the position inside the canal
                                                entrance_canal_distance=merged_lines.project(e_point_access)*60
                                                dist_first_pos_in_canal=merged_lines.project(first_pos_in_canal)*60                                              
                                                ##Calculate distance of every point
                                                incompl_route_full=incompl_route_full.assign(miles_travelled=incompl_route_full.geometry.apply(lambda x:merged_lines.project(x)*60))
                                                columns=incompl_route_full.columns.tolist()
                                                incompl_route_full=incompl_route_full.assign(miles_travelled=incompl_route_full.miles_travelled+distance_old_new)  
                                                entrance_info=pd.DataFrame([incompl_route_full.imo.iloc[-1],incompl_route_full.mmsi.iloc[-1],
                                                                            np.nan,lon,lat,np.nan,incompl_route_full.draught.max(),e_point_access,np.nan,
                                                                            "Entrance",np.nan,np.nan,merged_lines.project(e_point_access)*60]).T
                                                
                                                entrance_info.columns=incompl_route_full.columns
                                                incompl_route_full=incompl_route_full.assign(speed=(incompl_route_full.miles_travelled-incompl_route_full.miles_travelled.shift(1))/((incompl_route_full.timestamp_position-incompl_route_full.timestamp_position.shift(1))/np.timedelta64(1,'h')))
                                                incompl_route_full=pd.concat([incompl_route_full,entrance_info]).sort_values(by="miles_travelled").reset_index(drop=True)
                                                entrance_index=incompl_route_full.index[incompl_route_full.name=="Entrance"].tolist()[0]
                                                distances_x=[incompl_route_full.loc[entrance_index-1:entrance_index+1,"miles_travelled"]]
                                                if len(distances_x[0])==3:
                                                ##Interpolate. Full sequence
                                                    distances_x=[incompl_route_full.loc[entrance_index-1,"miles_travelled"],incompl_route_full.loc[entrance_index+1,"miles_travelled"]]
                                                    time_x=[0,(incompl_route_full.loc[entrance_index+1,"timestamp_position"]-incompl_route_full.loc[entrance_index-1,"timestamp_position"])/np.timedelta64(1,"h")]
                                                    interp_time=np.interp(incompl_route_full.loc[entrance_index,"miles_travelled"],distances_x,time_x)
                                                    incompl_route_full.loc[entrance_index,"timestamp_position"]=incompl_route_full.loc[entrance_index-1,"timestamp_position"]+timedelta(hours=interp_time)
                                                elif len(distances_x[0])==2:  ## Having two means that the last value is entrance. 
                                                #Since we requested to have first positions higher than check point, 
                                                #it means that the entrance positions is equal to the checkpoint.
                                                    incompl_route_full.loc[entrance_index-1,"name"]="Entrance"
                                                    incompl_route_full.drop(incompl_route_full.index[entrance_index],inplace=True)
                                                else:
                                                    break                                                 
                                                #Interpolate exit at the other side
                                                ##Gets first position out of canal after checkpoint. Calucalte average distance and interpolate position time
                                                first_position_out_filter=(group_of_complete.lat<=29.9318).idxmax()
                                                bef_exit_filter=(group_of_complete.lat>29.9318)[::-1].idxmax()                                 
                                                average_exit_speed=(group_of_complete.speed[first_position_out_filter]+group_of_complete.speed[bef_exit_filter])/2        
                                                
                                                exit_point=Point(32.5624,29.9318)
                                                exit_linestring=LineString(list(routes.geometry.iloc[2].coords)[::-1])
                                                dist_thres=(exit_linestring.project(exit_point)-exit_linestring.project(group_of_complete.geometry.loc[bef_exit_filter]))*60
                                                if average_exit_speed==0:
                                                    break
                                                else:
                                                    time_thres=timedelta(seconds=(dist_thres/average_exit_speed)*3600)
                                                ##Construction of missing values for df
                                                calculated_time_exit=group_of_complete.timestamp_position.loc[bef_exit_filter]+time_thres
                                                draught_out=group_of_complete.draught[first_position_out_filter]
                                                ship_mmsi=group_of_complete.mmsi[first_position_out_filter]
                                                time_in=incompl_route_full.timestamp_position[incompl_route_full.name=="Entrance"].tolist()[0]
                                                ##Creates a dataframe filtered with cluster positions(stoppage) in any port or anchorage inside suez canal
                                                direct_transit=group_of_complete[(group_of_complete.anchoring_cluster.notnull())&((group_of_complete.name=="Great Bitter Lake")|
                                                        (group_of_complete.name=="Port Taofik")|(group_of_complete.name=="Port Fuad")|(group_of_complete.name=="Port Said")|
                                                        (group_of_complete.name=="Said Container Terminal")|(group_of_complete.name=="Ismalia")|(group_of_complete.name=="Port Suez")|
                                                        (group_of_complete.name=="Port Aldabiya"))]
                                                if direct_transit.shape[0]==0 and ((calculated_time_exit-time_in)/np.timedelta64(1,"h"))<=24:
                                                    direct_transit_boolean=True
                                                    comment_on_stoppage=np.nan
                                                else:
                                                    direct_transit_boolean=False
                                                    stoppages=direct_transit.name.unique().tolist()
                                                    ##List of clustered (stops) times inside the canal
                                                    mask_stops=[]
                                                    for stop in stoppages:
                                                        first_of_stop=(direct_transit.name == stop).idxmax()
                                                        last_of_stop=(direct_transit.name == stop)[::-1].idxmax()
                                                        description="{}-{}-{}".format(stop,direct_transit.timestamp_position[first_of_stop],direct_transit.timestamp_position[last_of_stop])
                                                        mask_stops.append(description)     
                                                    comment_on_stoppage="/".join(mask_stops)                                        
                                                ##From the filtered dataframe absorb the first and last position time of anchoring as declared by anchoring cluster
                                                anchoring_in=group_of_complete.timestamp_position[((group_of_complete.anchoring_cluster.notnull())&(group_of_complete.name=="North Anchorage")).idxmax()]
                                                anchoring_out=incompl_route_full.timestamp_position.iloc[0]
                                                
                                                if nan==0 or nan==2:       
                                                    df_values.append([ship,ship_mmsi,"Southbound",access_anch,anchoring_in,anchoring_out,draught_in,draught_out,access_name,time_in,
                                                                      calculated_time_exit,direct_transit_boolean,comment_on_stoppage])###The case of IMO number
                                                elif nan==1:
                                                    df_values.append([np.nan,ship,"Southbound",access_anch,anchoring_in,anchoring_out,draught_in,draught_out,access_name,time_in,
                                                                      calculated_time_exit,direct_transit_boolean,comment_on_stoppage])###The case of IMO number
                                else:    
                                    fail_df.append([ship,2,group_of_complete.timestamp_position.iloc[0]])                                    
                            else:
                                fail_df.append([ship,1,group_of_complete.timestamp_position.iloc[0]])
                        else:
                            fail_df.append([ship,3,group_of_complete.timestamp_position.iloc[0]])
                    
            else:
                fail_df.append([ship,4,np.nan])
    df_export=pd.DataFrame(df_values)
    df_fail=pd.DataFrame(fail_df)
    return df_export,df_fail

####IMO numbers
#df_export_to,df_fail_to=iteration_clustered(imo)
#
#df_export_to=comm.gather(df_export_to,root=0)
#df_fail_to=comm.gather(df_fail_to,root=0)
#if rank==0:
#    pd.concat(df_export_to).to_csv("suez_canal_transits.csv",header=False,index=False,mode="a")
#    pd.concat(df_fail_to).to_csv("suez_canal_transits_failures.csv",header=False,index=False,mode="a")
#    
###MMSI numbers
#df_export_to,df_fail=iteration_clustered(mmsi)
#
#df_export_to=comm.gather(df_export_to,root=0)
#df_fail_to=comm.gather(df_fail_to,root=0)
#
#if rank==0:
#    pd.concat(df_export_to).to_csv("suez_canal_transits.csv",header=False,index=False,mode="a")
#    pd.concat(df_fail_to).to_csv("suez_canal_transits_failures.csv",header=False,index=False,mode="a")

##Terrestrial
df_export_to,df_fail_to=iteration_clustered(terrestrial)

df_export_to=comm.gather(df_export_to,root=0)
df_fail_to=comm.gather(df_fail_to,root=0)

if rank==0:
    pd.concat(df_export_to).to_csv("suez_canal_transits2.csv",header=False,index=False,mode="a")
    pd.concat(df_fail_to).to_csv("suez_canal_transits_failures2.csv",header=False,index=False,mode="a")
    
                       