# Suez Canal Transits algorithm
Generates Suez Canal Transit information from raw AIS data from January 2013 to June, 2019.
[Pseudocode](https://github.com/gabrielfuenmar/suez-canal-transits/blob/master/pseudo_suez.pdf) available at the repository.

Distributed computing setting under Sun Grid manager deployed in a Round Robin configuration. See [suez_mpi.sh](https://github.com/gabrielfuenmar/suez-canal-transits/blob/master/suez_mpi.sh). for Sungrid setting.

MPI setting recognizes any distributed enviroment and the resoruces allocated to the app.

AIS information used as input and Vessel Specifications are not displayed here as restricted by the suppliers.

Dependencies:

    scikit-learn 0.21.3
    numpy 1.17.2
    geopandas 0.6.1
    shapely 1.6.4
    pandas  0.25.1
    trajectory_distance 1.0
    mpi4py 3.0.2

Parameters:
    
    suez_poly: geopandas dataframe with Suez Canal polygons
    vessels_all: pandas dataframe of vessel specs. Not at the respository due to supplier restrictions.
    routes: geopandas dataframe with Suez routes through the Canal.
    access_routes: geopandas dataframe with Suez  and Said access routes.

Suez poly figure

![alt text](https://github.com/gabrielfuenmar/suez-canal-transits/blob/master/1_suez_canal_polygons.png)

Access route figure

![alt text](https://github.com/gabrielfuenmar/suez-canal-transits/blob/master/3_canal_access.png)

Returns:
  
    CSV file with transits information.

The information is build per vessel and is the result of a sequence of tests that validates that a transit has enough information to be throughly analized. Refer to [published paper](https://ieeexplore.ieee.org/document/9309882) for a description.


Credits: Gabriel Fuentes Lezcano
