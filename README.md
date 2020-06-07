# Suez Canal Transits algorithm
Generates Suez Canal Transit information from raw AIS data from January 2013 to June, 2019.
[Pseudocode](https://github.com/gabrielfuenmar/suez-canal-transits/blob/master/pseudo_suez.pdf) available at the repository.

Distributed computing setting under Sun Grid manager deployed in a Round Robin configuration see [suez_mpi.sh](https://github.com/gabrielfuenmar/suez-canal-transits/blob/master/suez_mpi.sh) for Sun Grid setting.

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

Parameter:
    
    suez_poly: geopandas [dataframe](https://github.com/gabrielfuenmar/suez-canal-transits/blob/master/1_suez_canal_polygons.png) with Suez Canal polygons
    vessels_all: pandas dataframe of vessel specs. Not at the respository due to    supplier restrictions.
    routes: geopandas dataframe with Suez routes through the Canal.
    access_routes: geopandas [dataframe](https://github.com/gabrielfuenmar/suez-canal-transits/blob/master/3_canal_access.png) with Suez  and Said access routes.
    
    
Returns:
  
    CSV file with transits information.

The information is build per vessel and is the result of a sequence of test that validates that a transit has enough information to be throughly analized. Refer to ##PAPER when published### section 3 for a description.


Credits: Gabriel Fuentes Lezcano
Licence: MIT License

Copyright (c) 2020 Gabriel Fuentes

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. © 2020 GitHub, Inc.
