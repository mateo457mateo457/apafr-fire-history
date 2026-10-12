# APAFR fire analysis

Analyzing new data provided in 9-2026 to derive number of fires and area burned per ignition type.

As of 9-15-2026, this is an experimental/pilot study to understand the quality of the data.  

As of 10-6-2026, in v2, we started to work towards making a script that processes the fire data in a way that can
be ported over to the MWB project.

In v3 we compared the new data set with the work we did on various published papers in 2010 and 2015.

v1 (10-2-2026): we are attempting to assign fires in the older data with fire IDs.
v2 (10-6-2026): another round where we attempt to apply what we learned in v1 to make a master fire data set.
v3 (10-12-2026): compare the new data set with the old one to verify that our results will agree with previous work.
    The results found a 3% differences in area burned between the two data sets, likely the result of the new data
    set having a procedure to identify unknown fires, along with small differences of unknown origin.

## Scripts and run order

Run scripts from the project root (`FIRE/`) in this order:

1. `process_fire_data.py` – Builds the master fire history from the raw
   shapefiles: classifies ignition source, groups polygons into fire events,
   resolves Unknown causes (2006-2009 yearly files, then a logistic model), and
   writes `data/processed/fire/fire_data.gpkg`. Also writes diagnostic maps
   (`maps/`) and check graphs (`data_checks/`).
2. `compare_new_data_with_old.py` – Compares `fire_data.gpkg` with the 2010
   compilation used in the published papers (`old_fire_data.csv`) over
   FY1997-FY2009. Prints totals and per-year differences and writes the
   `compare_*` figures to `data_checks/`.

Step 2 depends on the output of step 1.

## How to run

Requires Python 3.11 and the following packages:
geopandas, pandas, numpy, networkx, scikit-learn, matplotlib.

Setup (first time, from the project root):

    python -m venv .venv
    .venv\Scripts\activate
    pip install geopandas pandas numpy networkx scikit-learn matplotlib

Run (from the project root, with the venv active):

    python process_fire_data.py
    python compare_new_data_with_old.py

The scripts find data relative to the current working directory
(`Path.cwd() / "data"`), so they must be run from `FIRE/`. In PyCharm,
set the run configuration's working directory to the project root.

## Data layout

    data/
    ├── raw/
    │   └── fire/
    │       ├── master files/
    │       │   ├── 1977_2005/APAFR_Burn_History_1977_2005.shp   (1996-2005 used)
    │       │   └── 2006_2026/APAFR_Burn_Histroy_2006_2026.shp   (2006-2025 used)
    │       ├── yearly files/
    │       │   └── 2006/ ... 2009/   one shapefile per year (old-style schema,
    │       │                         used to recover 2006-2009 ignition sources)
    │       ├── boundary/boundary.shp  range boundary + named target areas
    │       └── 2010_fire_data/
    │           └── old_fire_data.csv  2010 compilation used in published papers
    │                                  (compiled from prescribedfires.sas7bdat and
    │                                  wildfires.sas7bdat)
    └── processed/
        └── fire/
            ├── fire_data.gpkg         master fire history, one row per fire event
            ├── maps/
            │   ├── known/             monthly maps, known fires by source
            │   └── unknown/           monthly maps, Unknown fires by predicted source
            └── data_checks/           check graphs (process_fire_data.py)
                                       and compare_* figures (compare_new_data_with_old.py)

Raw data provided by APAFR personnel in September 2026. Raw files are not modified;
all outputs are written to `data/processed/`.

## Known limitations

- **Prescribed fire events are approximate.** Each prescribed polygon is
  treated as one event. A real prescribed burn is a planned operation that can
  span non-adjacent units, but the 1996-2005 data has no Fire Zone/operation
  field to reconstruct this. Fire counts for Prescribed should be interpreted
  with this in mind; area burned is unaffected.
- **2010-2011 Unknowns rely on the model alone.** The 2006-2009 Unknowns were
  resolved cleanly from older shapefiles, but no yearly files were found for 
  2010-2011.
- **The Unknown-resolution model is somewhat weak for Lightning.** Military calls 
  are reliable; cross-validated precision of a Lightning call is ~0.75-0.85
  regardless of threshold. However, unknown fires total ~794 ha, so potential
  misclassification @20% is a very small share of total area burned.
- **Placeholder dates.** Records with an invalid day (e.g. 99) are set to the
  15th of the month; records with an unknown month are dropped (1 record).
- **Excluded causes.** Arson, off-post, previous-fire, catalytic converter,
  and heavy equipment fires are excluded rather than forced into a category.
- **Wildfire event grouping depends on two tolerances**: 10 m gap and +/-2
  days. Event counts (not area) are sensitive to these choices.