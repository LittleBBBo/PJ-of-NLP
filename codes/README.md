# Usage

```shell
# fetch data
cd codes/res/20news
python fetch_data.py
cd codes/res/wiki20020
python fetch_data.py

# build cython
cd codes/src/mylda/mylda/sample
sh build.sh
cd codes/src/dataground
sh build.sh

# run in ipython
cd codes/src
%run -i experimentArea.py
%run -i dataground/repr_models.py
compare_experiment(dg)
```

