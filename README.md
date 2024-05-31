run 
   ```bash
   pip install build
   python3 -m build
   ```
 to "compile" library; this is not published to pypl yet

in the mean time you need this repo tangent to the repo your working in i.e.

```
├── BPSK
│   └── bpsk.py
└── KUSignalLib
    ├── pyproject.toml
    ├── setup.cfg
    ├── src
    └── ...
```

then in your python script, put this at the top of your file

   ```python
   import sys
   sys.path.insert(0, '../KUSignalLib/src')
   ```

then you can use this library like you normaly would, i.e. 

```python
from KUSignalLib import examples as ex
```

