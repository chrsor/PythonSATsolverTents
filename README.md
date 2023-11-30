# How to run CDCL solver

``python cdcdl.py <path to sat instance>``

## Optional Parameters 
```
'-hc', '--heuristic', default='VSIDS': "VSIDS" oder "VMTF"
'-r', '--restart_after', default=100: False to turn off restarts
'-bd', '--benchmark_dir', default=None: Activates benchmark mode, will try to solve all sat instances in benchmark_dir in under a minute
'-pre', '--preprocessing', default=False: Turn on Self-Subsuming Resolution (True) (slow)
'-prf', '--proof', default=False: Show proof (True/False)
```
