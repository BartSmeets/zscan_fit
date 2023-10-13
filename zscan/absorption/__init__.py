"""Package containing modules to describe the absorption

## Contains
---
- dI_dz: define the differential equation describing the absorption within a sample
- intensity: return the beam intensity
- transmittance: return the transmittance through a sample
"""

import zscan.absorption.dI_dz
from zscan.absorption.intensity import intensity
import zscan.absorption.transmittance