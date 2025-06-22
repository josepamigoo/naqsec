from json import load as parse_json
from mqt.core import load
from mqt.qmap.na.zoned import RoutingAwareCompiler, ZonedNeutralAtomArchitecture

# IMPORTANT NOTE !! 
# This was the version used while the router was still in development. For any erroor, please referto the orignal repository QMAP.
# It could be that the router is not yet in the pip distribution of qmpa, in that case the repository must be cloned and built and this
# file and all the ones thar are in the same folder need to be located in the tools parent folder.

#--------------------------------------------------------------------

# load the architecture
arch = ZonedNeutralAtomArchitecture.from_json_file("architecture.json")

# load the settings
with open("settings.json", "r") as file:
    settings = file.read()

# create compiler
compiler = RoutingAwareCompiler.from_json_string(arch, settings)

# compile a QASM circuit
qc = load("naqsec/circuit_data/shor_RE_0_full_ft.qasm") #desired name
result = compiler.compile(qc)
naviz_name = "naqsec/circuit_data/shor_RE_0_full_ft.naviz"
# write result to file
with open(naviz_name, "w") as file:
    file.write(result)

# (optionally) replace all occurrences of u gates with the respective gate, e.g., h and x
with open(naviz_name, "r") as file:
     content = file.read()
content = content.replace("u 1.57080 0.00000 3.14159", "h")
content = content.replace("u 3.14159 0.00000 3.14159", "x")

# # overwrite the file with the modified content
with open(naviz_name, "w") as file:
     file.write(content)