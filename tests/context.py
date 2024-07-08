### context.py ###
# author: Elliott Walker
# last update: 8 July 2024
# description: adds full codebase to search context; see info in docs/tests.txt
# usage: `import context` (in header of a test script)
#   then, able to import code as e.g. `import windprofiles.atmo_calc`

import os
import sys
sys.path.insert(0, context_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
