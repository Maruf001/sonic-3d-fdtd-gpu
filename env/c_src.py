# Let future `!nvc++` calls work without re-exporting in every bash cell
import os
os.environ["PATH"] = "/content/nvhpc/Linux_x86_64/25.7/compilers/bin:" + os.environ["PATH"]
