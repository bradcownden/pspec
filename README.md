# pspec
 Julia code and explanations for calculating the pseudospectrum of quasinormal modes
 
 REQUIRED: Julia programming language (https://julialang.org/downloads/)
           NVIDIA GPU (Optional functionality in later versions)
           
To run the code, download Julia and open the default REPL. Use the "]" key to open the package management system, and add the packages that are imported in the source code. For example, pspec.jl requires the ThreadsX package as identified by the "using ThreadsX" call in the header. In the package management system, type "add ThreadsX" in the command prompt and Julia will source the package and install any required sub-packages. Repeat the procedure of adding third-party packages through the Jula REPL for all packages present in any \*.jl files (except anything starting with "Base"). 
 
 Once all packages have been installed, run pspec.jl from the command line using the syntax: julia -t 4 ./pspec.jl 10. This executes pspec with 4 tasks for parallelism and with 10 bits of arithmetic precision. To run in serial, omit the -t flag. If bits of precision is not specified, or if fewer than 32 bits are specified, pspec will default to double precision. If more than 32 bits are specified, pspec will run using the requested precision whenever possible. Finally, pspec reads the input file "Inputs.txt" for: spectral_N (degree of Chebyshev polynomial), xgrid_min (pseudospectrum x minmum), xgrid_max (pseudospectrum x maximum), ygrid_min (pseudospectrum y minimum), ygrid_max (pseudospectrum y maximum), p_gridx (pseudospectrum horizonal point density), and p_gridy (pseudospectrum vertical point density).
