var documenterSearchIndex = {"docs":
[{"location":"man/contributing/#Contributing","page":"Contributing","title":"Contributing","text":"","category":"section"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"This page serves as the contribution guide for the DCCR package. From top to bottom, the ways of contributing are:","category":"page"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"GitHub Issues: how to raise an issue with the project.\nJulia Development: how to download and interact with the package.\nGitFlow: how to directly contribute code to the package in an organized way on GitHub.\nDevelopment Details: how the internals of the package are currently setup if you would like to directly contribute code.","category":"page"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"Please also see the Attribution to learn about the authors and sources of support for the project.","category":"page"},{"location":"man/contributing/#Issues","page":"Contributing","title":"Issues","text":"","category":"section"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"The main point of contact is the GitHub issues page for the project. This is the easiest way to contribute to the project, as any issue you find or request you have will be addressed there by the authors of the package. Depending on the issue, the authors will collaborate with you, and after making changes they will link a pull request which addresses your concern or implements your proposed changes.","category":"page"},{"location":"man/contributing/#Julia-Development","page":"Contributing","title":"Julia Development","text":"","category":"section"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"As a Julia package, development follows the usual procedure:","category":"page"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"Clone the project from GitHub\nSwitch to or create the branch that you wish work on (see GitFlow).\nStart Julia at your development folder.\nInstantiate the package (i.e., download and install the package dependencies).","category":"page"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"For example, you can get the package and startup Julia with","category":"page"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"git clone git@github.com:AP6YC/DCCR.jl.git\njulia --project=.","category":"page"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"note: Note\nIn Julia, you must activate your project in the current REPL to point to the location/scope of installed packages. The above immediately activates the project when starting up Julia, but you may also separately startup the julia and activate the package with the interactive package manager via the ] syntax:julia\njulia> ]\n(@v1.8) pkg> activate .\n(DCCR) pkg>","category":"page"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"You may run the package's unit tests after the above setup in Julia with","category":"page"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"julia> using Pkg\njulia> Pkg.instantiate()\njulia> Pkg.test()","category":"page"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"or interactively though the Julia package manager with","category":"page"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"julia> ]\n(DCCR) pkg> instantiate\n(DCCR) pkg> test","category":"page"},{"location":"man/contributing/#GitFlow","page":"Contributing","title":"GitFlow","text":"","category":"section"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"The DCCR package follows the GitFlow git working model. The original post by Vincent Driessen outlines this methodology quite well, while Atlassian has a good tutorial as well. In summary:","category":"page"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"Create a feature branch off of the develop branch with the name feature/<my-feature-name>.\nCommit your changes and push to this feature branch.\nWhen you are satisfied with your changes, initiate a GitHub pull request (PR) to merge the feature branch with develop.\nIf the unit tests pass, the feature branch will first be merged with develop and then be deleted.\nReleases will be periodically initiated from the develop branch and versioned onto the master branch.\nImmediate bug fixes circumvent this process through a hotfix branch off of master.","category":"page"},{"location":"man/contributing/#Development-Details","page":"Contributing","title":"Development Details","text":"","category":"section"},{"location":"man/contributing/#Documentation","page":"Contributing","title":"Documentation","text":"","category":"section"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"These docs are currently hosted as a static site on the GitHub pages platform. They are setup to be built and served in a separate branch called gh-pages from the master/development branches of the project.","category":"page"},{"location":"man/contributing/#Package-Structure","page":"Contributing","title":"Package Structure","text":"","category":"section"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"The DCCR package has the following file structure:","category":"page"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"DCCR\n├── .github/workflows       // GitHub: workflows for testing and documentation.\n├── cluster                 // CI: cluster submission files and scripts.\n├── data                    // Data: original datsets.\n│   ├───packed              //      Compressed tarballs of datasets.\n│   └───unpacked            //      Decompressed datasets.\n├── dockerfiles             // CI: Docker image definitions.\n├── docs                    // Docs: documentation for the module.\n│   └───src                 //      Documentation source files.\n├── src                     // Source: majority of source code.\n├── test                    // Test: Unit, integration, and environment tests.\n├── work                    // Data: options and generated results.\n│   ├───configs             //      Experiment configuration files.\n│   ├───models              //      Saved model weights.\n│   └───results             //      Destination for generated figures, etc.\n├── .gitattributes          // Git: LFS settings, languages, etc.\n├── .gitignore              // Git: .gitignore for the whole project.\n├── CODE_OF_CONDUCT.md      // Doc: the code of conduct for contributors.\n├── CONTRIBUTING.md         // Doc: contributing guide (points to this page).\n├── LICENSE                 // Doc: the license to the project.\n├── Project.toml            // Julia: the Pkg.jl dependencies of the project.\n└── README.md               // Doc: the top-level readme for the project.","category":"page"},{"location":"man/contributing/#Type-Aliases","page":"Contributing","title":"Type Aliases","text":"","category":"section"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"For convenience in when defining types and function signatures, this package uses the NumericalTypeAliases.jl package and the aliases therein. The documentation for the abstract and concrete types provided by NumericalTypeAliases.jl can be found here.","category":"page"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"In this package, data samples are always Real-valued, whereas class labels are integered. Furthermore, independent class labels are always Int because of the Julia native support for a given system's signed native integer type.","category":"page"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"This project does not currently test for the support of arbitrary precision arithmetic because learning algorithms in general do not have a significant need for precision.","category":"page"},{"location":"man/contributing/#Attribution","page":"Contributing","title":"Attribution","text":"","category":"section"},{"location":"man/contributing/#Authors","page":"Contributing","title":"Authors","text":"","category":"section"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"This package is developed and maintained by Sasha Petrenko with sponsorship by the Applied Computational Intelligence Laboratory (ACIL).","category":"page"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"If you simply have suggestions for improvement, Sasha Petrenko (<sap625@mst.edu>) is the current developer and maintainer of the DCCR package, so please feel free to reach out with thoughts and questions.","category":"page"},{"location":"man/dev-index/#dev-main-index","page":"Dev Index","title":"Developer Index","text":"","category":"section"},{"location":"man/dev-index/","page":"Dev Index","title":"Dev Index","text":"This page lists the types and functions that are internal to the DCCR package. Because they are not part of the public API, these names might change relatively frequently between versions and so should not be relied upon.","category":"page"},{"location":"man/dev-index/","page":"Dev Index","title":"Dev Index","text":"All internal names are listed in the Index, and each of these entries link to the docstrings in the Docs section.","category":"page"},{"location":"man/dev-index/#Index","page":"Dev Index","title":"Index","text":"","category":"section"},{"location":"man/dev-index/","page":"Dev Index","title":"Dev Index","text":"This section contains a list of internal names that link to their corresponding Documentation.","category":"page"},{"location":"man/dev-index/#dev-index-methods","page":"Dev Index","title":"Methods","text":"","category":"section"},{"location":"man/dev-index/","page":"Dev Index","title":"Dev Index","text":"Pages   = [\"dev-index.md\"]\nModules = [DCCR]\nOrder = [:function]","category":"page"},{"location":"man/dev-index/#dev-index-types","page":"Dev Index","title":"Types","text":"","category":"section"},{"location":"man/dev-index/","page":"Dev Index","title":"Dev Index","text":"Pages   = [\"dev-index.md\"]\nModules = [DCCR]\nOrder = [:type]","category":"page"},{"location":"man/dev-index/#dev-index-types-2","page":"Dev Index","title":"Constants","text":"","category":"section"},{"location":"man/dev-index/","page":"Dev Index","title":"Dev Index","text":"Pages   = [\"dev-index.md\"]\nModules = [DCCR]\nOrder = [:constant]","category":"page"},{"location":"man/dev-index/#dev-index-docs","page":"Dev Index","title":"Docs","text":"","category":"section"},{"location":"man/dev-index/","page":"Dev Index","title":"Dev Index","text":"Documentation for all internal names are listed below.","category":"page"},{"location":"man/dev-index/","page":"Dev Index","title":"Dev Index","text":"Modules = [DCCR]\nPublic = false","category":"page"},{"location":"man/dev-index/#DCCR.BLOCK_TYPES","page":"Dev Index","title":"DCCR.BLOCK_TYPES","text":"The names of the blocks that are encountered during L2 experiments.\n\n\n\n\n\n","category":"constant"},{"location":"man/dev-index/#DCCR.JSON_INDENT","page":"Dev Index","title":"DCCR.JSON_INDENT","text":"Constant for pretty indentation spacing in JSON files.\n\n\n\n\n\n","category":"constant"},{"location":"man/dev-index/#DCCR.LOG_STATES","page":"Dev Index","title":"DCCR.LOG_STATES","text":"The enumerated states that an L2 logger log can be in.\n\n\n\n\n\n","category":"constant"},{"location":"man/dev-index/#DCCR.Agent","page":"Dev Index","title":"DCCR.Agent","text":"L2 agent supertype.\n\n\n\n\n\n","category":"type"},{"location":"man/dev-index/#DCCR.DDVFAAgent","page":"Dev Index","title":"DCCR.DDVFAAgent","text":"DDVFA-based L2 agent.\n\n\n\n\n\n","category":"type"},{"location":"man/dev-index/#DCCR.DDVFAAgent-Tuple{AdaptiveResonance.opts_DDVFA, AbstractDict}","page":"Dev Index","title":"DCCR.DDVFAAgent","text":"Constructor for a DDVFAAgent using the scenario dictionary and optional DDVFA keyword argument options.\n\nArguments\n\nopts::AbstractDict: keyword arguments for DDVFA options.\nscenario::AbstractDict: l2logger scenario as a dictionary.\n\n\n\n\n\n","category":"method"},{"location":"man/dev-index/#DCCR.DDVFAAgent-Tuple{AdaptiveResonance.opts_DDVFA}","page":"Dev Index","title":"DCCR.DDVFAAgent","text":"Creates a DDVFA agent with an empty experience queue.\n\nArguments\n\nddvfa_opts::opts_DDVFA: the options struct used to initialize the DDVFA module and set the logging params.\n\n\n\n\n\n","category":"method"},{"location":"man/dev-index/#DCCR.Data","page":"Dev Index","title":"DCCR.Data","text":"Abstract supertype for all Data structs in this library.\n\n\n\n\n\n","category":"type"},{"location":"man/dev-index/#DCCR.DataSplit","page":"Dev Index","title":"DCCR.DataSplit","text":"A basic struct for encapsulating the components of supervised training.\n\n\n\n\n\n","category":"type"},{"location":"man/dev-index/#DCCR.DataSplitCombined","page":"Dev Index","title":"DCCR.DataSplitCombined","text":"A struct for combining training and validation data, containing only train and test splits.\n\n\n\n\n\n","category":"type"},{"location":"man/dev-index/#DCCR.DataSplitCombined-Tuple{DCCR.DataSplit}","page":"Dev Index","title":"DCCR.DataSplitCombined","text":"Constructs a DataSplitCombined from an existing DataSplit by consolidating the training and validation data.\n\nArguments\n\ndata::DataSplit: the DataSplit struct for consolidating validation Features and Labels into the training data.\n\n\n\n\n\n","category":"method"},{"location":"man/dev-index/#DCCR.DataSplitIndexed","page":"Dev Index","title":"DCCR.DataSplitIndexed","text":"A basic struct for encapsulating the components of supervised training.\n\n\n\n\n\n","category":"type"},{"location":"man/dev-index/#DCCR.Experience","page":"Dev Index","title":"DCCR.Experience","text":"Experience block for an agent.\n\nTaken from l2logger_template.\n\n\n\n\n\n","category":"type"},{"location":"man/dev-index/#DCCR.Experience-Tuple{AbstractString, DCCR.SequenceNums, AbstractString}","page":"Dev Index","title":"DCCR.Experience","text":"Constructs an Experience, setting the update_model field based upon the block type.\n\nArguments\n\ntask_name::AbstractString: the name of the current task.\nseq_nums::SequenceNums: the block and experience number of the experience.\nblock_type::AbstractString: the block type ∈ [\"train\", \"test\"]. Using \"train\" sets update_model to true, \"test\" to false.\n\n\n\n\n\n","category":"method"},{"location":"man/dev-index/#DCCR.ExperienceQueue","page":"Dev Index","title":"DCCR.ExperienceQueue","text":"Alias for a queue of Experiences.\n\n\n\n\n\n","category":"type"},{"location":"man/dev-index/#DCCR.ExperienceQueueContainer","page":"Dev Index","title":"DCCR.ExperienceQueueContainer","text":"Container for the ExperienceQueue and some statistics about it.\n\n\n\n\n\n","category":"type"},{"location":"man/dev-index/#DCCR.ExperienceQueueContainer-Tuple{AbstractDict}","page":"Dev Index","title":"DCCR.ExperienceQueueContainer","text":"Creates a queue of Experiences from the scenario dictionary.\n\nArguments\n\nscenario_dict::AbstractDict: the scenario dictionary.\n\n\n\n\n\n","category":"method"},{"location":"man/dev-index/#DCCR.ExperienceQueueContainer-Tuple{}","page":"Dev Index","title":"DCCR.ExperienceQueueContainer","text":"Creates an empty ExperienceQueueContainer with an empty queue and zeroed stats.\n\n\n\n\n\n","category":"method"},{"location":"man/dev-index/#DCCR.Features","page":"Dev Index","title":"DCCR.Features","text":"Definition of features as a matrix of floating-point numbers of dimension (featuredim, nsamples).\n\n\n\n\n\n","category":"type"},{"location":"man/dev-index/#DCCR.LabeledDataset","page":"Dev Index","title":"DCCR.LabeledDataset","text":"A single dataset of Features, Targets, and human-readable string Labels.\n\n\n\n\n\n","category":"type"},{"location":"man/dev-index/#DCCR.LabeledDataset-Tuple{DCCR.LabeledDataset, DCCR.LabeledDataset}","page":"Dev Index","title":"DCCR.LabeledDataset","text":"A constructor for a LabeledDataset that merges two other LabeledDatasets.\n\nArguments\n\nd1::LabeledDataset: the first LabeledDataset to consolidate.\nd2::LabeledDataset: the second LabeledDataset to consolidate.\n\n\n\n\n\n","category":"method"},{"location":"man/dev-index/#DCCR.Labels","page":"Dev Index","title":"DCCR.Labels","text":"Definition of labels as a vector of strings.\n\n\n\n\n\n","category":"type"},{"location":"man/dev-index/#DCCR.MatrixData","page":"Dev Index","title":"DCCR.MatrixData","text":"Abstract type for Data structs that represent features as matrices.\n\n\n\n\n\n","category":"type"},{"location":"man/dev-index/#DCCR.SequenceNums","page":"Dev Index","title":"DCCR.SequenceNums","text":"Sequence numbers for a block and experience.\n\nTaken from l2logger_template.\n\n\n\n\n\n","category":"type"},{"location":"man/dev-index/#DCCR.StatsDict","page":"Dev Index","title":"DCCR.StatsDict","text":"Alias for a statistics dictionary being string keys mapping to any object.\n\n\n\n\n\n","category":"type"},{"location":"man/dev-index/#DCCR.Targets","page":"Dev Index","title":"DCCR.Targets","text":"Definition of targets as a vector of integers.\n\n\n\n\n\n","category":"type"},{"location":"man/dev-index/#DCCR.VectorLabeledDataset","page":"Dev Index","title":"DCCR.VectorLabeledDataset","text":"A single dataset of vectored labeled data with Features, Targets, and human-readable string Labels.\n\n\n\n\n\n","category":"type"},{"location":"man/dev-index/#DCCR.VectoredData","page":"Dev Index","title":"DCCR.VectoredData","text":"Abstract type for Data structs that represent features as vectors of matrices.\n\n\n\n\n\n","category":"type"},{"location":"man/dev-index/#Base.show-Tuple{IO, DCCR.DDVFAAgent}","page":"Dev Index","title":"Base.show","text":"Overload of the show function for DDVFAAgent.\n\nArguments\n\nio::IO: the current IO stream.\ncont::DDVFAAgent: the DDVFAAgent to print/display.\n\n\n\n\n\n","category":"method"},{"location":"man/dev-index/#Base.show-Tuple{IO, DCCR.ExperienceQueueContainer}","page":"Dev Index","title":"Base.show","text":"Overload of the show function for ExperienceQueueContainer.\n\nArguments\n\nio::IO: the current IO stream.\ncont::ExperienceQueueContainer: the ExperienceQueueContainer to print/display.\n\n\n\n\n\n","category":"method"},{"location":"man/dev-index/#Base.show-Tuple{IO, DataStructures.Deque{DCCR.Experience}}","page":"Dev Index","title":"Base.show","text":"Overload of the show function for ExperienceQueue.\n\nArguments\n\nio::IO: the current IO stream.\ncont::ExperienceQueueContainer: the ExperienceQueueContainer to print/display.\n\n\n\n\n\n","category":"method"},{"location":"man/dev-index/#DCCR.evaluate_agent!-Tuple{DCCR.Agent, DCCR.Experience, DCCR.VectoredData}","page":"Dev Index","title":"DCCR.evaluate_agent!","text":"Evaluates a single agent on a single experience, training or testing as needed.\n\nArguments\n\nagent::Agent: the Agent to evaluate.\nexp::Experience: the Experience to use for training/testing.\n\n\n\n\n\n","category":"method"},{"location":"man/dev-index/#DCCR.fields_to_dict!-Tuple{AbstractDict, Any}","page":"Dev Index","title":"DCCR.fields_to_dict!","text":"Adds entry to a dictionary from a struct with fields.\n\nMeant to be used with StatsDict.\n\nArguments\n\ndict::AbstractDict: the dictionary to add entries to.\nopts::Any: a struct containing fields, presumably of options, to add as key-value entries to the dict.\n\n\n\n\n\n","category":"method"},{"location":"man/dev-index/#DCCR.get_index_from_name-Union{Tuple{T}, Tuple{Vector{T}, AbstractString}} where T<:AbstractString","page":"Dev Index","title":"DCCR.get_index_from_name","text":"Gets an integer index of where a string name appears in a list of strings.\n\nArguments\n\nlabels::Vector{T} where T <: AbstractString: the list of strings to search.\nname::AbstractString: the name to search for in the list of labels.\n\n\n\n\n\n","category":"method"},{"location":"man/dev-index/#DCCR.initialize_exp_queue!-Tuple{DCCR.ExperienceQueueContainer, AbstractDict}","page":"Dev Index","title":"DCCR.initialize_exp_queue!","text":"Initializes an ExperienceQueueContainer from the provided scenario dictionary.\n\nArguments\n\neqc::ExperienceQueueContainer: the container with the queue and stats to initialize.\nscenario_dict::AbstractDict: the dictionary with the scenario regimes and block types.\n\n\n\n\n\n","category":"method"},{"location":"man/dev-index/#DCCR.is_complete-Tuple{DCCR.Agent}","page":"Dev Index","title":"DCCR.is_complete","text":"Checks if the Agent is done with its scenario queue.\n\nArguments\n\nagent::Agent: the agent to test scenario completion on.\n\n\n\n\n\n","category":"method"},{"location":"man/dev-index/#DCCR.json_load-Tuple{AbstractString}","page":"Dev Index","title":"DCCR.json_load","text":"Loads the JSON file, interpreted as a dictionary.\n\nArguments\n\nfilepath::AbstractString: the full file name (with path) to load.\n\n\n\n\n\n","category":"method"},{"location":"man/dev-index/#DCCR.json_save-Tuple{AbstractString, AbstractDict}","page":"Dev Index","title":"DCCR.json_save","text":"Saves the dictionary to a JSON file.\n\nArguments\n\nfilepath::AbstractString: the full file name (with path) to save to.\ndict::AbstractDict: the dictionary to save to the file.\n\n\n\n\n\n","category":"method"},{"location":"man/dev-index/#DCCR.log_data-Tuple{PyCall.PyObject, DCCR.Experience, Dict, Dict}","page":"Dev Index","title":"DCCR.log_data","text":"Logs data from an L2 Experience.\n\nArguments\n\ndata_logger::PyObject: the l2logger DataLogger.\nexp::Experience: the Experience that the Agent just processed.\nresults::Dict: the results from the Agent's Experience.\nstatus::AbstractString: string expressing if the experience was processed.\n\n\n\n\n\n","category":"method"},{"location":"man/dev-index/#DCCR.run_scenario-Tuple{DCCR.Agent, DCCR.VectoredData, PyCall.PyObject}","page":"Dev Index","title":"DCCR.run_scenario","text":"Runs an agent's scenario.\n\nArguments\n\nagent::Agent: a struct that contains an Agent and scenario.\ndata_logger::PyObject: a l2logger object.\n\n\n\n\n\n","category":"method"},{"location":"man/dev-index/#DCCR.sanitize_block_type-Tuple{AbstractString}","page":"Dev Index","title":"DCCR.sanitize_block_type","text":"Sanitize the selected block type against the BLOCK_TYPES constant.\n\nArguments\n\nblock_type::AbstractString: the selected block type.\n\n\n\n\n\n","category":"method"},{"location":"man/dev-index/#DCCR.sanitize_in_list-Union{Tuple{T}, Tuple{AbstractString, T, Vector{T}}} where T","page":"Dev Index","title":"DCCR.sanitize_in_list","text":"Sanitizes a selection within a list of acceptable options.\n\nArguments\n\nselection_type::AbstractString: a string describing the option in case it is misused.\nselection::Any: a single selection from a list.\n\n\n\n\n\n","category":"method"},{"location":"man/dev-index/#DCCR.sanitize_log_state-Tuple{AbstractString}","page":"Dev Index","title":"DCCR.sanitize_log_state","text":"Sanitize the selected log state against the LOG_STATES constant.\n\nArguments\n\nlog_state::AbstractString: the selected log state.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#main-index","page":"Index","title":"Index","text":"","category":"section"},{"location":"man/full-index/","page":"Index","title":"Index","text":"This page lists the core methods and types of the Julia component of the DCCR project.","category":"page"},{"location":"man/full-index/#Index","page":"Index","title":"Index","text":"","category":"section"},{"location":"man/full-index/","page":"Index","title":"Index","text":"This section enumerates the names exported by the package, each of which links to its corresponding Documentation.","category":"page"},{"location":"man/full-index/#index-modules","page":"Index","title":"Modules","text":"","category":"section"},{"location":"man/full-index/","page":"Index","title":"Index","text":"Pages   = [\"full-index.md\"]\nModules = [DCCR]\nOrder = [:module]","category":"page"},{"location":"man/full-index/#index-methods","page":"Index","title":"Methods","text":"","category":"section"},{"location":"man/full-index/","page":"Index","title":"Index","text":"Pages   = [\"full-index.md\"]\nModules = [DCCR]\nOrder = [:function]","category":"page"},{"location":"man/full-index/#index-types","page":"Index","title":"Types","text":"","category":"section"},{"location":"man/full-index/","page":"Index","title":"Index","text":"Pages   = [\"full-index.md\"]\nModules = [DCCR]\nOrder = [:type]","category":"page"},{"location":"man/full-index/#index-constants","page":"Index","title":"Constants","text":"","category":"section"},{"location":"man/full-index/","page":"Index","title":"Index","text":"Pages   = [\"full-index.md\"]\nModules = [DCCR]\nOrder = [:constant]","category":"page"},{"location":"man/full-index/#index-docs","page":"Index","title":"Docs","text":"","category":"section"},{"location":"man/full-index/","page":"Index","title":"Index","text":"This section lists the documentation for every exported name of the DCCR package.","category":"page"},{"location":"man/full-index/#index-modules-docs","page":"Index","title":"Modules","text":"","category":"section"},{"location":"man/full-index/","page":"Index","title":"Index","text":"Modules = [DCCR]\nPrivate = false\nOrder = [:module]","category":"page"},{"location":"man/full-index/#DCCR.DCCR","page":"Index","title":"DCCR.DCCR","text":"A module encapsulating the experimental driver code for the DCCR project.\n\nImports\n\nThe following names are imported by the package as dependencies:\n\nAdaptiveResonance\nBase\nColorSchemes\nCore\nDataStructures\nDocStringExtensions\nDrWatson\nJSON\nNumericalTypeAliases\nPkg\nProgressMeter\nPyCall\nReexport\n\nExports\n\nThe following names are exported and available when using the package:\n\nDCCR_VERSION\n\n\n\n\n\n","category":"module"},{"location":"man/full-index/#index-functions-docs","page":"Index","title":"Functions","text":"","category":"section"},{"location":"man/full-index/","page":"Index","title":"Index","text":"Modules = [DCCR]\nPrivate = false\nOrder = [:function]","category":"page"},{"location":"man/full-index/#index-types-docs","page":"Index","title":"Types","text":"","category":"section"},{"location":"man/full-index/","page":"Index","title":"Index","text":"Modules = [DCCR]\nPrivate = false\nOrder = [:type]","category":"page"},{"location":"man/full-index/#index-constants-docs","page":"Index","title":"Constants","text":"","category":"section"},{"location":"man/full-index/","page":"Index","title":"Index","text":"Modules = [DCCR]\nPrivate = false\nOrder = [:constant]","category":"page"},{"location":"man/full-index/#DCCR.DCCR_VERSION","page":"Index","title":"DCCR.DCCR_VERSION","text":"A constant that contains the version of the installed DCCR package.\n\nThis value is computed at compile time, so it may be used to programmatically verify the version of OAR that is installed in case a compat entry in your Project.toml is missing or otherwise incorrect.\n\n\n\n\n\n","category":"constant"},{"location":"examples/tutorials/julia/#julia","page":"Julia Tutorial","title":"Julia Tutorial","text":"","category":"section"},{"location":"examples/tutorials/julia/","page":"Julia Tutorial","title":"Julia Tutorial","text":"(Image: Source code) (Image: notebook) (Image: compat) (Image: Author) (Image: Update time)","category":"page"},{"location":"examples/tutorials/julia/#Overview","page":"Julia Tutorial","title":"Overview","text":"","category":"section"},{"location":"examples/tutorials/julia/","page":"Julia Tutorial","title":"Julia Tutorial","text":"TODO","category":"page"},{"location":"examples/tutorials/julia/#Setup","page":"Julia Tutorial","title":"Setup","text":"","category":"section"},{"location":"examples/tutorials/julia/","page":"Julia Tutorial","title":"Julia Tutorial","text":"First, we load some dependencies:","category":"page"},{"location":"examples/tutorials/julia/","page":"Julia Tutorial","title":"Julia Tutorial","text":"using Pkg","category":"page"},{"location":"examples/tutorials/julia/","page":"Julia Tutorial","title":"Julia Tutorial","text":"","category":"page"},{"location":"examples/tutorials/julia/","page":"Julia Tutorial","title":"Julia Tutorial","text":"This page was generated using DemoCards.jl and Literate.jl.","category":"page"},{"location":"examples/#examples","page":"Examples","title":"Examples","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"This section contains some examples using the DCCR package with topics ranging from how to the internals of package work to practical examples on different datasets.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"These examples are separated into the following sections:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"Tutorials: basic Julia examples that also include how low-level routines work in this package.\nExperiments: how to run experiments in the package.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"","category":"page"},{"location":"examples/#Tutorials","page":"Examples","title":"Tutorials","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"These examples demonstrate some low-level usage of the Julia programming language and subroutines of the DCCR project itself.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"<div class=\"grid-card-section\">","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"<div class=\"card grid-card\">\n<div class=\"grid-card-cover\">\n<div class=\"grid-card-description\">","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"This demo provides a quick example of how to run a Julia script.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"</div>","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"(Image: card-cover-image)","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"</div>\n<div class=\"grid-card-text\">","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"Julia Tutorial","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"</div>\n</div>","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"</div>","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"","category":"page"},{"location":"examples/#examples-attribution","page":"Examples","title":"Attribution","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"Icons used for the covers of these demo cards are attributed to the following sites:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"Lab icons created by Prosymbols - Flaticon (experiment_748518)\nIris icons created by Freepik - Flaticon (iris_4139395)\nGrammar icons created by Freepik - Flaticon (grammar_6749514)","category":"page"},{"location":"man/guide/#Package-Guide","page":"Guide","title":"Package Guide","text":"","category":"section"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"To work with the DCCR project, you should know:","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"How to install the package","category":"page"},{"location":"man/guide/#installation","page":"Guide","title":"Installation","text":"","category":"section"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"Because it is an experimental research repository, the DCCR package is not registered on JuliaHub. To set Julia component the project up, you must:","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"Download a Julia distribution and install it on your system\nGit clone this repository or download a zip.\nRun julia within the top of the DCCR directory, and run the following commands to instantiate the package:","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"julia> ]\n(@v1.9) pkg> activate .\n(DCCR) pkg> instantiate","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"This will download all of the dependencies of the project and precompile where possible.","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"note: Note\nThis project is still under development, so detailed usage guides beyond installation have not yet been written about the package's functionality. Please see the other sections of this documentation for examples, definition indices, and more.","category":"page"},{"location":"","page":"Home","title":"Home","text":"DocTestSetup = quote\n    using OAR, Dates\nend","category":"page"},{"location":"","page":"Home","title":"Home","text":"<img src=\"assets/logo.png\" width=\"300\">","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"These pages serve as the official documentation for the DCCR (Deep Clustering Context Recognition) project.","category":"page"},{"location":"#Manual-Outline","page":"Home","title":"Manual Outline","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This documentation is split into the following sections:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Pages = [\n    \"man/guide.md\",\n    \"../examples/index.md\",\n    \"man/contributing.md\",\n    \"man/full-index.md\",\n    \"man/dev-index.md\",\n]\nDepth = 1","category":"page"},{"location":"","page":"Home","title":"Home","text":"The Package Guide provides a tutorial to the full usage of the package, while Examples gives sample workflows with the various experiments of the project.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The Contributing section outlines how to contribute to the project. The Index enumerates all public types, functions, and other components with docstrings, whereas internals are listed in the Developer's Index.","category":"page"},{"location":"#About-These-Docs","page":"Home","title":"About These Docs","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Though several different programming languages are used throughout the project, these docs are built around the Julia component of the project using the Documenter.jl package.","category":"page"},{"location":"#Documentation-Build","page":"Home","title":"Documentation Build","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This documentation was built using Documenter.jl with the following version and OS:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using DCCR, Dates # hide\nprintln(\"DCCR v$(DCCR_VERSION) docs built $(Dates.now()) with Julia $(VERSION) on $(Sys.KERNEL)\") # hide","category":"page"}]
}