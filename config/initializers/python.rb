# frozen_string_literal: true

require 'pycall'
require 'pandas'
require 'numpy'
require 'matplotlib'
require 'matplotlib/pyplot'

SciPy = PyCall.import_module('scipy')
SciPySpecial = PyCall.import_module('scipy.special')
ScipyStats = PyCall.import_module('scipy.stats')
