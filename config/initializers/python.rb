# frozen_string_literal: true

require 'pycall'
require 'pandas'
require 'numpy'
require 'matplotlib'
require 'matplotlib/pyplot'
require 'matrix'

Matplotlib::Pyplot.switch_backend('Agg')

SciPy = PyCall.import_module('scipy')
SciPySpecial = PyCall.import_module('scipy.special')
ScipyStats = PyCall.import_module('scipy.stats')
StatModels = PyCall.import_module('statsmodels.api')
