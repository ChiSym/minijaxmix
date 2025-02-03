#!/bin/bash

python test.py # default test: run on several large benchmarks
# python test.py bernoulli # run inference on dataset of bernoulli noise
# python test.py data/medianHouseholdIncome # run inference on arbitrary custom dataset, in this case U.S. Census (ACS) median household income data