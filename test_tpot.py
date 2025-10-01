#!/usr/bin/env python3
"""Quick test of tpot_full_potential.py with reduced parameters."""
import sys
sys.path.insert(0, '/home/burak/Project_Nemuri/Beta')

# Temporarily modify config for quick test
import tpot_full_potential as tpot_script

# Override config for testing
tpot_script.TPOT_CONFIG['generations'] = 5
tpot_script.TPOT_CONFIG['population_size'] = 20
tpot_script.TPOT_CONFIG['cv'] = 3
tpot_script.N_BOOTSTRAP = 10
tpot_script.PERMUTATION_REPEATS = 5

print("Running TPOT with TEST configuration (fast mode)...")
print("  Generations: 5")
print("  Population: 20")
print("  CV: 3-fold")
print("  Bootstrap: 10 samples")
print("  Permutation: 5 repeats")
print()

tpot_script.main()
