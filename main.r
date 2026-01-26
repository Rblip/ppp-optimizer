# ======================================================================
#   PARAMETRIC PORTFOLIO POLICY - MAIN SCRIPT
# ======================================================================
#
#   This script runs the complete PPP analysis pipeline:
#     1. Data loading and panel construction (01_data_loading.r)
#     2. PPP optimization and backtesting (02_ppp_optimization.r)
#
#   Author: Peter Paul Dimke
#   Date:   January 2026
#
#                            All rights reserved.
# ==============================================================================
# Reference:
#   Brandt, M. W., Santa-Clara, P., & Valkanov, R. (2009).
#   "Parametric Portfolio Policies: Exploiting Characteristics in the 
#    Cross-Section of Equity Returns."
#   Review of Financial Studies, 22(9), 3411-3447.
#
# ==============================================================================

# Set working directory to script location (if running interactively)
# setwd("z:/ppp2")
source("01_data_loading.r")  # Load the data loading script
source("02_ppp_optimization.r")  # Load the PPP optimization script