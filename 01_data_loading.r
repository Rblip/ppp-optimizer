# ==============================================================================
#                    PANEL DATA CONSTRUCTION FROM YAHOO FINANCE
# ==============================================================================
# Author: Peter Paul Dimke
# Date:   January 2026
# Course: Financial Economics for Governance
# Purpose: Download stock data and create a clean panel dataset for PPP analysis
#
#                            All rights reserved.
# ==============================================================================
# What is Panel Data?
# -------------------
# Panel data (also called longitudinal data) has two dimensions:
#   1. Cross-sectional: Multiple entities (stocks/firms) observed
#   2. Time-series: Each entity observed over multiple time periods
#
# Structure: Each row = one stock on one date
#            Columns = identifier (ticker), date, and characteristics
#
# Example:
#   ticker  | date       | price  | volume   | momentum
#   --------|------------|--------|----------|----------
#   AAPL    | 2020-01-02 | 75.00  | 1000000  | 0.15
#   AAPL    | 2020-01-03 | 75.50  | 1200000  | 0.16
#   MSFT    | 2020-01-02 | 160.00 | 800000   | 0.10
#   MSFT    | 2020-01-03 | 161.00 | 850000   | 0.11
#
# ==============================================================================

# ==============================================================================
# STEP 0: LOAD REQUIRED PACKAGES
# ==============================================================================

library(quantmod)   # For downloading data from Yahoo Finance
library(data.table) # Fast data manipulation
library(dplyr)      # Data wrangling with clear syntax
library(zoo)        # Rolling window functions

# Suppress warnings about dplyr::lag conflicts
options(xts.warn_dplyr_breaks_lag = FALSE)

cat("=" |> rep(70) |> paste(collapse = ""), "\n")
cat("  PANEL DATA CONSTRUCTION FROM YAHOO FINANCE\n")
cat("=" |> rep(70) |> paste(collapse = ""), "\n\n")

# ==============================================================================
# STEP 1: DEFINE THE STOCK UNIVERSE - S&P 500
# ==============================================================================
# 
# The S&P 500 is a market-cap weighted index of 500 large US companies.
# It represents approximately 80% of the US equity market capitalization.
#
# Note: The actual S&P 500 constituents change over time (survivorship bias).
# For a production system, you would use point-in-time constituent data.
# Here we use current constituents for simplicity.
#
# ==============================================================================

# S&P 500 constituents (as of late 2025)
# Source: Wikipedia / S&P Dow Jones Indices
tickers <- c(
  # Information Technology
  "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "CSCO", "ACN", "AMD", "ADBE",
  "IBM", "TXN", "QCOM", "INTU", "AMAT", "NOW", "PANW", "ADI", "MU", "LRCX",
  "KLAC", "SNPS", "CDNS", "FTNT", "MCHP", "HPQ", "HPE", "KEYS", "ANSS", "ON",
  "NXPI", "CTSH", "GLW", "CDW", "MPWR", "FSLR", "TYL", "ZBRA", "NTAP", "SWKS",
  "JNPR", "PTC", "EPAM", "AKAM", "QRVO", "ENPH", "SEDG", "TER", "GEN", "FFIV",
  
  # Health Care
  "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "PFE", "DHR", "AMGN",
  "BMY", "CVS", "ELV", "MDT", "ISRG", "GILD", "CI", "SYK", "REGN", "VRTX",
  "ZTS", "BSX", "BDX", "MCK", "HUM", "EW", "A", "IDXX", "IQV", "CNC",
  "MTD", "DXCM", "CAH", "RMD", "ILMN", "HOLX", "ALGN", "BAX", "ZBH", "COO",
  "MOH", "BIIB", "WAT", "VTRS", "CRL", "LH", "DGX", "TECH", "HSIC", "XRAY",
  
  # Financials
  "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "BLK", "AXP",
  "C", "SCHW", "MMC", "PGR", "CB", "ICE", "CME", "AON", "MCO", "PNC",
  "USB", "TFC", "AIG", "MET", "AFL", "AJG", "TRV", "ALL", "PRU", "MSCI",
  "BK", "COF", "AMP", "STT", "TROW", "FITB", "NDAQ", "HBAN", "MTB", "RJF",
  "CINF", "CBOE", "KEY", "RF", "CFG", "NTRS", "SBNY", "FRC", "WRB", "L",
  
  # Consumer Discretionary
  "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "ORLY",
  "MAR", "CMG", "AZO", "GM", "F", "DHI", "ROST", "YUM", "HLT", "LVS",
  "EBAY", "DRI", "LEN", "GRMN", "BBY", "ULTA", "POOL", "PHM", "NVR", "TSCO",
  "WSM", "TPR", "APTV", "CCL", "WYNN", "MGM", "BWA", "RCL", "CZR", "DPZ",
  "EXPE", "KMX", "WHR", "HAS", "MHK", "NWL", "PVH", "RL", "VFC", "GPS",
  
  # Communication Services
  "GOOGL", "GOOG", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS", "CHTR",
  "EA", "ATVI", "WBD", "OMC", "IPG", "TTWO", "MTCH", "LUMN", "FOXA", "FOX",
  "PARA", "NWS", "NWSA", "DISH", "LYV",
  
  # Consumer Staples
  "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "EL",
  "KMB", "GIS", "KHC", "STZ", "ADM", "SYY", "HSY", "K", "KR", "MKC",
  "TSN", "CAG", "HRL", "CPB", "SJM", "CLX", "CHD", "BG", "TAP", "WBA",
  
  # Industrials
  "CAT", "RTX", "UNP", "HON", "UPS", "DE", "BA", "LMT", "GE", "ADP",
  "ETN", "WM", "ITW", "EMR", "CSX", "NSC", "PH", "GD", "NOC", "CTAS",
  "CARR", "TT", "PCAR", "JCI", "CPRT", "OTIS", "FAST", "VRSK", "RSG", "AME",
  "FDX", "LHX", "PWR", "CMI", "ROK", "IR", "DOV", "ODFL", "GWW", "EFX",
  "SWK", "XYL", "WAB", "J", "FTV", "LDOS", "TDG", "HWM", "IEX", "HII",
  
  # Energy
  "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "PXD",
  "WMB", "HES", "DVN", "HAL", "KMI", "FANG", "BKR", "CTRA", "OKE", "TRGP",
  "MRO", "APA",
  
  # Utilities
  "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "PCG", "ED",
  "WEC", "PEG", "AWK", "ES", "EIX", "DTE", "ETR", "FE", "AEE", "PPL",
  "CMS", "CNP", "EVRG", "ATO", "NI", "LNT", "PNW", "NRG",
  
  # Real Estate
  "AMT", "PLD", "EQIX", "CCI", "PSA", "O", "SPG", "WELL", "DLR", "VICI",
  "AVB", "EQR", "SBAC", "WY", "ARE", "VTR", "ESS", "MAA", "EXR", "UDR",
  "CBRE", "INVH", "IRM", "CPT", "HST", "KIM", "REG", "BXP", "FRT", "PEAK",
  
  # Materials
  "LIN", "APD", "SHW", "FCX", "ECL", "NUE", "NEM", "VMC", "MLM", "DOW",
  "DD", "PPG", "CTVA", "ALB", "IFF", "LYB", "CF", "FMC", "MOS", "CE",
  "EMN", "BALL", "PKG", "IP", "AVY", "SEE", "WRK", "AMCR"
)

# Remove any duplicates
tickers <- unique(tickers)

cat("Stock Universe: S&P 500\n")
cat("  Number of stocks:", length(tickers), "\n")
cat("  (Note: Some tickers may fail to download due to delistings/changes)\n\n")

# ==============================================================================
# STEP 2: SET THE TIME PERIOD
# ==============================================================================
#
# Yahoo Finance provides data back to:
#   - Major stocks: Often back to 1980s or IPO date
#   - Adjusted prices: Account for splits and dividends
#
# For portfolio optimization, we need sufficient history to:
#   - Calculate long-horizon characteristics (e.g., 12-month momentum)
#   - Have enough time periods for robust optimization
#
# ==============================================================================

start_date <- "2015-01-01"  # 10+ years of data
end_date   <- Sys.Date()    # Today

cat("Time Period:\n")
cat("  Start date:", start_date, "\n")
cat("  End date:  ", as.character(end_date), "\n\n")

# ==============================================================================
# STEP 3: DOWNLOAD DATA FROM YAHOO FINANCE
# ==============================================================================
#
# Yahoo Finance provides OHLCV data:
#   O = Open price (first trade of the day)
#   H = High price (highest trade of the day)
#   L = Low price (lowest trade of the day)
#   C = Close price (last trade of the day)
#   V = Volume (number of shares traded)
#   
# Plus: Adjusted Close = Close price adjusted for splits and dividends
#       This is what we use for return calculations!
#
# ==============================================================================

cat("Downloading data from Yahoo Finance...\n")
cat("-" |> rep(50) |> paste(collapse = ""), "\n")

# Download each stock and store in a list
prices_list <- list()

for (ticker in tickers) {
  result <- tryCatch({
    # getSymbols downloads OHLCV data as an xts (time-series) object
    data <- getSymbols(
      ticker, 
      src = "yahoo",           # Data source
      from = start_date,       # Start date
      to = end_date,           # End date
      auto.assign = FALSE      # Return the data, don't assign to global env
    )
    cat("  ✓", ticker, ":", nrow(data), "observations\n")
    data
  }, error = function(e) {
    cat("  ✗", ticker, ": FAILED -", e$message, "\n")
    NULL
  })
  
  if (!is.null(result)) {
    prices_list[[ticker]] <- result
  }
}

cat("-" |> rep(50) |> paste(collapse = ""), "\n")
cat("Successfully downloaded:", length(prices_list), "of", length(tickers), "stocks\n\n")

# ==============================================================================
# STEP 4: CONVERT TO PANEL DATA FORMAT
# ==============================================================================
#
# Yahoo Finance returns data in "wide" format (one xts object per stock).
# We need to convert to "long" panel format:
#   - Each row = one stock-date observation
#   - Columns = ticker, date, and all variables
#
# This is the standard format for cross-sectional asset pricing research.
#
# ==============================================================================

cat("Converting to panel data format...\n")

# Function to extract all OHLCV columns from an xts object
extract_ohlcv <- function(xts_data, ticker) {
  data.frame(
    ticker   = ticker,
    date     = index(xts_data),
    open     = as.numeric(Op(xts_data)),    # Op() extracts Open
    high     = as.numeric(Hi(xts_data)),    # Hi() extracts High
    low      = as.numeric(Lo(xts_data)),    # Lo() extracts Low
    close    = as.numeric(Cl(xts_data)),    # Cl() extracts Close
    volume   = as.numeric(Vo(xts_data)),    # Vo() extracts Volume
    adjusted = as.numeric(Ad(xts_data))     # Ad() extracts Adjusted Close
  )
}

# Apply to all stocks and combine into one data frame
panel_list <- mapply(
  extract_ohlcv, 
  prices_list, 
  names(prices_list), 
  SIMPLIFY = FALSE
)

# Combine all stock data frames into one panel
panel <- rbindlist(panel_list)

cat("  Raw panel dimensions:", nrow(panel), "rows x", ncol(panel), "columns\n\n")

# ==============================================================================
# STEP 5: CALCULATE DERIVED CHARACTERISTICS
# ==============================================================================
#
# From raw OHLCV data, we can derive many characteristics used in asset pricing:
#
# MOMENTUM FACTORS:
#   - Past returns predict future returns (Jegadeesh & Titman, 1993)
#   - Typically 12-month return, skipping the most recent month
#
# VOLATILITY FACTORS:
#   - Low volatility stocks tend to outperform (Ang et al., 2006)
#   - Measured as standard deviation of returns
#
# VOLUME/LIQUIDITY FACTORS:
#   - Illiquid stocks earn a premium (Amihud, 2002)
#
# TECHNICAL FACTORS:
#   - Price relative to moving averages
#   - Distance from 52-week high
#
# ==============================================================================

cat("Calculating derived characteristics (MONTHLY frequency)...\n")
cat("-" |> rep(50) |> paste(collapse = ""), "\n")

panel <- panel %>%
  # Sort by ticker and date (essential for lag/lead operations)
  arrange(ticker, date) %>%
  # Group by ticker to calculate within-stock time series
  group_by(ticker) %>%
  mutate(
    # =========================================================================
    # RETURNS (daily, needed for other calculations)
    # =========================================================================
    ret = (adjusted - lag(adjusted)) / lag(adjusted),
    
    # =========================================================================
    # MOMENTUM (12-month, skip last month)
    # =========================================================================
    # Classic Jegadeesh-Titman momentum signal
    # Return from t-252 to t-21 (12 months ago to 1 month ago)
    # Skipping the last month avoids short-term reversal contamination
    mom_12m = (lag(adjusted, 21) / lag(adjusted, 252)) - 1,
    
    # =========================================================================
    # VOLATILITY (monthly = 21 trading days)
    # =========================================================================
    # Realized volatility over past month
    # Low volatility stocks tend to outperform (Ang et al., 2006)
    vol_1m = rollapply(ret, width = 21, FUN = sd, fill = NA, align = "right"),
    
    # =========================================================================
    # ILLIQUIDITY - AMIHUD MEASURE
    # =========================================================================
    # Amihud (2002): Average of |return| / dollar volume
    # Measures price impact per dollar traded
    # Higher = more illiquid = harder to trade without moving price
    # 
    # Formula: ILLIQ = (1/D) * Σ |r_d| / (P_d * V_d)
    # Where D = number of days, r = return, P = price, V = volume
    #
    dollar_vol = close * volume,
    abs_ret_over_dvol = abs(ret) / dollar_vol,
    # Average over past month (21 days), scaled by 1e6 for readability
    illiq_1m = rollapply(abs_ret_over_dvol, width = 21, FUN = mean, 
                         fill = NA, align = "right") * 1e6,
    
    # =========================================================================
    # PRICE RELATIVE TO 12-MONTH HIGH
    # =========================================================================
    # George & Hwang (2004): Stocks near 52-week high continue to rise
    # Ratio of current price to highest price over past 252 trading days
    # Range: 0 to 1 (1 = at 52-week high)
    high_12m = rollapply(high, width = 252, FUN = max, fill = NA, align = "right"),
    price_to_high_12m = close / high_12m
    
  ) %>%
  ungroup()

cat("  ✓ Momentum: 12-month return (skip last month)\n")
cat("  ✓ Volatility: 1-month realized volatility\n")
cat("  ✓ Illiquidity: Amihud measure (|return| / dollar volume)\n")
cat("  ✓ Price level: Price relative to 12-month high\n")
cat("-" |> rep(50) |> paste(collapse = ""), "\n\n")

# ==============================================================================
# STEP 6: CONVERT TO MONTHLY FREQUENCY
# ==============================================================================
#
# MONTHLY REBALANCING
# -------------------
# Academic practice: Form portfolios at the END of each month using
# characteristics known at that time, then hold for one month.
#
# Why monthly?
#   - Reduces transaction costs (12 rebalances/year vs 252)
#   - Aligns with most factor research (Fama-French, momentum)
#   - Fundamental data (earnings, book value) updates quarterly
#
# Implementation:
#   1. Identify last trading day of each month
#   2. Use characteristics from that day
#   3. Calculate return from month-end t to month-end t+1
#
# ==============================================================================

cat("Converting to monthly frequency...\n")
cat("-" |> rep(50) |> paste(collapse = ""), "\n")

# Add year-month identifier
panel <- panel %>%
  mutate(year_month = format(date, "%Y-%m"))

# Find the last trading day of each month for each stock
month_ends <- panel %>%
  group_by(ticker, year_month) %>%
  filter(date == max(date)) %>%
  ungroup()

cat("  Month-end observations:", format(nrow(month_ends), big.mark = ","), "\n")

# Calculate MONTHLY forward returns
# This is the return from this month-end to next month-end
month_ends <- month_ends %>%
  arrange(ticker, date) %>%
  group_by(ticker) %>%
  mutate(
    # Next month's adjusted price
    adjusted_next = lead(adjusted),
    # Monthly forward return (what we earn by holding for 1 month)
    fwd_ret = (adjusted_next - adjusted) / adjusted
  ) %>%
  ungroup()

cat("  ✓ Monthly forward returns calculated\n")
cat("-" |> rep(50) |> paste(collapse = ""), "\n\n")

# Use month_ends as our panel from now on
panel <- month_ends

# ==============================================================================
# STEP 7: DATA QUALITY CHECKS
# ==============================================================================
#
# Panel data requires careful quality control:
#   1. Check for gaps in the time series
#   2. Remove stocks with too few observations
#   3. Handle missing values appropriately
#
# ==============================================================================

cat("Performing data quality checks...\n")
cat("-" |> rep(50) |> paste(collapse = ""), "\n")

# Check for date gaps within each stock
panel <- panel %>%
  arrange(ticker, date) %>%
  group_by(ticker) %>%
  mutate(
    date_lag  = lag(date),
    date_gap  = as.integer(date - date_lag)  # Gap in calendar days
  ) %>%
  ungroup()

# Summary of date gaps
gap_summary <- panel %>%
  filter(!is.na(date_gap)) %>%
  summarise(
    min_gap = min(date_gap),
    median_gap = median(date_gap),
    max_gap = max(date_gap),
    pct_large_gaps = mean(date_gap > 45) * 100  # % of gaps > 45 days (missed month)
  )

cat("  Date gap analysis (monthly):\n")
cat("    Minimum gap:", gap_summary$min_gap, "days\n")
cat("    Median gap: ", gap_summary$median_gap, "days (should be ~30)\n")
cat("    Maximum gap:", gap_summary$max_gap, "days\n")
cat("    % gaps > 45 days:", round(gap_summary$pct_large_gaps, 2), "%\n\n")

# Count observations per stock (in months)
obs_per_stock <- panel %>%
  group_by(ticker) %>%
  summarise(n_obs = n(), .groups = "drop") %>%
  arrange(n_obs)

cat("  Months per stock:\n")
cat("    Minimum:", min(obs_per_stock$n_obs), "\n")
cat("    Maximum:", max(obs_per_stock$n_obs), "\n")
cat("    Mean:   ", round(mean(obs_per_stock$n_obs)), "\n\n")

# ==============================================================================
# STEP 8: CREATE FINAL ANALYSIS DATASET
# ==============================================================================
#
# For PPP optimization, we need complete cases:
#   - All characteristics must be non-missing
#   - Forward returns must be available (not the last observation)
#
# ==============================================================================

# Define the characteristics we will use (4 factors, monthly frequency)
# 1. mom_12m        = Momentum (12-month return, skip last month)
# 2. vol_1m         = Volatility (1-month realized volatility)
# 3. illiq_1m       = Illiquidity (Amihud: |return| / dollar volume)
# 4. price_to_high_12m = Price relative to 12-month high
characteristics <- c("mom_12m", "vol_1m", "illiq_1m", "price_to_high_12m")

# Filter to complete cases
panel_clean <- panel %>%
  filter(
    !is.na(fwd_ret),   # Need forward return for optimization
    if_all(all_of(characteristics), ~ !is.na(.))  # All characteristics present
  )

cat("Final dataset (MONTHLY):\n")
cat("  Observations:", format(nrow(panel_clean), big.mark = ","), "\n")
cat("  Stocks:      ", n_distinct(panel_clean$ticker), "\n")
cat("  Date range:  ", as.character(min(panel_clean$date)), "to", 
    as.character(max(panel_clean$date)), "\n")
cat("  Months:      ", n_distinct(panel_clean$year_month), "\n")
cat("  Frequency:    MONTHLY (end-of-month rebalancing)\n\n")

# ==============================================================================
# STEP 9: EXAMINE THE PANEL STRUCTURE
# ==============================================================================

cat("Panel structure (first 10 rows):\n")
cat("-" |> rep(70) |> paste(collapse = ""), "\n")
print(head(panel_clean[, c("ticker", "date", "year_month", "adjusted", "fwd_ret", 
                           "mom_12m", "vol_1m", "illiq_1m", "price_to_high_12m")], 10))
cat("\n")

# Cross-sectional view: How many stocks per month?
stocks_per_month <- panel_clean %>%
  group_by(year_month) %>%
  summarise(n_stocks = n(), .groups = "drop")

cat("Stocks per month:\n")
cat("  Minimum:", min(stocks_per_month$n_stocks), "\n")
cat("  Maximum:", max(stocks_per_month$n_stocks), "\n")
cat("  Mean:   ", round(mean(stocks_per_month$n_stocks), 1), "\n\n")

# ==============================================================================
# STEP 10: SAVE THE PANEL DATASET
# ==============================================================================

output_file <- "yf_panel_data.csv"
fwrite(panel_clean, output_file)

cat("=" |> rep(70) |> paste(collapse = ""), "\n")
cat("  Panel data saved to:", output_file, "\n")
cat("  File size:", round(file.size(output_file) / 1024^2, 2), "MB\n")
cat("=" |> rep(70) |> paste(collapse = ""), "\n\n")

cat("Available characteristics for optimization:\n")
for (char in characteristics) {
  cat("  •", char, "\n")
}
cat("\nNext step: Run 02_ppp_optimization.r\n")
