# Training Pipeline Analysis

## Experiment Summary

**Experiment**: `exp_synthetic_test`  
**Date**: 2025-11-24  
**Data**: Synthetic SPY intraday (generated from daily Stooq data)  
**Training Steps**: 5,000 (2 epochs × 2,500 steps)  
**Model**: DQN (MlpPolicy)

---

## Training Performance

### Validation Metrics (Per Epoch)

| Epoch | Timesteps | Val Return | Val Sharpe | Best Model |
|-------|-----------|------------|------------|------------|
| 1     | 2,500     | **-3.63%** | -4.58      | ✓ Saved    |
| 2     | 5,000     | -4.09%     | -5.22      |            |

**Key Observations:**
- Best model achieved at Epoch 1 with -3.63% validation return
- Performance degraded in Epoch 2 (overfitting or insufficient training)
- Negative Sharpe ratios indicate poor risk-adjusted returns

---

## Test Set Performance

### RL Strategy vs Buy-and-Hold

| Metric | RL Strategy | Buy-and-Hold | Difference |
|--------|-------------|--------------|------------|
| **Final Equity** | 0.9840 | 1.0060 | -0.0220 |
| **Total Return** | **-1.60%** | **+0.60%** | **-2.20%** |
| **Max Drawdown** | -2.18% | -2.11% | -0.07% |
| **Sharpe Ratio** | -2.76 | +0.82 | -3.58 |

**Key Findings:**
- ❌ **RL strategy underperformed buy-and-hold by 2.20%**
- ❌ **Negative Sharpe ratio** indicates poor risk-adjusted returns
- ⚠️ **Similar drawdowns** but RL had worse recovery

### Trading Statistics

| Metric | Value |
|--------|-------|
| **Total Trades** | 95 |
| **Win Rate** | 27.4% |
| **Profit Factor** | 0.58 |
| **Avg Win** | +0.085% |
| **Avg Loss** | -0.056% |
| **Avg Trade Duration** | 2.5 steps |

**Key Observations:**
- ⚠️ **Low win rate** (27.4%) - strategy loses more often than wins
- ⚠️ **Profit factor < 1.0** (0.58) - losses exceed wins
- ⚠️ **Very short trades** (2.5 steps) - high turnover, transaction costs matter

### Action Distribution

| Action | Count | Percentage |
|--------|-------|------------|
| **Flat (0)** | 128 | 34.6% |
| **Long (1)** | 206 | 55.7% |
| **Short (2)** | 36 | 9.7% |

**Key Observations:**
- 📊 **Heavy long bias** (55.7% long vs 9.7% short)
- 📊 **Low short activity** - model rarely takes short positions
- 📊 **Moderate flat periods** (34.6%) - some position management

---

## Analysis & Insights

### 1. **Underperformance vs Buy-and-Hold**

The RL strategy lost **-1.60%** while buy-and-hold gained **+0.60%**, resulting in a **-2.20% underperformance**. This suggests:

- **Transaction costs matter**: With 95 trades over 370 steps, frequent trading likely eroded returns
- **Model not learning profitable patterns**: The agent hasn't identified edge in the synthetic data
- **Limited training**: 5,000 steps may be insufficient for DQN to learn complex trading patterns

### 2. **Poor Risk-Adjusted Returns**

**Sharpe Ratio: -2.76** (vs B&H: +0.82)

- Negative Sharpe indicates returns are below risk-free rate (or negative) with high volatility
- The strategy is taking risk without commensurate reward
- High volatility relative to mean return suggests unstable policy

### 3. **Trading Behavior**

- **Low win rate (27.4%)** but **profit factor 0.58** suggests:
  - When the model wins, it wins slightly more than it loses
  - But it loses too frequently to be profitable
  - Average win (+0.085%) > Average loss (-0.056%) is positive, but frequency matters

- **Short trade duration (2.5 steps)** indicates:
  - High-frequency trading behavior
  - May be reacting to noise rather than signal
  - Transaction costs compound quickly

### 4. **Action Distribution Bias**

- **55.7% long positions** vs **9.7% short** suggests:
  - Model learned a bullish bias (possibly from training data)
  - Underutilization of shorting capability
  - May be missing opportunities in downtrends

---

## Limitations & Context

### Synthetic Data Caveats

⚠️ **This experiment used synthetic intraday data** generated from daily bars. This means:
- Not real market microstructure
- No true intraday volatility patterns
- Synthetic data may not reflect real trading dynamics

### Training Constraints

- **Limited training steps**: 5,000 steps is relatively small for DQN
- **Small dataset**: 2,051 training bars may be insufficient
- **Simple features**: Hand-crafted technical indicators may not capture all patterns

### Expected Behavior

For a **learning sandbox**, these results are **expected and acceptable**:
- The goal is pipeline quality, not trading performance
- Demonstrates proper train/val/test splits
- Shows evaluation framework working correctly
- Negative performance highlights areas for improvement

---

## Recommendations for Improvement

### 1. **Increase Training**
- Train for more timesteps (100K+)
- More epochs with early stopping
- Larger replay buffer

### 2. **Feature Engineering**
- Add more sophisticated indicators
- Consider learned feature representations
- Market regime indicators

### 3. **Hyperparameter Tuning**
- Learning rate scheduling
- Exploration strategy (epsilon decay)
- Reward shaping

### 4. **Algorithm Improvements**
- Try PPO or SAC (continuous actions)
- Add risk constraints
- Reward function modifications

### 5. **Real Data**
- Use actual intraday data when available
- Test on multiple market regimes
- Validate on out-of-sample periods

---

## Conclusion

The training pipeline executed successfully, demonstrating:
- ✅ Proper data loading and splitting
- ✅ Feature engineering and normalization
- ✅ DQN training with validation
- ✅ Comprehensive test evaluation
- ✅ Metrics and visualization generation

**Performance is poor but expected** for a learning sandbox with limited training and synthetic data. The framework is solid and ready for experimentation with:
- Real market data
- Extended training
- Algorithm improvements
- Feature enhancements

---

*Generated: 2025-11-24*

