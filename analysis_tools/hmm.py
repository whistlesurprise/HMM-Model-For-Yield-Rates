import pandas as pd 
from enum import Enum
from decimal import Decimal,getcontext
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
class FedPolicy(Enum):
    QE = 0.3
    QT = 0.7

# Yield Curve Shape Enum
class YieldCurveShape(Enum):
    INVERTED = -1
    FLAT = 0
    NORMAL = 1

# Business Cycle Enum
class BusinessCycle(Enum):
    EXPANSION = 0.3
    PEAK = 0.7
    RECESSION = 0.9
    RECOVERY = 0.5

# Crisis Indicator Enum
class CrisisIndicator(Enum):
    NORMAL = 0
    STRESS = 1

def load_data(path):
    """
    Load data from a CSV file, set 'observation_date' as the index, and ensure the index is in datetime format.
    Interpolates missing values in the first column using linear interpolation.
    """
    yields = pd.read_csv(path)
    yields['observation_date'] = pd.to_datetime(yields['observation_date'])  # Convert to datetime
    yields.set_index('observation_date', inplace=True)  # Set as index
    yields.iloc[:, 0] = yields.iloc[:, 0].interpolate(method='linear')  # Interpolate first column
    return yields

def calculate_returns(df):
    df['DGS5'] = df['DGS5'].apply(lambda x: Decimal((x)).quantize(Decimal('0.01')))
    df['change_in_yields_in_bps'] = (df['DGS5'].diff() * 100).fillna(0)
    return df

def calculate_volatility(df):
    df['Volatility'] = df['change_in_yields_in_bps'].rolling(window=21).std()
    return df

def classify_dates(df, 
                   policy_periods={
                       'PRE_COVID_QE': ('2019-12-26', '2020-03-15'),
                       'COVID_QE': ('2020-03-15', '2022-03-01'),
                       'QT1': ('2017-10-01', '2019-07-31'),
                       'QT2': ('2022-06-01', '2024-12-31'),
                   },
                   crisis_periods={
                       'COVID_Crisis': ('2020-02-19', '2020-04-30'),
                   },
                   yield_curve_shapes={
                       'Pre-COVID_Normal': ('2019-12-26', '2020-02-18'),
                       'COVID_Inversion': ('2020-02-19', '2020-04-30'),
                       'Post-COVID_Steepening': ('2020-05-01', '2021-12-31'),
                       'Flattening_Trend': ('2022-01-01', '2022-12-31'),
                       'QT_Flat_or_Inverted': ('2023-01-01', '2024-12-26'),
                   },
                    business_cycles = {
                    'Expansion': ('2019-12-26', '2020-02-18'),  # Pre-pandemic growth.
                    'Recession': ('2020-02-19', '2020-04-30'),  # COVID-19 sharp decline.
                    'Recovery': ('2020-05-01', '2024-12-26')    # Post-pandemic recovery phase.
}):
    def classify_date(date, scheme_dict):
        for scheme, (start, end) in scheme_dict.items():
            if pd.Timestamp(start) <= date <= pd.Timestamp(end):
                return scheme
        return 'Unknown'

    # Apply classification
    df['Policy_Scheme'] = df.index.to_series().apply(lambda x: classify_date(x, policy_periods))
    df['Crisis_Scheme'] = df.index.to_series().apply(lambda x: classify_date(x, crisis_periods))
    df['Yield_Curve_Shape'] = df.index.to_series().apply(lambda x: classify_date(x, yield_curve_shapes))
    df['Business_Cycles'] = df.index.to_series().apply(lambda x: classify_date(x, business_cycles))
    return df

def calculate_rolling_z_scores(df, window=21):
    """
    Calculate rolling z-scores for the Volatility column.
    """
    rolling_mean = df['Volatility'].rolling(window=window).mean()
    rolling_std = df['Volatility'].rolling(window=window).std()
    
    df['Rolling_Volatility_Z_Score'] = (df['Volatility'] - rolling_mean) / rolling_std
    
    return df

def calculate_final_score(df):
    # Statistical Score
    def get_statistical_score(z):
        if z < 0.5:
            return 0.3
        elif 0.5 <= z <= 1.5:
            return 0.6
        else:
            return 0.9

    # Market Score
    def get_market_score(policy, curve):
        policy_score = 0.5  # Default neutral
        if 'QE' in policy:
            policy_score = 0.2
        elif 'QT' in policy:
            policy_score = 0.7
        
        # Curve shape score
        curve_score = YieldCurveShape.NORMAL.value if 'Normal' in curve else (
            YieldCurveShape.FLAT.value if 'Flat' in curve else YieldCurveShape.INVERTED.value
        )
        
        return (policy_score + (curve_score + 1) / 2) / 2  # Normalize to 0-1

    # Economic Score
    def get_economic_score(business_cycle, crisis):
        cycle_score = 0.5  # Default recovery
        if 'Expansion' in business_cycle:
            cycle_score = BusinessCycle.EXPANSION.value
        elif 'Recession' in business_cycle:
            cycle_score = BusinessCycle.RECESSION.value
        elif 'Peak' in business_cycle:
            cycle_score = BusinessCycle.PEAK.value

        crisis_score = 1 if 'Stress' in crisis else 0
        return (cycle_score + crisis_score) / 2

    # Calculate final score for each row
    scores = []
    for index, row in df.iterrows():
        stat_score = get_statistical_score(row['Rolling_Volatility_Z_Score'])
        market_score = get_market_score(row['Policy_Scheme'], row['Yield_Curve_Shape'])
        economic_score = get_economic_score(row['Business_Cycles'], row['Crisis_Scheme'])

        # Weighted formula
        final_score = (0.6 * stat_score) + (0.25 * market_score) + (0.15 * economic_score)

        # Adjustments for High/Low overrides
        if ('QT' in row['Policy_Scheme'] and 'Inverted' in row['Yield_Curve_Shape'] and 
            'Rate hiking' in row['Policy_Scheme']) or 'Stress' in row['Crisis_Scheme']:
            final_score = 1.0  # High
        elif ('QE' in row['Policy_Scheme'] and 'Normal' in row['Yield_Curve_Shape'] and 
              'Rate cutting' in row['Policy_Scheme'] and 'Expansion' in row['Business_Cycles']):
            final_score = 0.1  # Low

        scores.append(final_score)

    df['Final_Score'] = scores

    # Classify states
    def classify_state(score):
        if score < 0.4:
            return 'Low'
        elif 0.4 <= score <= 0.7:
            return 'Medium'
        else:
            return 'High'

    df['State'] = df['Final_Score'].apply(classify_state)

    return df

def compute_transition_matrix(df):
    """
    Compute the transition matrix for the hidden states.
    """
    # Calculate transition counts
    transition_counts = pd.crosstab(df['State'], df['State'].shift(-1), normalize='index')
    
    # Normalize the transition counts to ensure each row sums to 1
    transition_matrix = transition_counts.div(transition_counts.sum(axis=1), axis=0)
    
    return transition_matrix

def plot_yield_states(df):
    # Compute transition matrix
    transition_matrix = compute_transition_matrix(df)
    
    # Plot yield rates
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['DGS5'], label='Yield Rate', color='blue')
    
    # Annotate states with different colors
    colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    for state in df['State'].unique():
        state_dates = df[df['State'] == state].index
        plt.scatter(state_dates, df.loc[state_dates, 'DGS5'], label=state, color=colors[state], alpha=0.6)
    
    # Annotate transition probabilities
    for i, state in enumerate(transition_matrix.index):
        for j, next_state in enumerate(transition_matrix.columns):
            prob = transition_matrix.loc[state, next_state]
            plt.text(i, j, f'{prob:.2f}', ha='center', va='center', fontsize=12, color='black')
    
    plt.xlabel('Date')
    plt.ylabel('Yield Rate')
    plt.title('Yield Rates and States with Transition Probabilities')
    plt.legend()
    plt.show()











        
        
        








 





