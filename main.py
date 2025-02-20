from analysis_tools import hmm_garch

def main():
    # Load data
    df = hmm_garch.load_data('/Users/cemkoymen/Desktop/hmm_garch/Data/DGS5.csv')
    
    # Calculate returns
    returns = hmm_garch.calculate_returns(df)
    
    # Calculate volatility
    volatility = hmm_garch.calculate_volatility(returns)
    
    # Classify dates based on volatility
    labelling = hmm_garch.classify_dates(volatility)
    
    # Calculate rolling z-scores
    z_scored_labels = hmm_garch.calculate_rolling_z_scores(labelling)
    
    # Calculate final states
    final_states = hmm_garch.calculate_final_score(z_scored_labels)
    
    # Compute transition matrix
    transition_matrix = hmm_garch.compute_transition_matrix(final_states)
    
    # Plot yield states
    hmm_garch.plot_yield_states(final_states)
    
    # Print transition matrix
    print("Transition Matrix:")
    print(transition_matrix)

if __name__ == "__main__":
    main()





