from utils import data_preparation
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

df = data_preparation.load_data('/Users/cemkoymen/Desktop/hmm_garch/Data/DGS5.csv')
returns = data_preparation.calculate_returns(df)
volatility = data_preparation.calculate_volatility(returns)
labelling = data_preparation.classify_dates(volatility)
z_scored_labels = data_preparation.calculate_rolling_z_scores(labelling)
final_states = data_preparation.calculate_final_score(z_scored_labels)
transition_matrix = data_preparation.compute_transition_matrix(z_scored_labels)
print("Transition Matrix:")
print(transition_matrix)
hmm_model = data_preparation.train_hmm(z_scored_labels)
predicted_states = data_preparation.predict_states(hmm_model, z_scored_labels)
print(predicted_states[['State', 'Predicted_Yield_State']])
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['DGS5'], label='5-Year Yield')
plt.scatter(df.index, df['Predicted_Yield_State'], c=df['Predicted_Yield_State'], cmap='viridis', label='Hidden States')
plt.legend()
plt.title('Hidden Markov Model - Predicted Yield States')
plt.show()





