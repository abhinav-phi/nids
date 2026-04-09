import joblib
s = joblib.load('scaler.pkl')
m = joblib.load('model.pkl')
print('Scaler expects:', s.n_features_in_)
print('Model expects:', m.n_features_in_)