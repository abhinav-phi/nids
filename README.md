├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_training.ipynb
├── src/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── alerts.py
│   │   │   ├── predict.py
│   │   │   └── stats.py
│   │   ├── __init__.py
│   │   ├── database.py
│   │   ├── main.py
│   │   ├── models.py
│   │   └── schemas.py
│   ├── capture/
│   │   ├── __init__.py
│   │   └── sniffer.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── extractor.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── train.py
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── sim_bruteforce.py
│   │   ├── sim_ddos.py
│   │   ├── sim_mixed.py
│   │   └── sim_portscan.py
│   └── __init__.py
├── tests/
│   ├── test_api.py
│   └── test_extractor.py
├── label_encoder.pkl
├── LICENSE
├── model.pkl
├── package-lock.json
├── README.md
├── requirements.txt
└── scaler.pkl