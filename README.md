## Project Structure

```text
├── notebooks/             
│   ├── 01_eda.ipynb        
│   └── 02_training.ipynb   
├── src/                    
│   ├── api/               
│   │   ├── routes/         
│   │   ├── database.py     
│   │   ├── main.py         
│   │   ├── models.py       
│   │   └── schemas.py      
│   ├── capture/            
│   │   └── sniffer.py      
│   ├── features/           
│   │   └── extractor.py    
│   ├── model/              
│   │   ├── evaluate.py     
│   │   ├── predict.py      
│   │   └── train.py        
│   └── simulation/         
│       ├── sim_bruteforce.py
│       ├── sim_ddos.py
│       ├── sim_mixed.py
│       └── sim_portscan.py
├── tests/                  
├── label_encoder.pkl       
├── model.pkl               
├── scaler.pkl              
├── requirements.txt        
└── README.md               
