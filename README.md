> A Machine Learning project to **predict water suitability** for plants, bacteria, and warm/cold water fish in aquaponics systems.  
> Built with **Flask** for the web interface and **Scikit-learn** for the ML pipeline.

---

##Overview

This project takes **water parameters** (pH, Dissolved Oxygen, Temperature, Ammonia, Nitrite, Nitrate)  
and predicts if the water is suitable for:

- 🌱 Plants  
- 🧫 Bacteria  
- 🐟 Warm Water Fish  
- ❄️ Cold Water Fish  

It also provides **tips to improve water quality** if some organisms are mis


## Setup
pip install -r requirements.txt

## Train
python src/train_voting_classifier.py

## Serve
python src/app.py


