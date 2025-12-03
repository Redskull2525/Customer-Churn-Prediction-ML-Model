🧾 1. Customer Information Columns
Column Name	Type	Description
State	Categorical	US state where the customer lives
Area code	Categorical	Customer’s regional phone area code
Phone	Categorical	Customer’s phone number (unique identifier, not useful for ML)
📞 2. Customer Account Behavior
Column Name	Type	Description
Account length	Numerical	Number of days the customer has been with the company
International plan	Categorical (Yes/No)	Indicates if customer has an international calling plan
Voice mail plan	Categorical (Yes/No)	Indicates if customer has voicemail service
Number vmail messages	Numerical	Number of voice messages stored
📈 3. Customer Usage Metrics
Local Calls / Charges
Column Name	Type	Description
Total day minutes	Numerical	Minutes used during daytime
Total day calls	Numerical	Number of calls made during daytime
Total day charge	Numerical	Charge for daytime calls
Total eve minutes	Numerical	Minutes used during evening
Total eve calls	Numerical	Number of evening calls
Total eve charge	Numerical	Evening calling charge
Total night minutes	Numerical	Minutes used during night
Total night calls	Numerical	Number of night calls
Total night charge	Numerical	Night-time calling charge
International Calls / Charges
Column Name	Type	Description
Total intl minutes	Numerical	International call minutes used
Total intl calls	Numerical	Total number of international calls
Total intl charge	Numerical	International call charge
🛠 4. Customer Service Interaction
Column Name	Type	Description
Customer service calls	Numerical	Number of times customer contacted support
🎯 5. Target Variable
Column Name	Type	Description
Churn	Categorical (Yes/No)	Whether the customer left the telecom company
