Customer Churn with Logistic Regression
We are addressing the issue of customers leaving our land-line business for cable competitors. To understand who is leaving, we use a telecommunications dataset for predicting customer churn. This dataset contains customer information, making it straightforward to derive actionable insights. Retaining existing customers is typically more cost-effective than acquiring new ones, so our focus is on predicting which customers will stay with us.

The dataset includes:

Customers who left within the last month (Churn column)
Services each customer signed up for (phone, multiple lines, internet, online security, online backup, device protection, tech support, streaming TV and movies)
Customer account details (tenure, contract type, payment method, paperless billing status, monthly charges, total charges)
Demographic information (gender, age range, partners, dependents)
We start by loading and preprocessing the data, then we select relevant features for modeling. The Logistic Regression model from Scikit-learn is used to find parameters using different numerical optimizers. This model also supports regularization to address overfitting.

We fit our model with the training set and evaluate it using metrics like the Jaccard index, confusion matrix, classification report, and log loss to determine its performance in predicting customer churn. These insights help us develop targeted retention strategies to keep our customers.



