Sales Forecasting: A Comparative Analysis
This project is a deep dive into sales forecasting using machine learning models. The primary goal is to predict future sales with high accuracy by comparing the performance of two popular algorithms: Linear Regression and XGBoost.

Model Performance Metrics
The models were evaluated using three key regression metrics:


MAE (Mean Absolute Error): Measures the average absolute difference between predicted and actual sales. A lower value indicates higher accuracy.

RMSE (Root Mean Squared Error): Measures the square root of the average of the squared differences between predicted and actual sales. It penalizes larger errors more heavily. A lower value is better.

R² (R-squared): Represents the proportion of the variance for a dependent variable that's explained by the independent variables. A higher value (closer to 1) indicates a better fit.

```
Key Findings & Model Comparison
The results clearly indicate a superior performance from the XGBoost model.

Model                 MAE          RMSE            R²


Linear Regression    1471.59      3168.02        0.9730


XGBoost              1224.22      3062.33        0.9748
```
Conclusion
The XGBoost model is the clear winner for this sales forecasting task. It consistently outperforms the Linear Regression model across all three key metrics. With a lower MAE and RMSE, it provides more accurate predictions, and its higher R² score demonstrates that it captures a greater proportion of the sales variance. These results confirm that the XGBoost model is the ideal choice for a real-world application, offering a more robust and reliable sales forecast.
